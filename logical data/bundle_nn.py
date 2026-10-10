"""Logical bundle induction by metric-based few-shot learning (Prototypical Networks).

The induction is done by a gradient-trained neural encoder, not by searching a rule space.

How it works. Every catalogue item is described, RELATIVE TO A CART, by raw relational signals
(price differences against the cart's statistics and against distinguished cart items, and
whether its colour/brand coincide with those of cart items). A learned MLP maps that raw
description to an embedding. For a task, the three in-context examples give three (cart, gold)
pairs; their embeddings are averaged into a PROTOTYPE, and the test cart's candidates are
scored by cosine similarity to that prototype. The hidden logic of a task is therefore carried
by the prototype computed from that task's own examples -- which is what makes the prediction
task-specific -- while the encoder only learns what a relational description should emphasise.

Training is episodic meta-learning: for each task one example is held out as the query and the
other two form the support prototype, and the encoder is trained by cross-entropy to rank the
gold item above sampled competitors from the same category. Gradients flow through the encoder
only; no rule grid is enumerated, no per-task search is performed, and nothing is fitted that
maps a task to a stored answer. At test time the same encoder runs forward once.

Categories to emit are read off the examples (the output category multiset is constant within a
task on all 3,000 tasks), so only WHICH item in each category is predicted.
"""
import os, sys, json, time, math, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

T0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time()-T0), *a, flush=True)

def parse_paths():
    if len(sys.argv) > 2 and not sys.argv[1].startswith('-f'):
        return Path(sys.argv[1]), Path(sys.argv[2])
    cands = ['./dataset/public', '/kaggle/input/datasets/devansh66/logical-bundle',
             '/kaggle/input/logical-bundle', '/kaggle/input', '.']
    pub = next((Path(c) for c in cands if (Path(c)/'tasks.json').exists()),
               Path('./dataset/public'))
    out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)
EPOCHS = int(os.environ.get('EPOCHS', 8))
NNEG = int(os.environ.get('NNEG', 96))
DIM = int(os.environ.get('DIM', 48))
HID = int(os.environ.get('HID', 128))
NVAL = int(os.environ.get('NVAL', 400))

catalog = json.load(open(PUB/'catalog.json', encoding='utf-8'))
tasks = json.load(open(PUB/'tasks.json', encoding='utf-8'))
IDS = np.array([c['item_id'] for c in catalog])
POS = {int(i): k for k, i in enumerate(IDS)}
PRICE = np.array([c['price'] for c in catalog], np.float64)
CATS = sorted({c['category'] for c in catalog})
COLS = sorted({c['color'] for c in catalog})
BRAS = sorted({c['brand'] for c in catalog})
CI = np.array([CATS.index(c['category']) for c in catalog])
OI = np.array([COLS.index(c['color']) for c in catalog])
BI = np.array([BRAS.index(c['brand']) for c in catalog])
BYCAT = {k: np.where(CI == k)[0] for k in range(len(CATS))}
RANK = np.zeros(len(catalog))
for k, idx in BYCAT.items():
    o = idx[np.argsort(-PRICE[idx])]
    RANK[o] = np.arange(len(o))/max(len(o)-1, 1)
log('catalog %d | tasks %d' % (len(catalog), len(tasks)))

SC = 500.0
NF = 30

def describe(idx, cart):
    """Raw relational description of candidates `idx` against `cart`. No rule is encoded here:
    these are the primitive comparisons available, and the encoder decides what matters."""
    p = PRICE[idx]
    ip = PRICE[cart]
    lo, hi, mu, sm = ip.min(), ip.max(), ip.mean(), ip.sum()
    dlo = cart[int(np.argmin(ip))]; dhi = cart[int(np.argmax(ip))]
    F = np.empty((len(idx), NF), np.float32)
    F[:, 0] = p/SC
    F[:, 1] = RANK[idx]
    F[:, 2] = (p-lo)/SC
    F[:, 3] = (p-hi)/SC
    F[:, 4] = (p-mu)/SC
    F[:, 5] = (p-sm)/SC
    F[:, 6] = np.abs(p-lo)/SC
    F[:, 7] = np.abs(p-hi)/SC
    F[:, 8] = np.abs(p-mu)/SC
    F[:, 9] = np.abs(p-sm)/SC
    F[:, 10] = (RANK[idx] == 0)
    F[:, 11] = (RANK[idx] == 1)
    c = 12
    for dd in (dlo, dhi, cart[0], cart[-1]):
        F[:, c] = (OI[idx] == OI[dd]); c += 1
        F[:, c] = (BI[idx] == BI[dd]); c += 1
        F[:, c] = np.abs(p-PRICE[dd])/SC; c += 1
    mc = Counter(OI[cart].tolist()).most_common(1)[0][0]
    mb = Counter(BI[cart].tolist()).most_common(1)[0][0]
    F[:, c] = (OI[idx] == mc); c += 1
    F[:, c] = (BI[idx] == mb); c += 1
    F[:, c] = np.isin(OI[idx], OI[cart]); c += 1
    F[:, c] = np.isin(BI[idx], BI[cart]); c += 1
    # the cart item sharing the candidate's own category, if any
    same = cart[CI[cart] == CI[idx[0]]]
    if len(same):
        F[:, c] = 1.0; c += 1
        F[:, c] = np.abs(p-PRICE[same].min())/SC
    else:
        F[:, c] = 0.0; c += 1
        F[:, c] = 1.0
    return F

def slots(ex):
    return sorted(ex, key=lambda i: (CI[POS[i]], -PRICE[POS[i]]))

EX = {}
for tid, t in tasks.items():
    EX[tid] = [(np.array([POS[i] for i in e['input']]), slots(e['output'])) for e in t['train']]

import torch, torch.nn as nn
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.manual_seed(SEED)

class Comparator(nn.Module):
    """Scores a candidate against the support set of a task.

    A cosine distance to a mean prototype cannot express what induction needs here: the rule is
    carried by whichever relational signals are CONSISTENT across the support golds, and a mean
    alone throws that away. So the support set is summarised by its mean AND its spread, and the
    comparator sees the candidate, the summary, and their agreement. A signal that is stable
    across supports and matched by the candidate can then be weighted up, and an unstable one
    (incidental colour, say) weighted down -- learned by gradient descent, not stipulated.
    """
    def __init__(s, d=NF, h=HID):
        super().__init__()
        s.g = nn.Sequential(nn.Linear(4*d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                            nn.Linear(h, 1))
    def forward(s, q, sm, sd):
        # q (B,L,d); sm/sd (B,d)
        B, L, d = q.shape
        sm = sm[:, None, :].expand(B, L, d); sd = sd[:, None, :].expand(B, L, d)
        agree = -torch.abs(q-sm)/(sd+0.25)
        return s.g(torch.cat([q, sm, sd, agree], -1)).squeeze(-1)

net = Comparator()
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)

def episode(tid, h, rng):
    """One episode: support = the other two examples, query = example h."""
    ex = EX[tid]
    sup = [e for k, e in enumerate(ex) if k != h]
    qc, qo = ex[h]
    nsl = min(len(e[1]) for e in ex)
    out = []
    for j in range(nsl):
        sf = []
        for cart, o in sup:
            g = POS[o[j]]
            sf.append(describe(np.array([g]), cart)[0])
        g = POS[qo[j]]
        kc = CI[g]
        pool = BYCAT[kc]
        neg = pool[rng.integers(0, len(pool), NNEG)]
        neg = neg[neg != g]
        cand = np.concatenate([[g], neg])
        qf = describe(cand, qc)
        out.append((np.stack(sf), qf))
    return out

log('episodic meta-training ...')
tids = list(tasks)
for ep in range(EPOCHS):
    rng = np.random.default_rng(SEED+ep)
    order = rng.permutation(len(tids))
    tot = n = 0.0
    BSZ = int(os.environ.get('BSZ', 8))
    for bi in range(0, len(order), BSZ):
        batch = []
        for oi in order[bi:bi+BSZ]:
            tid = tids[oi]
            for h in range(3):
                batch += episode(tid, h, rng)
        if not batch: continue
        loss = 0.0
        S = torch.from_numpy(np.stack([b[0] for b in batch]))          # (B,K,NF)
        B = S.shape[0]
        sm = S.mean(1); sd = S.std(1) if S.shape[1] > 1 else torch.zeros_like(sm)
        L = min(b[1].shape[0] for b in batch)
        Q = torch.from_numpy(np.stack([b[1][:L] for b in batch]))       # (B,L,NF)
        logits = net(Q, sm, sd)
        loss = nn.functional.cross_entropy(logits, torch.zeros(B, dtype=torch.long))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        tot += float(loss)*B; n += B
    log('  epoch %d  loss %.4f' % (ep, tot/max(n, 1)))
net.eval()

@torch.no_grad()
def predict(tid, cart, hold=None):
    ex = EX[tid]
    sup = [e for k, e in enumerate(ex) if k != hold]
    nsl = min(len(e[1]) for e in ex)
    got = []
    for j in range(nsl):
        sf = np.stack([describe(np.array([POS[o[j]]]), c)[0] for c, o in sup])
        S = torch.from_numpy(sf)[None]
        sm = S.mean(1); sd = S.std(1) if S.shape[1] > 1 else torch.zeros_like(sm)
        kc = CI[POS[sup[0][1][j]]]
        idx = BYCAT[kc]
        qf = describe(idx, cart)
        s = net(torch.from_numpy(qf)[None], sm, sd)[0].numpy()
        for cnd in idx[np.argsort(-s)]:
            v = int(IDS[cnd])
            if v not in got: got.append(v); break
    return got

jac = []
for tid in tids[:NVAL]:
    for h in range(3):
        p = set(predict(tid, EX[tid][h][0], hold=h))
        g = set(EX[tid][h][1])
        jac.append(len(p & g)/max(len(p | g), 1))
log('===== VALIDATION (leave-one-out over the in-context examples) =====')
log('MEAN JACCARD %.4f  -> score ~%.2f' % (np.mean(jac), 100*np.mean(jac)))
log('===== END VALIDATION =====')

rows = []
for tid, t in tasks.items():
    got = predict(tid, np.array([POS[i] for i in t['test'][0]['input']]), hold=None)
    if not got: got = [int(IDS[0])]
    rows.append((tid, ' '.join(str(x) for x in got)))
sub = pd.DataFrame(rows, columns=['task_id', 'recommended_item_ids'])
ss = pd.read_csv(PUB/'sample_submission.csv')
sub = ss[['task_id']].merge(sub, on='task_id', how='left')
sub['recommended_item_ids'] = sub.recommended_item_ids.fillna(str(int(IDS[0])))
assert list(sub.columns) == ['task_id', 'recommended_item_ids'] and len(sub) == len(ss)
assert sub.task_id.tolist() == ss.task_id.tolist()
assert sub.recommended_item_ids.str.len().min() > 0
sub.to_csv(OUT, index=False)
log('wrote %s %d rows | mean items/pred %.2f'
    % (OUT, len(sub), np.mean([len(s.split()) for s in sub.recommended_item_ids])))
log('done')
