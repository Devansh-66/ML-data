"""Logical bundle induction: per-task few-shot learning of a relational scorer over candidates.

Why this shape. Measured on the 3,000 tasks:
  output SIZE and output CATEGORY multiset are constant within a task (both 1.0000), so WHICH
  categories to emit is given; only WHICH ITEM in each category has to be induced.
  A two-stage pipeline -- induce colour/brand, then pick an item in that cell -- reaches only
  0.2195 cell accuracy under honest leave-one-out, because attributes and selection are
  ENTANGLED. Inspecting failures shows why: in many tasks the item is chosen by a price
  relation to the cart (e.g. nearest to the cheapest input) and its colour and brand are simply
  whatever that item happens to have; in others colour and brand are copied from the input item
  that shares the output's category, and the item is the cheapest of that cell.

So this scores CANDIDATE ITEMS directly. Each candidate is described by relational features
against the current cart (price relations, and whether its colour/brand match distinguished
input items), and a small linear scorer is fitted PER TASK on that task's own three examples --
few-shot supervised learning, with the examples as the training set. Nothing enumerates a
hypothesis space: the weights are estimated from data by ridge regression, and the same fixed
feature vocabulary is used for every task.
"""
import os, sys, json, time
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
LAM = float(os.environ.get('LAM', 1.0))
NVAL = int(os.environ.get('NVAL', 600))
catalog = json.load(open(PUB/'catalog.json', encoding='utf-8'))
tasks = json.load(open(PUB/'tasks.json', encoding='utf-8'))
log('catalog %d | tasks %d' % (len(catalog), len(tasks)))

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
# price rank inside a category, as a 0..1 position
RANK = np.zeros(len(catalog))
for k, idx in BYCAT.items():
    o = idx[np.argsort(-PRICE[idx])]
    RANK[o] = np.arange(len(o))/max(len(o)-1, 1)

NF = 22

def feats(idx, inp):
    """Relational description of every candidate in `idx` against the cart `inp`."""
    p = PRICE[idx]
    ip = PRICE[inp]
    lo, hi, mu, sm = ip.min(), ip.max(), ip.mean(), ip.sum()
    sc = 500.0
    # distinguished cart items: cheapest, dearest, first, last
    d = {'lo': inp[int(np.argmin(ip))], 'hi': inp[int(np.argmax(ip))],
         'f': inp[0], 'l': inp[-1]}
    F = np.empty((len(idx), NF))
    F[:, 0] = RANK[idx]
    F[:, 1] = p/sc
    F[:, 2] = -np.abs(p-lo)/sc
    F[:, 3] = -np.abs(p-hi)/sc
    F[:, 4] = -np.abs(p-mu)/sc
    F[:, 5] = -np.abs(p-sm)/sc
    F[:, 6] = (p > hi).astype(float)
    F[:, 7] = (p < lo).astype(float)
    c = 8
    for key in ('lo', 'hi', 'f', 'l'):
        F[:, c] = (OI[idx] == OI[d[key]]).astype(float); c += 1
        F[:, c] = (BI[idx] == BI[d[key]]).astype(float); c += 1
    # majority colour/brand of the cart, and mere presence anywhere in it
    mc = Counter(OI[inp].tolist()).most_common(1)[0][0]
    mb = Counter(BI[inp].tolist()).most_common(1)[0][0]
    F[:, c] = (OI[idx] == mc).astype(float); c += 1
    F[:, c] = (BI[idx] == mb).astype(float); c += 1
    F[:, c] = np.isin(OI[idx], OI[inp]).astype(float); c += 1
    F[:, c] = np.isin(BI[idx], BI[inp]).astype(float); c += 1
    F[:, c] = 1.0
    return F

def fit_ridge(X, y, lam=LAM):
    A = X.T @ X + lam*np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)

def solve_task(t, hold=None):
    """Fit the scorer on this task's examples (optionally holding one out) and rank candidates."""
    ex = [(e['input'], sorted(e['output'], key=lambda i: (CI[POS[i]], i)))
          for e in t['train']]
    use = [e for k, e in enumerate(ex) if k != hold]
    cats = [CI[POS[o]] for o in use[0][1]]
    preds = {}
    for j, kc in enumerate(cats):
        idx = BYCAT[kc]
        Xs, ys = [], []
        for inp, out in use:
            ii = np.array([POS[i] for i in inp])
            if j >= len(out): continue
            g = POS[out[j]]
            F = feats(idx, ii)
            y = (idx == g).astype(np.float64)
            if y.sum() == 0: continue
            Xs.append(F); ys.append(y)
        if not Xs: continue
        w = fit_ridge(np.vstack(Xs), np.concatenate(ys))
        preds[j] = (kc, w)
    return ex, cats, preds

def predict(preds, cats, inp):
    ii = np.array([POS[i] for i in inp])
    got = []
    for j, kc in enumerate(cats):
        if j not in preds: continue
        _, w = preds[j]
        idx = BYCAT[kc]
        s = feats(idx, ii) @ w
        order = idx[np.argsort(-s)]
        for cand in order:
            v = int(IDS[cand])
            if v not in got: got.append(v); break
    return got

# honest leave-one-out: fit on 2 examples, predict the held-out third
jac = []
for tid in list(tasks)[:NVAL]:
    t = tasks[tid]
    for h in range(3):
        ex, cats, preds = solve_task(t, hold=h)
        p = set(predict(preds, cats, ex[h][0]))
        g = set(ex[h][1])
        jac.append(len(p & g)/max(len(p | g), 1))
log('===== VALIDATION (leave-one-out over the in-context examples) =====')
log('MEAN JACCARD %.4f  -> score ~%.2f' % (np.mean(jac), 100*np.mean(jac)))
log('===== END VALIDATION =====')

rows = []
for tid, t in tasks.items():
    ex, cats, preds = solve_task(t, hold=None)
    got = predict(preds, cats, t['test'][0]['input'])
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
