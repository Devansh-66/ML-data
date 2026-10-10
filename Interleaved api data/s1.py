"""Interleaved session disentanglement: recovered state machine + learned candidate ranker.

The generator's grammar was measured on train and every rule below held on 100% of sessions,
so they are enforced as hard constraints, not learned preferences:

  login is always first and logout always last, exactly one of each per session
  login -> search always
  add_to_cart is always immediately preceded by view_item, and its item was viewed earlier
  view_item's category always equals the session's most recent search category
  checkout is always exactly one event before logout, and cart_size equals the session's
    total add_to_cart count

Two consequences shape everything. The session count is observable (the login count), and
declaring the i-th login to be session i removes the label permutation entirely, so the
search never explores relabellings.

Beam width was swept 96 -> 1500 and mean ARI moved only .744 -> .767 while exact recovery
stayed pinned at .417, which says the SCORING model was the ceiling rather than the search.
So the hand-factorised score is replaced by a ranker trained listwise on exactly the
decisions the beam faces: events where two or more sessions could legally take the event.
Only the released public files are read.
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
    cands = ['./dataset/public',
             '/kaggle/input/datasets/devansh66/interleaved-api-sequence-disentanglement',
             '/kaggle/input/interleaved-api-sequence-disentanglement', '/kaggle/input', '.']
    pub = next((Path(c) for c in cands if (Path(c)/'train.csv').exists()),
               Path('./dataset/public'))
    out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)
BEAM = int(os.environ.get('BEAM', 400))
NVAL = int(os.environ.get('NVAL', 1000))
EPOCHS = int(os.environ.get('EPOCHS', 40))
LAM = float(os.environ.get('LAM', 0.5))
SKIPTEST = int(os.environ.get('SKIPTEST', 0))
MAXK = 4

LOGIN, SEARCH, VIEW, ADD, CHECK, OUT_ = 0, 1, 2, 3, 4, 5
AI = {'login': LOGIN, 'search': SEARCH, 'view_item': VIEW,
      'add_to_cart': ADD, 'checkout': CHECK, 'logout': OUT_}
CATIDX = {'electronics': 1, 'books': 2, 'clothing': 3, 'home': 4, 'sports': 5}

train = pd.read_csv(PUB/'train.csv'); test = pd.read_csv(PUB/'test.csv')
log('train %s test %s' % (train.shape, test.shape))

def parse(df):
    return [[(AI[x['action']], x.get('item_id', 0), x.get('category'), x.get('cart_size', -1))
             for x in json.loads(ev)] for ev in df.events]

Etr, Ete = parse(train), parse(test)
Ytr = [[int(x) for x in c.split(',')] for c in train.cluster_labels]

# ---------------- generative factors, kept as ranker features ----------------
TRANS = np.ones((6, 6)); DEPTH = np.ones(64); SAMECAT = np.ones(2)
MAXG = 16
# The gap to a session's previous event is the interleaving prior. Timestamps turned out to be
# exactly 1..N so they carry nothing on their own, but this gap is the main thing separating
# sessions once they share an active category; adding it alone moved ARI .644 -> .739. Gaps
# stretch with concurrency, hence conditioning on the live-session count.
GAP = np.ones((5, MAXG))
for ev, lab in zip(Etr, Ytr):
    prev, done = {}, set()
    for i, (e, c) in enumerate(zip(ev, lab)):
        na = max(1, len(prev)-len(done))
        if c in prev: GAP[min(na, 4), min(i-prev[c], MAXG-1)] += 1
        prev[c] = i
        if e[0] == OUT_: done.add(c)
    s = defaultdict(list)
    for e, c in zip(ev, lab): s[c].append(e)
    for seq in s.values():
        viewed, cur = [], None
        for i, (a, it, ca, cs) in enumerate(seq):
            if i: TRANS[seq[i-1][0], a] += 1
            if a == SEARCH: SAMECAT[1 if ca == cur else 0] += 1; cur = ca
            if a == VIEW: viewed.append(it)
            if a == ADD and it in viewed:
                DEPTH[min(len(viewed)-1-viewed[::-1].index(it), 63)] += 1
LT = np.log(TRANS/TRANS.sum(1, keepdims=True))
LD = np.log(DEPTH/DEPTH.sum())
LS = np.log(SAMECAT/SAMECAT.sum())
LG = np.log(GAP/GAP.sum(1, keepdims=True))

class St:
    """One session's state. `viewed` stays ordered because add_to_cart depth is scored."""
    __slots__ = ('last', 'cat', 'viewed', 'nadd', 'done', 'pos', 'lpos', 'cnt')
    def __init__(s, pos=0):
        s.last = LOGIN; s.cat = None; s.viewed = (); s.nadd = 0
        s.done = False; s.pos = pos; s.lpos = pos; s.cnt = 1
    def copy(s):
        n = St.__new__(St)
        for k in St.__slots__: setattr(n, k, getattr(s, k))
        return n
    def key(s): return (s.last, s.cat, s.viewed, s.nadd, s.done, s.pos, s.cnt)

def score_move(st, a, it, ca, cs):
    """Hand-built log-score, or None when the grammar forbids the move. Pre-filters candidates
    and enters the ranker as one feature; the ranker decides among whatever survives."""
    if st.done: return None
    if st.last == CHECK and a != OUT_: return None      # checkout is followed only by logout
    if a == SEARCH:
        if st.last not in (LOGIN, VIEW, ADD): return None
        return LT[st.last, a] + LS[1 if ca == st.cat else 0]
    if a == VIEW:
        if st.last not in (SEARCH, VIEW, ADD): return None
        if st.cat is None or it//100 != CATIDX[st.cat]: return None
        return LT[st.last, a]
    if a == ADD:
        if st.last != VIEW: return None                 # always immediately after a view
        if it not in st.viewed: return None             # must have been viewed in-session
        return LT[st.last, a] + LD[min(len(st.viewed)-1-st.viewed[::-1].index(it), 63)]
    if a == CHECK:
        if st.last not in (VIEW, ADD): return None
        if cs != st.nadd: return None                   # cart_size pins the session exactly
        return LT[st.last, a]
    if a == OUT_:
        if st.last not in (VIEW, ADD, CHECK): return None
        return LT[st.last, a]
    return None

def apply_move(st, a, it, ca, pos):
    n = st.copy()
    n.pos = pos; n.cnt += 1
    if a == SEARCH: n.cat = ca
    elif a == VIEW: n.viewed = n.viewed + (it,)
    elif a == ADD: n.nadd += 1
    elif a == OUT_: n.done = True
    n.last = a
    return n

def feat(st, a, it, ca, pos, na, m, gap, mingap, ncand, rank):
    d = (len(st.viewed)-1-st.viewed[::-1].index(it)) if (a == ADD and it in st.viewed) else -1
    v = [gap, min(gap, MAXG-1), gap-mingap, 1.0*(gap == mingap), rank, na, ncand,
         m, LG[min(max(1, na), 4), min(gap, MAXG-1)],
         len(st.viewed), st.nadd, st.cnt, pos-st.lpos, pos,
         d, 1.0*(d == 0), 1.0*(a == VIEW and it in st.viewed),
         1.0*(a == SEARCH and ca == st.cat), 1.0*(st.cat is None),
         1.0*(st.last == CHECK), 1.0*(st.nadd > 0)]
    v += [1.0*(st.last == k) for k in range(6)]
    v += [1.0*(a == k) for k in range(6)]
    return v
ND = 21+12

def candidates(sts, a, it, ca, cs, pos):
    """Legal sessions for this event, plus the per-event context the ranker compares over."""
    raw = []
    for j, st in enumerate(sts):
        m = score_move(st, a, it, ca, cs)
        if m is not None: raw.append((j, st, m, pos-st.pos))
    if not raw: return []
    mg = min(o[3] for o in raw)
    rank = {i: r for r, i in enumerate(sorted(range(len(raw)), key=lambda i: raw[i][3]))}
    na = max(1, sum(1 for x in sts if not x.done))
    return [(j, st, m + LG[min(na, 4), min(g, MAXG-1)],
             feat(st, a, it, ca, pos, na, m, g, mg, len(raw), rank[i]))
            for i, (j, st, m, g) in enumerate(raw)]

# ---------------- training data: only the ambiguous decisions ----------------
# Teacher forcing on the true labels. Events with a single legal session are forced at
# inference too, so they carry no decision to learn and are skipped.
FX, FY, FM = [], [], []
amb = tot = 0
for ev, lab in zip(Etr, Ytr):
    sts = {}
    for pos, (a, it, ca, cs) in enumerate(ev):
        c = lab[pos]
        if a == LOGIN: sts[c] = St(pos); continue
        keys = sorted(sts)
        cand = candidates([sts[k] for k in keys], a, it, ca, cs, pos)
        tot += 1
        if len(cand) > 1:
            amb += 1
            ti = next((i for i, (j, _, _, _) in enumerate(cand) if keys[j] == c), None)
            if ti is not None and ti < MAXK:
                F = np.zeros((MAXK, ND), np.float32); M = np.zeros(MAXK, np.float32)
                for i, (_, _, _, f) in enumerate(cand[:MAXK]):
                    F[i] = f; M[i] = 1.0
                FX.append(F); FM.append(M); FY.append(ti)
        sts[c] = apply_move(sts[c], a, it, ca, pos)
FX = np.stack(FX); FM = np.stack(FM); FY = np.array(FY)
log('decisions: %d total, %d ambiguous (%.3f) -> %d training groups'
    % (tot, amb, amb/tot, len(FX)))

_f = FX.reshape(-1, ND)[FM.reshape(-1) > 0]
MU, SD = _f.mean(0), _f.std(0)+1e-6

import torch, torch.nn as nn
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.manual_seed(SEED)

class Rank(nn.Module):
    """Scores each candidate independently, then softmaxes over the candidate set. Weights are
    shared across candidates, so the model cannot key on a session's index."""
    def __init__(s, d=ND, h=64):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                            nn.Linear(h, 1))
    def forward(s, x, m):
        return s.f(x).squeeze(-1).masked_fill(m < 0.5, -1e9)

Xn = torch.tensor((FX-MU)/SD, dtype=torch.float32)
Mt = torch.tensor(FM); Yt = torch.tensor(FY)
n = len(Xn); pm = torch.randperm(n); VI, TI = pm[:n//10], pm[n//10:]
net = Rank()
opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS, pct_start=0.3)
for ep in range(EPOCHS):
    net.train(); perm = TI[torch.randperm(len(TI))]
    for k in range(0, len(perm), 1024):
        b = perm[k:k+1024]
        loss = nn.functional.cross_entropy(net(Xn[b], Mt[b]), Yt[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sch.step()
net.eval()
with torch.no_grad():
    acc = (net(Xn[VI], Mt[VI]).argmax(1) == Yt[VI]).float().mean().item()
hb = FX[VI.numpy()][:, :, 7] + np.where(FM[VI.numpy()] > 0, 0.0, -1e9)
log('ranker: held-out decision accuracy %.4f (hand-built score alone %.4f)'
    % (acc, (hb.argmax(1) == FY[VI.numpy()]).mean()))

# the same forward pass in numpy, so a beam step scores all candidates in one shot
WT = [p.detach().numpy() for p in net.f.parameters()]
def _gelu(x): return 0.5*x*(1.0+np.tanh(0.7978845608*(x+0.044715*x**3)))
def rank_scores(X):
    z = (X-MU)/SD
    for i in range(0, len(WT)-2, 2): z = _gelu(z@WT[i].T + WT[i+1])
    return (z@WT[-2].T + WT[-1]).ravel()

def solve(ev, beam=BEAM):
    K = sum(1 for e in ev if e[0] == LOGIN)
    if K <= 1: return [0]*len(ev)
    cur = [(0.0, (), [])]
    for pos, (a, it, ca, cs) in enumerate(ev):
        if a == LOGIN:
            cur = [(sc, sts+(St(pos),), lb+[len(sts)]) for sc, sts, lb in cur]
            continue
        rows, meta, groups = [], [], []
        for hi, (sc, sts, lb) in enumerate(cur):
            cand = candidates(sts, a, it, ca, cs, pos)
            if not cand: continue
            g0 = len(rows)
            for (j, st, mg, f) in cand:
                rows.append(f); meta.append((j, st, mg))
            groups.append((hi, g0, len(cand)))
        if not rows: return greedy(ev)
        S = rank_scores(np.asarray(rows, np.float32))
        seen = {}
        for hi, g0, nc in groups:
            sc, sts, lb = cur[hi]
            z = S[g0:g0+nc]; mx = z.max()
            z = z - (mx + np.log(np.exp(z-mx).sum()))
            for t in range(nc):
                j, st, mg = meta[g0+t]
                ns = list(sts); ns[j] = apply_move(st, a, it, ca, pos); ns = tuple(ns)
                k = tuple(x.key() for x in ns)
                v = sc + mg + LAM*float(z[t])
                if k not in seen or v > seen[k][0]: seen[k] = (v, ns, lb+[j])
        cur = sorted(seen.values(), key=lambda x: -x[0])[:beam]
    fin = [c for c in cur if all(s.done for s in c[1])]
    return (fin or cur)[0][2]

def greedy(ev):
    """Constraint-respecting single pass; only reached if the beam dead-ends."""
    sts, lb = [], []
    for pos, (a, it, ca, cs) in enumerate(ev):
        if a == LOGIN: sts.append(St(pos)); lb.append(len(sts)-1); continue
        cand = candidates(tuple(sts), a, it, ca, cs, pos)
        if not cand:
            j = next((i for i, s in enumerate(sts) if not s.done), 0)
        else:
            g = np.asarray([c[2] for c in cand], np.float64)
            j = cand[int(np.argmax(g + LAM*rank_scores(
                np.asarray([c[3] for c in cand], np.float32))))][0]
        sts[j] = apply_move(sts[j], a, it, ca, pos); lb.append(j)
    return lb

def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    C = np.zeros((a.max()+1, b.max()+1))
    for x, y in zip(a, b): C[x, y] += 1
    def c2(v): return (v*(v-1)/2).sum()
    si, sj, sij = c2(C.sum(1)), c2(C.sum(0)), c2(C)
    exp = si*sj/(n*(n-1)/2); mx = (si+sj)/2
    return 1.0 if mx == exp else (sij-exp)/(mx-exp)

idx = np.random.RandomState(SEED).permutation(len(Etr))[:NVAL]
sc = [max(0.0, ari(solve(Etr[i]), Ytr[i])) for i in idx]
by = defaultdict(list)
for i, v in zip(idx, sc): by[len(set(Ytr[i]))].append(v)
log('===== VALIDATION (send this back) =====')
log('MEAN ARI %.4f on %d train rows | exactly recovered %.3f'
    % (np.mean(sc), len(idx), np.mean([v >= 0.9999 for v in sc])))
log('by session count: %s' % {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())})
log('===== END VALIDATION =====')
if SKIPTEST: sys.exit(0)

pred = [solve(e) for e in Ete]
sub = pd.DataFrame({'id': test.id.values,
                    'cluster_labels': [','.join(map(str, p)) for p in pred]})
assert list(sub.columns) == ['id', 'cluster_labels'] and len(sub) == len(test)
assert sub.id.tolist() == test.id.tolist() and not sub.cluster_labels.isna().any()
for s, n_ in zip(sub.cluster_labels, test.num_events):
    p = s.split(',')
    assert len(p) == n_ and all(x.isdigit() for x in p)
sub.to_csv(OUT, index=False)
log('wrote %s %d rows | mean clusters %.2f'
    % (OUT, len(sub), np.mean([len(set(p)) for p in pred])))
log('done')
