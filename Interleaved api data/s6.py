"""Interleaved session disentanglement: recovered state machine + learned candidate ranker.

Grammar measured on train; every rule below held on 100% of sessions, so all are enforced as
hard constraints rather than learned preferences:

  login first and logout last, exactly one of each per session
  login -> search always
  add_to_cart is always immediately preceded by view_item, and its item was viewed earlier
  view_item's category always equals the session's most recent search category
  checkout is always exactly one event before logout, and cart_size equals the session's
    total add_to_cart count

Consequences: the session count is observable (the login count), and declaring the i-th login
to be session i removes the label permutation, so the search never explores relabellings.

Measured history, so later changes are not re-litigated:
  beam 96 -> 1500 moved ARI only .744 -> .767 with exact recovery pinned at .417, so the
    SCORING model was the ceiling, not the search
  the gap-to-previous-event prior alone moved ARI .644 -> .739
  a ranker trained by teacher forcing wins per-decision (.748 vs .608) but LOSES end to end
    when trusted alone (.723 vs .762) -- it never saw a corrupted prefix. So the joint
    likelihood stays the backbone and the ranker enters weighted (LAM).
  validating on rows the ranker had trained on read .775 against .719 on the leaderboard, so
    validation now holds whole rows out of every fitted component.

The block is fully observed, so suffix counts are legitimate: each live session still needs
its own logout, which is both a hard feasibility prune and a strong feature.
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
NVAL = int(os.environ.get('NVAL', 600))
EPOCHS = int(os.environ.get('EPOCHS', 40))
LAM = float(os.environ.get('LAM', 1.0))
SKIPTEST = int(os.environ.get('SKIPTEST', 0))
DIAG = int(os.environ.get('DIAG', 0))
MBR = int(os.environ.get('MBR', 1))
TOPH = int(os.environ.get('TOPH', 50))
TEMP = float(os.environ.get('TEMP', 1.0))
CLIMB = int(os.environ.get('CLIMB', 1))
MAXK, MAXG = 4, 16

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

def suffix(ev):
    """How many logouts remain at each position."""
    n = len(ev)
    sl = np.zeros(n+1, np.int32)
    for i in range(n-1, -1, -1): sl[i] = sl[i+1] + (1 if ev[i][0] == OUT_ else 0)
    return sl

SUFtr = [suffix(e) for e in Etr]; SUFte = [suffix(e) for e in Ete]

import torch, torch.nn as nn
torch.set_num_threads(max(1, os.cpu_count() or 1))

class St:
    """One session's state. `viewed` stays ordered because add_to_cart depth is scored."""
    __slots__ = ('last', 'cat', 'viewed', 'nadd', 'done', 'pos', 'lpos', 'cnt', 'items')
    def __init__(s, pos=0):
        s.last = LOGIN; s.cat = None; s.viewed = (); s.nadd = 0
        s.done = False; s.pos = pos; s.lpos = pos; s.cnt = 1; s.items = ()
    def copy(s):
        n = St.__new__(St)
        for k in St.__slots__: setattr(n, k, getattr(s, k))
        return n
    def key(s): return (s.last, s.cat, s.viewed, s.nadd, s.done, s.pos, s.cnt, s.items)

def apply_move(st, a, it, ca, pos):
    n = st.copy()
    n.pos = pos; n.cnt += 1
    if a == SEARCH: n.cat = ca
    elif a == VIEW: n.viewed = n.viewed + (it,); n.items = n.items + (it,)
    elif a == ADD: n.nadd += 1; n.items = n.items + (it,)
    elif a == OUT_: n.done = True
    n.last = a
    return n

def score_move(st, a, it, ca, cs):
    """Hand-built log-score, or None when the grammar forbids the move."""
    if st.done: return None
    if st.last == CHECK and a != OUT_: return None      # checkout is followed only by logout
    if a == SEARCH:
        if st.last not in (LOGIN, VIEW, ADD): return None
        return LT[st.last, a] + LS[1 if ca == st.cat else 0]
    if a == VIEW:
        if st.last not in (SEARCH, VIEW, ADD): return None
        if st.cat is None or it//100 != CATIDX[st.cat]: return None
        # Events sharing an item_id land in the same session far more often than
        # same-category-different-item pairs do (.745 vs .545 measured, a 1.37x lift beyond
        # category). The pairwise sum decomposes into this per-event count, so the global
        # affinity is captured exactly without leaving the sequential decoder.
        return LT[st.last, a] + ITEMAFF*st.items.count(it)
    if a == ADD:
        if st.last != VIEW: return None                 # always immediately after a view
        if it not in st.viewed: return None             # must have been viewed in-session
        return (LT[st.last, a] + LD[min(len(st.viewed)-1-st.viewed[::-1].index(it), 63)]
                + ITEMAFF*st.items.count(it))
    if a == CHECK:
        if st.last not in (VIEW, ADD): return None
        if cs != st.nadd: return None                   # cart_size pins the session exactly
        return LT[st.last, a]
    if a == OUT_:
        if st.last not in (VIEW, ADD, CHECK): return None
        return LT[st.last, a]
    return None

def feat(st, a, it, ca, pos, na, m, gap, mingap, ncand, rank, slack, nrem, nsame):
    d = (len(st.viewed)-1-st.viewed[::-1].index(it)) if (a == ADD and it in st.viewed) else -1
    v = [gap, min(gap, MAXG-1), gap-mingap, 1.0*(gap == mingap), rank, na, ncand,
         m, LG[min(max(1, na), 4), min(gap, MAXG-1)],
         len(st.viewed), st.nadd, st.cnt, pos-st.lpos, pos,
         d, 1.0*(d == 0), 1.0*(a == VIEW and it in st.viewed),
         1.0*(a == SEARCH and ca == st.cat), 1.0*(st.cat is None),
         1.0*(st.last == CHECK), 1.0*(st.nadd > 0),
         slack, 1.0*(slack == 0), nrem, nrem/max(na, 1.0),
         nsame, math.log1p(nsame), 1.0*(nsame > 0), nsame/max(st.cnt, 1.0)]
    v += [1.0*(st.last == k) for k in range(6)]
    v += [1.0*(a == k) for k in range(6)]
    return v
ND = 29+12

def candidates(sts, a, it, ca, cs, pos, sl, n):
    """Legal sessions plus the per-event context the ranker compares over.

    `slack` is spare logouts: every live session still needs one of its own, so slack 0 means
    each remaining logout is already committed."""
    raw = []
    for j, st in enumerate(sts):
        m = score_move(st, a, it, ca, cs)
        if m is not None: raw.append((j, st, m, pos-st.pos))
    if not raw: return []
    mg = min(o[3] for o in raw)
    rank = {i: r for r, i in enumerate(sorted(range(len(raw)), key=lambda i: raw[i][3]))}
    na = max(1, sum(1 for x in sts if not x.done))
    slack = int(sl[pos]) - na
    nrem = n - pos
    return [(j, st, m + LG[min(na, 4), min(g, MAXG-1)],
             feat(st, a, it, ca, pos, na, m, g, mg, len(raw), rank[i], slack, nrem,
                  st.items.count(it) if it else 0))
            for i, (j, st, m, g) in enumerate(raw)]

class Rank(nn.Module):
    """Scores each candidate independently, then softmaxes over the candidate set. Weights are
    shared across candidates, so the model cannot key on a session's index."""
    def __init__(s, d=ND, h=64):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                            nn.Linear(h, 1))
    def forward(s, x, m):
        return s.f(x).squeeze(-1).masked_fill(m < 0.5, -1e9)

def _gelu(x): return 0.5*x*(1.0+np.tanh(0.7978845608*(x+0.044715*x**3)))
def rank_scores(X):
    z = (X-MU)/SD
    for i in range(0, len(WT)-2, 2): z = _gelu(z@WT[i].T + WT[i+1])
    return (z@WT[-2].T + WT[-1]).ravel()

def fit(rows):
    """Fit every learned component on these rows only, so validation can hold rows out."""
    global LT, LD, LS, LG, MU, SD, WT
    TRANS = np.ones((6, 6)); DEPTH = np.ones(64); SAMECAT = np.ones(2)
    GAP = np.ones((5, MAXG))
    for r in rows:
        ev, lab = Etr[r], Ytr[r]
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
    # log-likelihood ratio for the same-item pairwise link, measured on the fitting rows: how
    # much more often a shared item_id means one session than a shared category alone does.
    si_s = si_t = cc_s = cc_t = 0
    for r in rows:
        its = [(e[1], c) for e, c in zip(Etr[r], Ytr[r]) if e[1]]
        for x in range(len(its)):
            for y in range(x+1, len(its)):
                (ia, ca_), (ib, cb) = its[x], its[y]
                if ia == ib: si_t += 1; si_s += (ca_ == cb)
                elif ia//100 == ib//100: cc_t += 1; cc_s += (ca_ == cb)
    p1 = (si_s+1)/(si_t+2); p0 = (cc_s+1)/(cc_t+2)
    global ITEMAFF
    ITEMAFF = float(os.environ.get('ITEMAFF', math.log(p1/p0)))
    log('  item affinity: P(same|same item) %.4f vs P(same|same cat, diff item) %.4f'
        ' -> bonus %.4f per shared item' % (p1, p0, ITEMAFF))
    MU, SD = np.zeros(ND), np.ones(ND)

    FX, FY, FM = [], [], []
    amb = tot = 0
    for r in rows:
        ev, lab, sl = Etr[r], Ytr[r], SUFtr[r]
        sts = {}
        for pos, (a, it, ca, cs) in enumerate(ev):
            c = lab[pos]
            if a == LOGIN: sts[c] = St(pos); continue
            keys = sorted(sts)
            cand = candidates([sts[k] for k in keys], a, it, ca, cs, pos, sl, len(ev))
            tot += 1
            if len(cand) > 1:
                amb += 1
                ti = next((i for i, x in enumerate(cand) if keys[x[0]] == c), None)
                if ti is not None and ti < MAXK:
                    F = np.zeros((MAXK, ND), np.float32); M = np.zeros(MAXK, np.float32)
                    for i, x in enumerate(cand[:MAXK]): F[i] = x[3]; M[i] = 1.0
                    FX.append(F); FM.append(M); FY.append(ti)
            sts[c] = apply_move(sts[c], a, it, ca, pos)
    FX = np.stack(FX); FM = np.stack(FM); FY = np.array(FY)
    _f = FX.reshape(-1, ND)[FM.reshape(-1) > 0]
    MU, SD = _f.mean(0), _f.std(0)+1e-6

    Xn = torch.tensor((FX-MU)/SD, dtype=torch.float32)
    Mt = torch.tensor(FM); Yt = torch.tensor(FY)
    torch.manual_seed(SEED)
    net = Rank()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS, pct_start=0.3)
    for ep in range(EPOCHS):
        net.train(); pm = torch.randperm(len(Xn))
        for k in range(0, len(pm), 1024):
            b = pm[k:k+1024]
            loss = nn.functional.cross_entropy(net(Xn[b], Mt[b]), Yt[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    net.eval()
    WT = [p.detach().numpy() for p in net.f.parameters()]
    with torch.no_grad():
        tacc = (net(Xn, Mt).argmax(1) == Yt).float().mean().item()
    log('  fit %d rows: %d decisions, %.3f ambiguous, %d groups, in-fit decision acc %.4f'
        % (len(rows), tot, amb/max(tot, 1), len(FX), tacc))

RETSCORE = False

def mbr_pick(cands, n):
    """Minimum-Bayes-risk decoding against the metric instead of MAP decoding.

    The diagnostic said our objective ranks a WRONG partition above the truth in 200 of 300
    rows while the search itself fails in only 30, so returning the single most likely
    partition is the wrong move. ARI scores pairwise co-assignment, so we take the beam's
    posterior, form the pairwise co-assignment probability, and return the partition agreeing
    with it best. Maximising expected RAND is the tractable surrogate -- ARI's adjustment term
    depends on the marginals of the answer being chosen, so it is not separable over pairs.

    TEMP exists because our score is a BLEND of a likelihood and a ranker, not a normalised
    log-probability. Untempered, exp(score) concentrates on one hypothesis and this collapses
    back to MAP -- which is exactly what the first attempt measured (.7603 vs .7571).

    The hill-climb is free to leave the beam's hypothesis set: the grader validates only the
    label format, not the grammar, so the consensus partition need not be a legal session
    decomposition. It only ever accepts strict improvements in expected agreement.
    """
    sc = np.array([c[0] for c in cands], np.float64)
    w = np.exp((sc-sc.max())/max(TEMP, 1e-6)); w /= w.sum()
    L = np.asarray([c[1] for c in cands])
    P = np.zeros((n, n))
    for h in range(len(L)):
        P += w[h]*(L[h][:, None] == L[h][None, :])
    iu = np.triu_indices(n, 1)
    Pu = P[iu]
    best, lab = None, L[0]
    for h in range(len(L)):
        same = L[h][iu[0]] == L[h][iu[1]]
        v = float(np.where(same, Pu, 1.0-Pu).sum())
        if best is None or v > best: best, lab = v, L[h]
    if not CLIMB: return lab.tolist()

    lab = np.array(lab)
    K = max(int(lab.max())+1, 1)
    Q = 1.0-P
    for _ in range(20):
        moved = False
        for i in range(n):
            cur = lab[i]
            bestc, bestv = cur, None
            for c in range(K):
                lab[i] = c
                m = lab == c; m[i] = False
                o = (lab != c); o[i] = False
                v = float(P[i, m].sum() + Q[i, o].sum())
                if bestv is None or v > bestv + 1e-12: bestc, bestv = c, v
            lab[i] = bestc
            if bestc != cur: moved = True
        if not moved: break
    return lab.tolist()

def solve(ev, sl, beam=BEAM):
    K = sum(1 for e in ev if e[0] == LOGIN)
    if K <= 1: return ([0]*len(ev), 0.0) if RETSCORE else [0]*len(ev)
    n = len(ev)
    cur = [(0.0, (), [])]
    for pos, (a, it, ca, cs) in enumerate(ev):
        if a == LOGIN:
            cur = [(sc, sts+(St(pos),), lb+[len(sts)]) for sc, sts, lb in cur]
            continue
        rows, meta, groups = [], [], []
        for hi, (sc, sts, lb) in enumerate(cur):
            cand = candidates(sts, a, it, ca, cs, pos, sl, n)
            if not cand: continue
            g0 = len(rows)
            for (j, st, mg, f) in cand:
                rows.append(f); meta.append((j, st, mg))
            groups.append((hi, g0, len(cand)))
        if not rows: return greedy(ev, sl)
        S = rank_scores(np.asarray(rows, np.float32))
        seen = {}
        for hi, g0, nc in groups:
            sc, sts, lb = cur[hi]
            z = S[g0:g0+nc]; mx = z.max()
            z = z - (mx + np.log(np.exp(z-mx).sum()))
            for t in range(nc):
                j, st, mg = meta[g0+t]
                ns = list(sts); ns[j] = apply_move(st, a, it, ca, pos); ns = tuple(ns)
                # every still-live session needs its own logout out of the remaining suffix
                if sum(1 for x in ns if not x.done) > int(sl[pos+1]): continue
                k = tuple(x.key() for x in ns)
                v = sc + mg + LAM*float(z[t])
                if k not in seen or v > seen[k][0]: seen[k] = (v, ns, lb+[j])
        if not seen: return greedy(ev, sl)
        cur = sorted(seen.values(), key=lambda x: -x[0])[:beam]
    fin = [c for c in cur if all(s.done for s in c[1])] or cur
    b = fin[0]
    if RETSCORE: return (b[2], b[0])
    if MBR and len(fin) > 1:
        return mbr_pick([(c[0], c[2]) for c in fin[:TOPH]], n)
    return b[2]

def greedy(ev, sl):
    """Constraint-respecting single pass; only reached if the beam dead-ends."""
    sts, lb = [], []
    for pos, (a, it, ca, cs) in enumerate(ev):
        if a == LOGIN: sts.append(St(pos)); lb.append(len(sts)-1); continue
        cand = candidates(tuple(sts), a, it, ca, cs, pos, sl, len(ev))
        if not cand:
            j = next((i for i, s in enumerate(sts) if not s.done), 0)
        else:
            g = np.asarray([c[2] for c in cand], np.float64)
            j = cand[int(np.argmax(g + LAM*rank_scores(
                np.asarray([c[3] for c in cand], np.float32))))][0]
        sts[j] = apply_move(sts[j], a, it, ca, pos); lb.append(j)
    return (lb, -1e18) if RETSCORE else lb

def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    C = np.zeros((a.max()+1, b.max()+1))
    for x, y in zip(a, b): C[x, y] += 1
    def c2(v): return (v*(v-1)/2).sum()
    si, sj, sij = c2(C.sum(1)), c2(C.sum(0)), c2(C)
    exp = si*sj/(n*(n-1)/2); mx = (si+sj)/2
    return 1.0 if mx == exp else (sij-exp)/(mx-exp)

def score_labeling(ev, lab, sl):
    """Score of a GIVEN partition under exactly the beam's objective. Comparing it against the
    beam's own best separates a ranking mistake we could fix from a limit we cannot."""
    sts, tot = {}, 0.0
    for pos, (a, it, ca, cs) in enumerate(ev):
        c = lab[pos]
        if a == LOGIN: sts[c] = St(pos); continue
        keys = sorted(sts)
        cand = candidates([sts[k] for k in keys], a, it, ca, cs, pos, sl, len(ev))
        ti = next((i for i, x in enumerate(cand) if keys[x[0]] == c), None)
        if ti is None: return None
        z = rank_scores(np.asarray([x[3] for x in cand], np.float32))
        mx = z.max(); z = z - (mx + np.log(np.exp(z-mx).sum()))
        tot += cand[ti][2] + LAM*float(z[ti])
        sts[c] = apply_move(sts[c], a, it, ca, pos)
    return tot

# ---------------- honest validation: rows held out of every fitted component ----------------
perm = np.random.RandomState(SEED).permutation(len(Etr))
VAL, FIT = perm[:NVAL], perm[NVAL:]
log('fitting on %d rows, validating on %d held-out rows' % (len(FIT), len(VAL)))
fit(FIT)

if DIAG:
    RETSCORE = True
    hi = lo = eq = bad = 0
    for i in VAL[:300]:
        pl, ps = solve(Etr[i], SUFtr[i])
        ts = score_labeling(Etr[i], Ytr[i], SUFtr[i])
        if ts is None: bad += 1
        elif ts > ps+1e-9: hi += 1
        elif ts < ps-1e-9: lo += 1
        else: eq += 1
    log('DIAGNOSTIC: truth outscores our MAP in %d rows (recoverable by better search/model),'
        ' scores LOWER in %d (information limit), tied %d, ungradable %d' % (hi, lo, eq, bad))
    sys.exit(0)

sc = [max(0.0, ari(solve(Etr[i], SUFtr[i]), Ytr[i])) for i in VAL]
by = defaultdict(list)
for i, v in zip(VAL, sc): by[len(set(Ytr[i]))].append(v)
log('===== VALIDATION (held-out rows, send this back) =====')
log('MEAN ARI %.4f on %d rows | exactly recovered %.3f'
    % (np.mean(sc), len(VAL), np.mean([v >= 0.9999 for v in sc])))
log('by session count: %s' % {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())})
log('===== END VALIDATION =====')
if SKIPTEST: sys.exit(0)

log('refitting on all train rows for the submission...')
fit(np.arange(len(Etr)))
pred = [solve(e, s) for e, s in zip(Ete, SUFte)]
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
