"""Interleaved session disentanglement: EXACT enumeration of grammar-legal partitions + MBR.

The generator is identified from train, and every rule below was verified on all 5000 rows:

  login first / logout last, exactly one of each per session          (15027/15027)
  login -> search always
  view_item's category == the session's most recent search category   (75052/75052)
  add_to_cart's item == the item viewed IMMEDIATELY before it         (30068/30068)
  checkout is exactly one event before logout, cart_size == session add count  (8896/8896)
  searches per session in {1,2,3}, uniform
  views per search block in {1,2,3,4}, uniform
  timestamps are exactly 1..N, so they add nothing beyond position

Those constraints are tight enough that the legal partition count is TINY -- median 8, p90
~1200 -- so the entire partition set is enumerated exactly and beam search is unnecessary.
Two things follow: the truth is always inside the legal set, and the posterior over it is
exact, so MBR decoding uses true pairwise marginals rather than a beam's near-duplicate
max-marginals (which is why beam MBR had gained only +0.003).

What the content contributes: almost nothing. Searches, views, adds, items and categories are
all uniform, so the content log-likelihood is CONSTANT across legal partitions in ~74% of
rows. Only two content terms vary -- whether a search repeats the session's previous category
(p=.1001, vs .225 for each other category) and whether a session with a non-empty cart checks
out (p=.701). Action bigrams and add-depth were REMOVED: they vary across partitions while the
true likelihood ratio is 1, so they actively voted for wrong partitions.

The discriminative signal is therefore almost entirely the interleaving law: each session
advances by a gap ~U{1..10}, making the stream strongly turn-taking. The smallest-gap
candidate is the true one only 25.5% of the time -- well below chance.

Rejected after measurement, recorded so they are not retried:
  same-item affinity between view events -- lift 0.995, i.e. nothing. The apparent 1.37x lift
    was entirely the add_to_cart/preceding-view pair, which is now a hard constraint.
  per-session pace as identity (spread .128); session length vs start position (corr -.005);
    distinct categories per session (only 20% of blocks disjoint); model capacity (gradient
    boosting scored .728 against an MLP's .748 on the same decisions).
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
NVAL = int(os.environ.get('NVAL', 600))
MBRT = float(os.environ.get('MBRT', 2.0))   # not TEMP: that collides with %TEMP% on Windows
PCAP = int(os.environ.get('PCAP', 60000))      # legal partitions kept before falling back
NODECAP = int(os.environ.get('NODECAP', 500000))
BEAM = int(os.environ.get('BEAM', 300))
USECAT = int(os.environ.get('USECAT', 1))
USECO = int(os.environ.get('USECO', 1))
SKIPTEST = int(os.environ.get('SKIPTEST', 0))
MAXG = 16

LOGIN, SEARCH, VIEW, ADD, CHECK, OUT_ = 0, 1, 2, 3, 4, 5
AI = {'login': LOGIN, 'search': SEARCH, 'view_item': VIEW,
      'add_to_cart': ADD, 'checkout': CHECK, 'logout': OUT_}
CATIDX = {'electronics': 1, 'books': 2, 'clothing': 3, 'home': 4, 'sports': 5}
MAXS, MAXV = 3, 4                              # searches per session; views per search block

train = pd.read_csv(PUB/'train.csv'); test = pd.read_csv(PUB/'test.csv')
log('train %s test %s' % (train.shape, test.shape))

def parse(df):
    return [[(AI[x['action']], x.get('item_id', 0),
              CATIDX.get(x.get('category'), 0), x.get('cart_size', -1))
             for x in json.loads(ev)] for ev in df.events]

Etr, Ete = parse(train), parse(test)
Ytr = [[int(x) for x in c.split(',')] for c in train.cluster_labels]

def suffix(ev):
    n = len(ev)
    sl = np.zeros(n+1, np.int32)
    for i in range(n-1, -1, -1): sl[i] = sl[i+1] + (1 if ev[i][0] == OUT_ else 0)
    return sl
SUFtr = [suffix(e) for e in Etr]; SUFte = [suffix(e) for e in Ete]

def fit(rows):
    """Only the terms that actually vary across legal partitions are fitted."""
    global LG, LCAT_SAME, LCAT_DIFF, LCO, LNOCO
    GAP = np.ones((5, MAXG))
    csame = cdiff = 1.0
    nco = ncart = 1.0
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
            cur = 0
            for (a, it, ca, cs) in seq:
                if a == SEARCH:
                    if cur: csame += (ca == cur); cdiff += (ca != cur)
                    cur = ca
            if any(x[0] == ADD for x in seq):
                ncart += 1; nco += any(x[0] == CHECK for x in seq)
    LG = np.log(GAP/GAP.sum(1, keepdims=True))
    p = csame/(csame+cdiff)
    # "different" spreads over the other FOUR categories, so its per-category probability is
    # (1-p)/4. Scoring it as a single two-way outcome over-penalised a repeat by ~1.4 nats.
    LCAT_SAME, LCAT_DIFF = math.log(p), math.log((1.0-p)/4.0)
    q = nco/ncart
    LCO, LNOCO = math.log(q), math.log(max(1.0-q, 1e-9))
    log('  fit %d rows: P(search repeats category) %.4f | P(checkout | cart non-empty) %.4f'
        % (len(rows), p, q))

LAST, CAT, LASTV, NADD, DONE, POS, NS, VB, HASCO = range(9)
def new_state(pos): return [LOGIN, 0, -1, 0, 0, pos, 0, 0, 0]

def legal_inc(st, a, it, ca, cs, pos, na):
    """Grammar check plus the log-score increment, or None when illegal."""
    if st[DONE]: return None
    if st[LAST] == CHECK and a != OUT_: return None
    if a == SEARCH:
        if st[LAST] not in (LOGIN, VIEW, ADD) or st[NS] >= MAXS: return None
        inc = (LCAT_SAME if ca == st[CAT] else LCAT_DIFF) if (USECAT and st[CAT]) else 0.0
    elif a == VIEW:
        if st[LAST] not in (SEARCH, VIEW, ADD): return None
        if not st[CAT] or it//100 != st[CAT] or st[VB] >= MAXV: return None
        inc = 0.0
    elif a == ADD:
        if st[LAST] != VIEW or it != st[LASTV]: return None   # always the last item viewed
        inc = 0.0
    elif a == CHECK:
        if st[LAST] not in (VIEW, ADD) or cs != st[NADD]: return None
        inc = LCO if USECO else 0.0
    elif a == OUT_:
        if st[LAST] not in (VIEW, ADD, CHECK): return None
        inc = LNOCO if (USECO and st[NADD] > 0 and not st[HASCO]) else 0.0
    else:
        return None
    return inc + LG[min(max(1, na), 4), min(pos-st[POS], MAXG-1)]

def advance(st, a, it, ca, pos):
    if a == SEARCH: st[CAT] = ca; st[NS] += 1; st[VB] = 0
    elif a == VIEW: st[LASTV] = it; st[VB] += 1
    elif a == ADD: st[NADD] += 1
    elif a == CHECK: st[HASCO] = 1
    elif a == OUT_: st[DONE] = 1
    st[LAST] = a; st[POS] = pos

def enumerate_parts(ev, sl):
    """Every grammar-legal partition with its score; None if the row blows a cap."""
    n = len(ev)
    S, lab, out = [], [0]*n, []
    nodes = 0
    def rec(pos, score, live):
        nonlocal nodes
        if len(out) > PCAP or nodes > NODECAP: return False
        nodes += 1
        if pos == n:
            out.append((score, lab[:])); return True
        a, it, ca, cs = ev[pos]
        if a == LOGIN:
            S.append(new_state(pos)); lab[pos] = len(S)-1
            r = rec(pos+1, score, live+1)
            S.pop()
            return r
        for j in range(len(S)):
            st = S[j]
            inc = legal_inc(st, a, it, ca, cs, pos, live)
            if inc is None: continue
            old = st[:]
            advance(st, a, it, ca, pos)
            nl = live - (1 if a == OUT_ else 0)
            lab[pos] = j
            if nl <= int(sl[pos+1]):     # every live session still needs its own logout
                if not rec(pos+1, score+inc, nl):
                    S[j][:] = old; return False
            S[j][:] = old
        return True
    ok = rec(0, 0.0, 0)
    if not ok or not out or len(out) > PCAP: return None
    return out

def beam_solve(ev, sl, beam=BEAM):
    """Fallback for the few rows whose legal set is too large to enumerate."""
    n = len(ev)
    cur = [(0.0, (), [])]
    for pos, (a, it, ca, cs) in enumerate(ev):
        nxt = []
        for sc, sts, lb in cur:
            if a == LOGIN:
                nxt.append((sc, sts+(tuple(new_state(pos)),), lb+[len(sts)])); continue
            live = sum(1 for x in sts if not x[DONE])
            for j in range(len(sts)):
                st = list(sts[j])
                inc = legal_inc(st, a, it, ca, cs, pos, live)
                if inc is None: continue
                advance(st, a, it, ca, pos)
                nl = live - (1 if a == OUT_ else 0)
                if nl > int(sl[pos+1]): continue
                ns = list(sts); ns[j] = tuple(st)
                nxt.append((sc+inc, tuple(ns), lb+[j]))
        if not nxt: return [0]*n
        nxt.sort(key=lambda x: -x[0]); cur = nxt[:beam]
    return cur[0][2]

def mbr(parts, n):
    """Exact MBR over the exact posterior: the partition best matching the true pairwise
    co-assignment marginals. MBRT flattens the posterior, which is measurably underconfident,
    so consensus is not dominated by the single most likely partition."""
    sc = np.array([p[0] for p in parts], np.float64)
    w = np.exp((sc-sc.max())/max(MBRT, 1e-6)); w /= w.sum()
    L = np.asarray([p[1] for p in parts])
    iu = np.triu_indices(n, 1)
    A = (L[:, iu[0]] == L[:, iu[1]])
    P = w @ A
    val = (A*P + (~A)*(1.0-P)).sum(1)
    return L[int(np.argmax(val))].tolist()

def solve(ev, sl):
    K = sum(1 for e in ev if e[0] == LOGIN)
    if K <= 1: return [0]*len(ev), 1
    parts = enumerate_parts(ev, sl)
    if parts is None: return beam_solve(ev, sl), -1
    if len(parts) == 1: return parts[0][1], 1
    return mbr(parts, len(ev)), len(parts)

def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    C = np.zeros((a.max()+1, b.max()+1))
    for x, y in zip(a, b): C[x, y] += 1
    def c2(v): return (v*(v-1)/2).sum()
    si, sj, sij = c2(C.sum(1)), c2(C.sum(0)), c2(C)
    exp = si*sj/(n*(n-1)/2); mx = (si+sj)/2
    return 1.0 if mx == exp else (sij-exp)/(mx-exp)

# ---------------- honest validation: rows held out of every fitted component ----------------
perm = np.random.RandomState(SEED).permutation(len(Etr))
VAL, FIT = perm[:NVAL], perm[NVAL:]
log('fitting on %d rows, validating on %d held-out rows' % (len(FIT), len(VAL)))
fit(FIT)

sc, sizes, nfb = [], [], 0
for i in VAL:
    p, npart = solve(Etr[i], SUFtr[i])
    if npart < 0: nfb += 1
    else: sizes.append(npart)
    sc.append(max(0.0, ari(p, Ytr[i])))
by = defaultdict(list)
for i, v in zip(VAL, sc): by[len(set(Ytr[i]))].append(v)
log('===== VALIDATION (held-out rows, send this back) =====')
log('MEAN ARI %.4f on %d rows | exactly recovered %.3f'
    % (np.mean(sc), len(VAL), np.mean([v >= 0.9999 for v in sc])))
log('by session count: %s' % {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())})
log('legal partitions per row: median %d p90 %d max %d | beam fallback on %d rows'
    % (np.median(sizes or [0]), np.percentile(sizes or [0], 90), max(sizes or [0]), nfb))
log('===== END VALIDATION =====')
if SKIPTEST: sys.exit(0)

log('refitting on all train rows for the submission...')
fit(np.arange(len(Etr)))
pred = [solve(e, s)[0] for e, s in zip(Ete, SUFte)]
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
