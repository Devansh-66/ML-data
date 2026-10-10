"""Interleaved session disentanglement: exact enumeration of legal partitions + MBR decoding."""
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
MBRT = float(os.environ.get('MBRT', 1.0))
PCAP = int(os.environ.get('PCAP', 250000))
NODECAP = int(os.environ.get('NODECAP', 2000000))
BEAM = int(os.environ.get('BEAM', 300))
TOPM = int(os.environ.get('TOPM', 40))
SKIPTEST = int(os.environ.get('SKIPTEST', 0))
MAXG = 16

LOGIN, SEARCH, VIEW, ADD, CHECK, OUT_ = 0, 1, 2, 3, 4, 5
AI = {'login': LOGIN, 'search': SEARCH, 'view_item': VIEW,
      'add_to_cart': ADD, 'checkout': CHECK, 'logout': OUT_}
CATIDX = {}
MAXS = MAXV = 0
RELAX = 0

train = pd.read_csv(PUB/'train.csv'); test = pd.read_csv(PUB/'test.csv')
log('train %s test %s' % (train.shape, test.shape))

RAWtr = [json.loads(ev) for ev in train.events]
RAWte = [json.loads(ev) for ev in test.events]
Ytr = [[int(x) for x in c.split(',')] for c in train.cluster_labels]

def learn_schema():
    global CATIDX, MAXS, MAXV
    pair = defaultdict(Counter)
    ms = mv = 1
    for raw, lab in zip(RAWtr, Ytr):
        s = defaultdict(list)
        for e, c in zip(raw, lab): s[c].append(e)
        for seq in s.values():
            cur, ns, vb = None, 0, 0
            for e in seq:
                a = e['action']
                if a == 'search':
                    cur = e['category']; ns += 1
                    mv = max(mv, vb); vb = 0
                elif a == 'view_item':
                    vb += 1
                    if cur: pair[cur][e['item_id']//100] += 1
            mv = max(mv, vb); ms = max(ms, ns)
    CATIDX = {c: cnt.most_common(1)[0][0] for c, cnt in pair.items()}
    MAXS, MAXV = ms, mv
    log('learned schema: %d categories | max searches/session %d | max views/block %d'
        % (len(CATIDX), MAXS, MAXV))
learn_schema()

def parse(raws):
    return [[(AI[x['action']], x.get('item_id', 0),
              CATIDX.get(x.get('category'), 0), x.get('cart_size', -1))
             for x in raw] for raw in raws]

Etr, Ete = parse(RAWtr), parse(RAWte)

def suffix(ev):
    n = len(ev)
    sl = np.zeros(n+1, np.int32)
    for i in range(n-1, -1, -1): sl[i] = sl[i+1] + (1 if ev[i][0] == OUT_ else 0)
    return sl
SUFtr = [suffix(e) for e in Etr]; SUFte = [suffix(e) for e in Ete]

def fit_phi(rows, iters=4000, lr=4.0):
    pat = Counter()
    for r in rows:
        ev, lab = Etr[r], Ytr[r]
        last, done = {}, set()
        for i, (e, c) in enumerate(zip(ev, lab)):
            if e[0] != LOGIN and last:
                live = [k for k in last if k not in done][:4]
                if len(live) > 1 and c in live:
                    g = tuple(min(i-last[k], MAXG-1) for k in live)
                    pat[(len(live), g, live.index(c))] += 1
            last[c] = i
            if e[0] == OUT_: done.add(c)
    NA = np.array([k[0] for k in pat], np.int64)
    TI = np.array([k[2] for k in pat], np.int64)
    W = np.array([pat[k] for k in pat], np.float64)
    G = np.full((len(pat), 4), -1, np.int64)
    for r_, k in enumerate(pat): G[r_, :len(k[1])] = k[1]
    M = G >= 0
    Gc = np.where(M, G, 0)
    NAc = np.repeat(np.minimum(NA, 4)[:, None], 4, 1)
    ix = np.arange(len(pat))
    phi = np.zeros((5, MAXG))
    Wn = W/W.sum()
    pr = None
    for _ in range(iters):
        z = np.where(M, phi[NAc, Gc], -1e9)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)*M
        pr = e/np.maximum(e.sum(1, keepdims=True), 1e-12)
        tgt = np.zeros_like(pr); tgt[ix, TI] = 1.0
        d = (tgt-pr)*Wn[:, None]*M
        gr = np.zeros_like(phi)
        np.add.at(gr, (NAc[M], Gc[M]), d[M])
        phi = phi + lr*gr
        phi = phi - phi[:, :1]
    acc = float((W*(pr.argmax(1) == TI)).sum()/W.sum())
    log('  interleaving logit: %d patterns / %d decisions | pick acc %.4f'
        % (len(pat), int(W.sum()), acc))
    return phi

def fit(rows):
    global LCAT_SAME, LCAT_DIFF, LCO, LNOCO, PHI, EPHI
    csame = cdiff = 1.0
    nco = ncart = 1.0
    for r in rows:
        ev, lab = Etr[r], Ytr[r]
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
    p = csame/(csame+cdiff)
    nc = max(len(CATIDX)-1, 1)
    LCAT_SAME, LCAT_DIFF = math.log(p), math.log((1.0-p)/nc)
    q = nco/ncart
    LCO, LNOCO = math.log(q), math.log(max(1.0-q, 1e-9))
    log('  fit %d rows: P(cat repeat) %.4f | P(checkout | cart) %.4f' % (len(rows), p, q))
    PHI = fit_phi(rows)
    EPHI = np.exp(PHI)

LAST, CAT, LASTV, NADD, DONE, POS, NS, VB, HASCO = range(9)
def new_state(pos): return [LOGIN, 0, -1, 0, 0, pos, 0, 0, 0]

def legal_inc(st, a, it, ca, cs, pos, na, den):
    if st[DONE]: return None
    if st[LAST] == CHECK and a != OUT_: return None
    if a == SEARCH:
        if st[LAST] not in (LOGIN, VIEW, ADD) or (not RELAX and st[NS] >= MAXS): return None
        inc = (LCAT_SAME if ca == st[CAT] else LCAT_DIFF) if st[CAT] else 0.0
    elif a == VIEW:
        if st[LAST] not in (SEARCH, VIEW, ADD): return None
        if not st[CAT] or it//100 != st[CAT] or (not RELAX and st[VB] >= MAXV): return None
        inc = 0.0
    elif a == ADD:
        if st[LAST] != VIEW or it != st[LASTV]: return None
        inc = 0.0
    elif a == CHECK:
        if st[LAST] not in (VIEW, ADD) or cs != st[NADD]: return None
        inc = LCO
    elif a == OUT_:
        if st[LAST] not in (VIEW, ADD, CHECK): return None
        inc = LNOCO if (st[NADD] > 0 and not st[HASCO]) else 0.0
    else:
        return None
    return inc + PHI[min(max(1, na), 4), min(pos-st[POS], MAXG-1)] - den

def advance(st, a, it, ca, pos):
    if a == SEARCH: st[CAT] = ca; st[NS] += 1; st[VB] = 0
    elif a == VIEW: st[LASTV] = it; st[VB] += 1
    elif a == ADD: st[NADD] += 1
    elif a == CHECK: st[HASCO] = 1
    elif a == OUT_: st[DONE] = 1
    st[LAST] = a; st[POS] = pos

def denom(states, pos, live):
    na = min(max(live, 1), 4)
    ss = 0.0
    for st in states:
        if not st[DONE]: ss += EPHI[na, min(pos-st[POS], MAXG-1)]
    return math.log(ss) if ss > 0 else 0.0

def enumerate_parts(ev, sl):
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
        den = denom(S, pos, live)
        for j in range(len(S)):
            st = S[j]
            inc = legal_inc(st, a, it, ca, cs, pos, live, den)
            if inc is None: continue
            old = st[:]
            advance(st, a, it, ca, pos)
            nl = live - (1 if a == OUT_ else 0)
            lab[pos] = j
            if nl <= int(sl[pos+1]):
                if not rec(pos+1, score+inc, nl):
                    S[j][:] = old; return False
            S[j][:] = old
        return True
    ok = rec(0, 0.0, 0)
    if not ok or len(out) > PCAP: return None
    return out

def beam_solve(ev, sl, beam=BEAM):
    cur = [(0.0, (), [])]
    for pos, (a, it, ca, cs) in enumerate(ev):
        nxt = []
        for sc, sts, lb in cur:
            if a == LOGIN:
                nxt.append((sc, sts+(tuple(new_state(pos)),), lb+[len(sts)])); continue
            live = sum(1 for x in sts if not x[DONE])
            den = denom(sts, pos, live)
            for j in range(len(sts)):
                st = list(sts[j])
                inc = legal_inc(st, a, it, ca, cs, pos, live, den)
                if inc is None: continue
                advance(st, a, it, ca, pos)
                nl = live - (1 if a == OUT_ else 0)
                if nl > int(sl[pos+1]): continue
                ns = list(sts); ns[j] = tuple(st)
                nxt.append((sc+inc, tuple(ns), lb+[j]))
        if not nxt: return []
        best = {}
        for sc2, sts2, lb2 in nxt:
            if sts2 not in best or sc2 > best[sts2][0]: best[sts2] = (sc2, sts2, lb2)
        cur = sorted(best.values(), key=lambda x: -x[0])[:beam]
    return [(sc, lb) for sc, sts, lb in cur]

def mbr(parts, n):
    sc = np.array([p[0] for p in parts], np.float64)
    o = np.argsort(-sc)[:max(TOPM, 2)]
    sc = sc[o]
    L = np.asarray([parts[i][1] for i in o])
    w = np.exp((sc-sc.max())/max(MBRT, 1e-6)); w /= w.sum()
    m, K = len(L), int(L.max())+1
    OH = np.zeros((m, n, K), np.float64)
    for i in range(m): OH[i, np.arange(n), L[i]] = 1.0
    C = np.einsum('ink,jnl->ijkl', OH, OH)
    def c2(x): return x*(x-1.0)/2.0
    sij = c2(C).sum((2, 3))
    a = c2(OH.sum(1)).sum(1)
    exp = np.outer(a, a)/c2(float(n))
    den = (a[:, None]+a[None, :])/2.0 - exp
    small = np.abs(den) < 1e-12
    ARI = np.where(small, 1.0, (sij-exp)/np.where(small, 1.0, den))
    return L[int(np.argmax(ARI @ w))].tolist()

def crude(ev):
    n = len(ev); lab = [0]*n; last = []; done = []
    for pos, (a, it, ca, cs) in enumerate(ev):
        if a == LOGIN:
            last.append(pos); done.append(False); lab[pos] = len(last)-1; continue
        if not last: continue
        cand = [j for j in range(len(last)) if not done[j]] or list(range(len(last)))
        j = max(cand, key=lambda z: pos-last[z])
        last[j] = pos; lab[pos] = j
        if a == OUT_: done[j] = True
    return lab

def solve(ev, sl):
    global RELAX
    K = sum(1 for e in ev if e[0] == LOGIN)
    if K <= 1: return [0]*len(ev), 1, 0
    parts = enumerate_parts(ev, sl)
    if parts is None:
        bp = beam_solve(ev, sl)
        if bp: return (mbr(bp, len(ev)) if len(bp) > 1 else bp[0][1]), -1, 1
        return crude(ev), 0, 1
    if not parts:
        RELAX = 1
        try:
            parts = enumerate_parts(ev, sl)
            if parts is None:
                bp = beam_solve(ev, sl)
                if bp: return (mbr(bp, len(ev)) if len(bp) > 1 else bp[0][1]), -1, 2
                parts = []
        finally:
            RELAX = 0
        if not parts: return crude(ev), 0, 3
        return (mbr(parts, len(ev)) if len(parts) > 1 else parts[0][1]), len(parts), 2
    if len(parts) == 1: return parts[0][1], 1, 0
    return mbr(parts, len(ev)), len(parts), 0

def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    C = np.zeros((a.max()+1, b.max()+1))
    for x, y in zip(a, b): C[x, y] += 1
    def c2(v): return (v*(v-1)/2).sum()
    si, sj, sij = c2(C.sum(1)), c2(C.sum(0)), c2(C)
    exp = si*sj/(n*(n-1)/2); mx = (si+sj)/2
    return 1.0 if mx == exp else (sij-exp)/(mx-exp)

perm = np.random.RandomState(SEED).permutation(len(Etr))
VAL, FIT = perm[:NVAL], perm[NVAL:]
log('fitting on %d rows, validating on %d held-out rows' % (len(FIT), len(VAL)))
fit(FIT)

sc, sizes, nfb, stat = [], [], 0, Counter()
for i in VAL:
    p, npart, stt = solve(Etr[i], SUFtr[i])
    stat[stt] += 1
    if npart < 0: nfb += 1
    else: sizes.append(npart)
    sc.append(max(0.0, ari(p, Ytr[i])))
by = defaultdict(list)
for i, v in zip(VAL, sc): by[len(set(Ytr[i]))].append(v)
log('===== VALIDATION (held-out rows) =====')
log('MEAN ARI %.4f on %d rows | exactly recovered %.3f'
    % (np.mean(sc), len(VAL), np.mean([v >= 0.9999 for v in sc])))
log('by session count: %s' % {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())})
log('legal partitions: median %d p90 %d max %d | beam fallback %d rows'
    % (np.median(sizes or [0]), np.percentile(sizes or [0], 90), max(sizes or [0]), nfb))
log('path: %d exact, %d over-cap beam, %d relaxed caps, %d crude'
    % (stat[0], stat[1], stat[2], stat[3]))
log('===== END VALIDATION =====')
if SKIPTEST: sys.exit(0)

log('refitting on all train rows...')
fit(np.arange(len(Etr)))
res = [solve(e, s) for e, s in zip(Ete, SUFte)]
pred = [r[0] for r in res]
ts = Counter(r[2] for r in res)
log('TEST path: %d exact, %d over-cap beam, %d relaxed caps, %d crude'
    % (ts[0], ts[1], ts[2], ts[3]))
log('GENERALISATION: %s' % ('clean' if (ts[2]+ts[3]) == 0 else
                            'WARNING: %d test rows broke an inferred constraint' % (ts[2]+ts[3])))
sub = pd.DataFrame({'id': test.id.values,
                    'cluster_labels': [','.join(map(str, p)) for p in pred]})
assert list(sub.columns) == ['id', 'cluster_labels'] and len(sub) == len(test)
assert sub.id.tolist() == test.id.tolist() and not sub.cluster_labels.isna().any()
for s, n_ in zip(sub.cluster_labels, test.num_events):
    q = s.split(',')
    assert len(q) == n_ and all(x.isdigit() for x in q)
sub.to_csv(OUT, index=False)
log('wrote %s %d rows | mean clusters %.2f'
    % (OUT, len(sub), np.mean([len(set(p)) for p in pred])))
log('done')
