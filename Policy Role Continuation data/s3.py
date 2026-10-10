"""Policy role continuation: per-role linear classifiers + metric-optimal thresholds.

The metric is 0.8*macro_role_F1 + 0.2*exact_bundle_accuracy over 8 roles, so every role is
worth 0.1 of the score regardless of how rare it is. Prevalence runs from Actor .428 down to
Reversibility .0018 (21 train rows), which is why argmax-style models do badly: the
P(next|current) transition baseline reaches .440 exact accuracy but only .188 overall, because
five of the eight roles get F1 = 0. So this predicts each role independently and tunes a
per-role threshold directly against the real metric.

Implemented on numpy/scipy alone -- no sklearn -- so it cannot break on a version mismatch.
"""
import os, sys, re, time, math, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy import sparse as sp
from scipy.optimize import minimize

T0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time()-T0), *a, flush=True)

def parse_paths():
    if len(sys.argv) > 2 and not sys.argv[1].startswith('-f'):
        return Path(sys.argv[1]), Path(sys.argv[2])
    cands = ['./dataset/public',
             '/kaggle/input/datasets/devansh66/policy-role-continuation',
             '/kaggle/input/policy-role-continuation', '/kaggle/input', '.']
    pub = next((Path(c) for c in cands if (Path(c)/'train.csv').exists()),
               Path('./dataset/public'))
    out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)
NFOLD = int(os.environ.get('NFOLD', 5))
LAM = float(os.environ.get('LAM', 3.0))
# Averaging over regularisation strengths: the sweep peaked broadly (LAM 3-8 within noise of
# each other, 0.3 and 20 clearly worse), so no single value is well determined and the mean of
# several is steadier than betting on one.
LAMS = [float(x) for x in os.environ.get('LAMS', '1,3,8').split(',') if x]
RATE = int(os.environ.get('RATE', 1))
PRIOR = int(os.environ.get('PRIOR', 1))
NBIN = int(os.environ.get('NBIN', 4))
TAIL = int(os.environ.get('TAIL', 12))
EF1 = int(os.environ.get('EF1', 0))
MAXR = int(os.environ.get('MAXR', 2))
ROLES = ["Actor", "Compliance", "InstrumentType", "Objective",
         "Reference", "Resource", "Reversibility", "Time"]
RI = {r: i for i, r in enumerate(ROLES)}

train = pd.read_csv(PUB/'train.csv'); test = pd.read_csv(PUB/'test.csv')
for d in (train, test):
    for c in ('current_text', 'context', 'current_bundle'):
        d[c] = d[c].fillna('').astype(str)
log('train %s test %s' % (train.shape, test.shape))

Y = np.zeros((len(train), 8), np.int8)
for i, b in enumerate(train.next_bundle):
    for r in str(b).split('|'):
        if r in RI: Y[i, RI[r]] = 1
log('role prevalence: %s' % dict(zip(ROLES, np.round(Y.mean(0), 4))))

# ---------------- pseudo-document groups ----------------
# Test rows come from unseen legal instruments and the brief warns that a random split is not
# valid, but document ids are concealed and the file is shuffled (position_fraction does not
# increase down the rows). Neighbouring spans in one article do share most of their 55-token
# context window, so rows are linked when their context token sets overlap heavily.
def toks(s): return re.findall(r'[a-z]{3,}', str(s).lower())
CTOK = [set(toks(c)) for c in train.context]
dfc = Counter()
for s in CTOK: dfc.update(s)
rare = {w for w, n in dfc.items() if 2 <= n <= 60}
inv = defaultdict(list)
for i, s in enumerate(CTOK):
    for w in s:
        if w in rare: inv[w].append(i)
par = list(range(len(train)))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: par[ra] = rb
cand = Counter()
for w, rows in inv.items():
    if len(rows) > 40: continue
    for a in range(len(rows)):
        for b in range(a+1, len(rows)):
            cand[(rows[a], rows[b])] += 1
for (a, b), k in cand.items():
    if k < 2: continue
    ia, ib = CTOK[a], CTOK[b]
    u = len(ia | ib)
    if u and len(ia & ib)/u >= 0.35: union(a, b)
grp = np.array([find(i) for i in range(len(train))])
_, grp = np.unique(grp, return_inverse=True)
gs = Counter(grp)
log('pseudo-groups: %d | largest %d | singleton share %.3f | rows in multi-row groups %.3f'
    % (len(gs), max(gs.values()), sum(1 for v in gs.values() if v == 1)/len(gs),
       np.mean([gs[g] > 1 for g in grp])))

# ---------------- tf-idf, implemented directly ----------------
def analyze(s, kind, lo, hi):
    if kind == 'word':
        w = re.findall(r'\b\w\w+\b', s.lower())
        out = []
        for n in range(lo, hi+1):
            out += [' '.join(w[i:i+n]) for i in range(len(w)-n+1)]
        return out
    out = []
    for w in re.findall(r'\b\w+\b', s.lower()):
        p = ' '+w+' '
        for n in range(lo, hi+1):
            out += [p[i:i+n] for i in range(len(p)-n+1)]
    return out

def tf_fit(docs, kind, lo, hi, min_df, max_feat=None):
    dfq = Counter()
    for d in docs: dfq.update(set(analyze(d, kind, lo, hi)))
    it = [(w, n) for w, n in dfq.items() if n >= min_df]
    if max_feat and len(it) > max_feat:
        it.sort(key=lambda x: -x[1]); it = it[:max_feat]
    vocab = {w: i for i, (w, _) in enumerate(it)}
    n = len(docs)
    idf = np.zeros(len(vocab))
    for w, i in vocab.items(): idf[i] = math.log((1.0+n)/(1.0+dfq[w]))+1.0
    return vocab, idf, (kind, lo, hi)

def tf_apply(docs, model):
    vocab, idf, (kind, lo, hi) = model
    ind, ptr, dat = [], [0], []
    for d in docs:
        c = Counter(t for t in analyze(d, kind, lo, hi) if t in vocab)
        if c:
            ks = np.fromiter((vocab[k] for k in c), np.int32, len(c))
            vs = np.fromiter((1.0+math.log(v) for v in c.values()), np.float64, len(c))
            vs = vs*idf[ks]
            nrm = np.sqrt((vs*vs).sum())
            if nrm > 0: vs = vs/nrm
            ind.extend(ks.tolist()); dat.extend(vs.tolist())
        ptr.append(len(ind))
    return sp.csr_matrix((dat, ind, ptr), shape=(len(docs), len(vocab)))

def dense_feats(d):
    cb = np.zeros((len(d), 8), np.float32)
    for i, b in enumerate(d.current_bundle):
        for r in str(b).split('|'):
            if r in RI: cb[i, RI[r]] = 1
    pf = d.position_fraction.values.astype(np.float32)
    sl = np.log1p(d.span_length.values.astype(np.float32))
    nt = np.array([len(str(x).split()) for x in d.current_text], np.float32)
    nc = np.array([b.count('|')+1 for b in d.current_bundle], np.float32)
    return sp.csr_matrix(np.column_stack([
        cb, pf, pf*pf, (pf < 0.12).astype(np.float32), (pf > 0.88).astype(np.float32),
        sl, np.log1p(nt), nc, np.ones(len(d), np.float32)]))

def pbin(d):
    return np.minimum((d.position_fraction.values*NBIN).astype(int), NBIN-1)

def trans_prior(fit_df, y_fit, others, alpha=2.0):
    """P(role j in next_bundle | current_bundle), estimated on the FIT rows only.

    current_bundle -> next_bundle is the strongest single relation in the data (the transition
    baseline alone reaches .44 exact accuracy), but a linear model over one-hot current_bundle
    cannot express it as cleanly as the conditional rate itself. Fitted per fold so no row ever
    sees its own label."""
    base = y_fit.mean(0)
    tab = defaultdict(lambda: np.zeros(8))
    cnt = Counter()
    for b, yy in zip(fit_df.current_bundle, y_fit):
        tab[b] += yy; cnt[b] += 1
    tab2 = defaultdict(lambda: np.zeros(8))
    cnt2 = Counter()
    for b, pb, yy in zip(fit_df.current_bundle, pbin(fit_df), y_fit):
        tab2[(b, pb)] += yy; cnt2[(b, pb)] += 1
    def f(d):
        out = np.zeros((len(d), 16), np.float32)
        for i, (b, pb) in enumerate(zip(d.current_bundle, pbin(d))):
            n = cnt.get(b, 0)
            p1 = (tab[b]+alpha*base)/(n+alpha) if n else base
            k = (b, pb); n2 = cnt2.get(k, 0)
            # where the article currently is changes what comes next, so the prior is also
            # conditioned on a position bin, backing off to the bundle-only rate when sparse
            p2 = (tab2[k]+alpha*p1)/(n2+alpha) if n2 else p1
            out[i, :8] = p1; out[i, 8:] = p2
        return out
    return f(fit_df), [f(d) for d in others]

def tail_col(d):
    # the tokens immediately before the span carry more about what follows than the whole
    # 55-token window, where the signal is diluted
    return [' '.join(str(c).split()[-TAIL:]) for c in d.context]

def cross_col(d):
    return [b+'@'+str(p) for b, p in zip(d.current_bundle, pbin(d))]

def build(fit_df, others, y_fit=None):
    specs = [('current_text', 'word', 1, 2, 2, None),
             ('current_text', 'char', 3, 5, 3, 120000),
             ('context', 'word', 1, 2, 3, 80000)]
    F = [dense_feats(fit_df)]
    O = [[dense_feats(d)] for d in others]
    tm = tf_fit(tail_col(fit_df), 'word', 1, 2, 2, None)
    F.append(tf_apply(tail_col(fit_df), tm))
    for k, d in enumerate(others): O[k].append(tf_apply(tail_col(d), tm))
    if PRIOR and y_fit is not None:
        pf, po = trans_prior(fit_df, y_fit, others)
        F.append(sp.csr_matrix(pf))
        for k in range(len(others)): O[k].append(sp.csr_matrix(po[k]))
    for col, kind, lo, hi, mdf, mf in specs:
        m = tf_fit(fit_df[col].tolist(), kind, lo, hi, mdf, mf)
        F.append(tf_apply(fit_df[col].tolist(), m))
        for k, d in enumerate(others): O[k].append(tf_apply(d[col].tolist(), m))
    for colf in (lambda d: list(d.current_bundle), cross_col):
        cb = sorted(set(colf(fit_df)))
        cbi = {b: i for i, b in enumerate(cb)}
        def onehot(d, cbi=cbi, cb=cb, colf=colf):
            vals = colf(d)
            r = np.arange(len(d))
            keep = np.array([v in cbi for v in vals])
            c = np.array([cbi.get(v, 0) for v in vals])
            return sp.csr_matrix((np.ones(int(keep.sum())), (r[keep], c[keep])),
                                 shape=(len(d), len(cb)))
        F.append(onehot(fit_df))
        for k, d in enumerate(others): O[k].append(onehot(d))
    return sp.hstack(F).tocsr(), [sp.hstack(m).tocsr() for m in O]

def fit_logistic(X, y, lam=LAM):
    """L2 logistic regression with balanced class weights, solved with L-BFGS."""
    n, d = X.shape
    yy = np.where(y > 0, 1.0, -1.0)
    npos = max(int((y > 0).sum()), 1); nneg = max(n-npos, 1)
    w = np.where(y > 0, 0.5*n/npos, 0.5*n/nneg)
    def f(b):
        z = X @ b[:d] + b[d]
        m = -yy*z
        loss = np.sum(w*np.logaddexp(0.0, m)) + 0.5*lam*np.dot(b[:d], b[:d])
        s = w*(-yy)/(1.0+np.exp(-m))
        g = np.empty(d+1)
        g[:d] = X.T @ s + lam*b[:d]
        g[d] = s.sum()
        return loss, g
    r = minimize(f, np.zeros(d+1), jac=True, method='L-BFGS-B',
                 options={'maxiter': 300, 'maxfun': 400})
    return r.x

def predict_proba(X, b):
    z = X @ b[:-1] + b[-1]
    return 1.0/(1.0+np.exp(-z))

def prior_correct(P, pri):
    """Undo the balanced-class-weight prior so the probabilities mean what they say.

    Training with balanced weights implies a 0.5 prior; the expected-F1 rule below needs
    probabilities on the true prior, so each column is shifted back to its own prevalence.
    """
    out = np.empty_like(P)
    for j in range(P.shape[1]):
        pi = min(max(float(pri[j]), 1e-6), 1-1e-6)
        p = np.clip(P[:, j], 1e-9, 1-1e-9)
        out[:, j] = pi*p/(pi*p + (1.0-pi)*(1.0-p))
    return out

def ef1_thresholds(P):
    """Per-role cutoffs that maximise EXPECTED F1 on the rows being predicted.

    For a nested set taken in probability order, E[F1] ~= 2*sum(p in S)/(|S| + sum(all p)).
    Maximising that over the prefix length gives the operating point directly from the
    probabilities of the rows at hand, so it re-derives itself on the test distribution instead
    of importing a cutoff (or a rate) tuned on train. That matters here: absolute cutoffs put
    test Compliance at 1.9x its train prevalence.
    """
    th = np.zeros(P.shape[1])
    for j in range(P.shape[1]):
        p = np.sort(P[:, j])[::-1]
        S = p.sum()
        k = np.arange(1, len(p)+1)
        ef1 = 2.0*np.cumsum(p)/(k+S)
        b = int(np.argmax(ef1))
        th[j] = p[b] - 1e-12
    return th

def cap_roles(pred, P, TH, maxr):
    """No observed bundle has more than two roles, so never emit three."""
    if maxr <= 0: return pred
    over = np.where(pred.sum(1) > maxr)[0]
    for i in over:
        idx = np.where(pred[i])[0]
        keep = idx[np.argsort(-(P[i, idx]-TH[idx]))][:maxr]
        pred[i] = False
        pred[i, keep] = True
    return pred

def score_of(P, TH, truth, maxr=0):
    pred = P >= TH[None, :]
    empty = np.where(~pred.any(1))[0]
    if len(empty):
        pick = np.argmax(P[empty]/np.maximum(TH, 1e-9)[None, :], 1)
        pred[empty, pick] = True
    if maxr: pred = cap_roles(pred, P, TH, maxr)
    f1 = []
    for j in range(8):
        tp = int((pred[:, j] & (truth[:, j] == 1)).sum())
        fp = int((pred[:, j] & (truth[:, j] == 0)).sum())
        fn = int((~pred[:, j] & (truth[:, j] == 1)).sum())
        f1.append(0.0 if 2*tp+fp+fn == 0 else 2.0*tp/(2*tp+fp+fn))
    ex = float((pred == (truth == 1)).all(1).mean())
    return 0.8*float(np.mean(f1))+0.2*ex, float(np.mean(f1)), ex, f1, pred

def group_folds(g, k):
    u = np.unique(g)
    rs = np.random.RandomState(SEED); rs.shuffle(u)
    asg = {x: i % k for i, x in enumerate(u)}
    return np.array([asg[x] for x in g])

fold = group_folds(grp, NFOLD)
OOF = np.zeros((len(train), 8))
for f in range(NFOLD):
    tr_i = np.where(fold != f)[0]; va_i = np.where(fold == f)[0]
    Xtr, (Xva,) = build(train.iloc[tr_i], [train.iloc[va_i]], Y[tr_i])
    for j in range(8):
        y = Y[tr_i, j]
        if y.sum() < 3:
            OOF[va_i, j] = float(y.mean()); continue
        OOF[va_i, j] = np.mean([predict_proba(Xva, fit_logistic(Xtr, y, L)) for L in LAMS], 0)
    log('  fold %d/%d done (%d train / %d val)' % (f+1, NFOLD, len(tr_i), len(va_i)))

TH = np.full(8, 0.5)
best = score_of(OOF, TH, Y)[0]
for _ in range(8):
    moved = False
    for j in range(8):
        for t in np.unique(np.quantile(OOF[:, j], np.linspace(0.30, 0.9995, 80))):
            T2 = TH.copy(); T2[j] = t
            s = score_of(OOF, T2, Y)[0]
            if s > best + 1e-9: best, TH, moved = s, T2, True
    if not moved: break
s, mf, ex, f1, pr = score_of(OOF, TH, Y)
RATES = pr.mean(0)

def rate_thresholds(P, rates):
    return np.array([np.quantile(P[:, j], max(0.0, 1.0-rates[j])) for j in range(8)])

# The submission uses RATE-matched cutoffs, so the rates themselves are what should be tuned.
# Previously they were read off the absolute-threshold solution, i.e. optimised in the wrong
# space. Coordinate ascent directly on the predicted rate per role fixes that mismatch.
def tune_rates(P, truth, init):
    r = init.copy()
    b = score_of(P, rate_thresholds(P, r), truth, MAXR)[0]
    base = truth.mean(0)
    for _ in range(8):
        moved = False
        for j in range(8):
            grid = np.unique(np.clip(np.concatenate([
                base[j]*np.array([0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.5, 5.0]),
                r[j]*np.array([0.6, 0.8, 0.9, 1.1, 1.25, 1.5])]), 1e-4, 0.9))
            for v in grid:
                r2 = r.copy(); r2[j] = v
                sc = score_of(P, rate_thresholds(P, r2), truth, MAXR)[0]
                if sc > b + 1e-9: b, r, moved = sc, r2, True
        if not moved: break
    return r, b

RATES, brate = tune_rates(OOF, Y, pr.mean(0))
sr = score_of(OOF, rate_thresholds(OOF, RATES), Y, MAXR)
PRI = Y.mean(0)
OOFC = prior_correct(OOF, PRI)
TE = ef1_thresholds(OOFC)
se = score_of(OOFC, TE, Y, MAXR)
log('OOF: tuned-threshold %.4f | rate-matched %.4f | expected-F1 %.4f' % (s, sr[0], se[0]))
log('  tuned rates: %s' % dict(zip(ROLES, np.round(RATES, 4))))
log('  rate-matched per-role F1: %s' % dict(zip(ROLES, np.round(sr[3], 3))))
log('  expected-F1 per-role F1: %s' % dict(zip(ROLES, np.round(se[3], 3))))
log('  expected-F1 macro %.4f exact %.4f' % (se[1], se[2]))
log('===== VALIDATION (grouped OOF) =====')
log('SCORE %.4f = 0.8 * %.4f macro F1 + 0.2 * %.4f exact' % (s, mf, ex))
log('per-role F1: %s' % dict(zip(ROLES, np.round(f1, 3))))
log('thresholds:  %s' % dict(zip(ROLES, np.round(TH, 4))))
log('predicted vs true role rate: %s' % {
    r: (round(float(pr[:, i].mean()), 3), round(float(Y[:, i].mean()), 3))
    for i, r in enumerate(ROLES)})
log('===== END VALIDATION =====')

Xall, (Xte,) = build(train, [test], Y)
PT = np.zeros((len(test), 8))
for j in range(8):
    y = Y[:, j]
    if y.sum() < 3:
        PT[:, j] = float(y.mean()); continue
    PT[:, j] = np.mean([predict_proba(Xte, fit_logistic(Xall, y, L)) for L in LAMS], 0)

# The absolute cutoffs do not transfer: on test they put Compliance at .34 against a true
# prevalence of .10, because probabilities shift on unseen legal instruments. Selecting the
# same predicted RATE per role is invariant to that shift and keeps the F1 operating point.
if EF1:
    PS = prior_correct(PT, PRI)
    THT = ef1_thresholds(PS)
elif RATE:
    PS, THT = PT, rate_thresholds(PT, RATES)
else:
    PS, THT = PT, TH
log('test thresholds: %s' % dict(zip(ROLES, np.round(THT, 5))))
pred = PS >= THT[None, :]
empty = np.where(~pred.any(1))[0]
if len(empty):
    pred[empty, np.argmax(PS[empty]/np.maximum(THT, 1e-9)[None, :], 1)] = True
pred = cap_roles(pred, PS, THT, MAXR)
bundles = ['|'.join([ROLES[j] for j in range(8) if pred[i, j]]) for i in range(len(test))]
sub = pd.DataFrame({'id': test.id.values, 'next_bundle': bundles})
assert list(sub.columns) == ['id', 'next_bundle'] and len(sub) == len(test)
assert sub.id.tolist() == test.id.tolist() and not sub.next_bundle.isna().any()
for b in sub.next_bundle:
    p = b.split('|')
    assert p and all(x in RI for x in p) and len(set(p)) == len(p)
    assert p == sorted(p, key=lambda x: RI[x])
sub.to_csv(OUT, index=False)
log('wrote %s %d rows | distinct bundles %d | mean roles/bundle %.2f'
    % (OUT, len(sub), sub.next_bundle.nunique(), np.mean([b.count('|')+1 for b in bundles])))
log('test predicted role rates: %s' % dict(zip(ROLES, np.round(pred.mean(0), 4))))
log('done')
