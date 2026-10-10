"""Coalition trace reconstruction: per-meeting NMF source attribution + key-shared sequence model.

Test meetings are disjoint from train, so nothing here may depend on a voice identity. Every
feature is a RELATIVE quantity within its own row (energy share, similarity rank), and the
sequence model applies the same weights to all four keys, so a meeting's identities cannot
be memorised even in principle.
"""
import os, sys, time, math, random, wave
from pathlib import Path
import numpy as np, pandas as pd

T0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time()-T0), *a, flush=True)

def parse_paths():
    if len(sys.argv) > 2 and not sys.argv[1].startswith('-f'):
        return Path(sys.argv[1]), Path(sys.argv[2])
    cands = ['./dataset/public',
             '/kaggle/input/datasets/devansh66/reference-key-coalition',
             '/kaggle/input/reference-key-coalition', '/kaggle/input', '.']
    pub = next((Path(c) for c in cands if (Path(c)/'train.csv').exists()),
               Path('./dataset/public'))
    out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)

SR, NBIN, BINS = 8000, 24, 0.5
NFFT, HOP = 256, 80                     # 32 ms window, 10 ms hop
FPB = int(BINS*SR/HOP)                  # 50 frames per bin
RANK = int(os.environ.get('RANK', 12))   # dictionary components per reference key. Dropping
                 # this to 4 was tried on the theory that 12 lets any key reconstruct any
                 # spectrum; the share AUC was unchanged (.647 vs .645), so 12 stays -- it is
                 # what the 0.44 configuration used.
NMFIT = int(os.environ.get('NMFIT', 40))
EPOCHS = int(os.environ.get('EPOCHS', 120))
NSEED = int(os.environ.get('NSEED', 3))
TIME_LIMIT = float(os.environ.get('TIME_LIMIT', 4500))
SUBSET = int(os.environ.get('SUBSET', 0))

train = pd.read_csv(PUB/'train.csv'); test = pd.read_csv(PUB/'test.csv')
if SUBSET:
    train = train.groupby('case_id').head(SUBSET).reset_index(drop=True)
    test = test.groupby('case_id').head(SUBSET).reset_index(drop=True)
log('train %s test %s | %d train meetings, %d test meetings'
    % (train.shape, test.shape, train.case_id.nunique(), test.case_id.nunique()))

# ---------------- signal front end ----------------
def melbank(nmel=40):
    f = np.linspace(0, SR/2, NFFT//2+1)
    m = 2595*np.log10(1+f/700); lo, hi = 0, m[-1]
    c = np.linspace(lo, hi, nmel+2)
    B = np.zeros((nmel, len(f)))
    for i in range(nmel):
        l, ctr, r = c[i], c[i+1], c[i+2]
        B[i] = np.clip(np.minimum((m-l)/(ctr-l+1e-9), (r-m)/(r-ctr+1e-9)), 0, None)
    return B/(B.sum(1, keepdims=True)+1e-9)
MEL = melbank()
WIN = np.hanning(NFFT).astype(np.float32)

def read_wav(p):
    with wave.open(str(PUB/p), 'rb') as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return a.astype(np.float32)/32768.0

def spec(x):
    """Power spectrogram. NMF operates here because overlapping voices add in power."""
    n = 1+max(0, (len(x)-NFFT))//HOP
    idx = np.arange(NFFT)[None] + HOP*np.arange(n)[:, None]
    F = np.fft.rfft(x[idx]*WIN, axis=1)
    return (np.abs(F)**2).T.astype(np.float64) + 1e-10      # (freq, frames)

def nmf_dict(V, r, it=60, seed=0):
    """KL-divergence NMF, multiplicative updates. Learns a spectral dictionary for one key."""
    rs = np.random.RandomState(seed)
    F, T = V.shape
    W = rs.rand(F, r)+0.1; H = rs.rand(r, T)+0.1
    for _ in range(it):
        R = W@H + 1e-10
        H *= (W.T@(V/R))/(W.sum(0)[:, None]+1e-10)
        R = W@H + 1e-10
        W *= ((V/R)@H.T)/(H.sum(1)[None]+1e-10)
        W /= W.sum(0, keepdims=True)+1e-10
    return W

def nmf_activate(V, W, it=NMFIT, seed=0):
    """H >= 0 with the dictionary FIXED: attributes each query frame's power to the keys."""
    rs = np.random.RandomState(seed)
    H = rs.rand(W.shape[1], V.shape[1])+0.1
    ws = W.sum(0)[:, None]+1e-10
    for _ in range(it):
        H *= (W.T@(V/(W@H+1e-10)))/ws
    return H

DCT = np.cos(np.pi/40*(np.arange(40)+0.5)[None]*np.arange(1, 14)[:, None])

def mfcc(V):
    """Identity lives in the spectral envelope, so cepstral coefficients are the right space
    for a who-is-this comparison; per-frame log-mel cosine mostly measures phonetic content."""
    M = np.log(MEL@V + 1e-8)
    C = DCT@M
    return C/(np.linalg.norm(C, axis=0, keepdims=True)+1e-9)

PWIN, PLO, PHI = 512, 20, 200            # 64 ms window; 40 Hz .. 400 Hz
def f0_track(x):
    n = 1+max(0, (len(x)-PWIN))//HOP
    idx = np.arange(PWIN)[None] + HOP*np.arange(n)[:, None]
    fr = x[idx]*np.hanning(PWIN)
    F = np.fft.rfft(fr, n=2*PWIN, axis=1)
    ac = np.fft.irfft(np.abs(F)**2, axis=1)[:, :PWIN]
    ac /= (ac[:, :1]+1e-9)
    seg = ac[:, PLO:PHI]
    lag = seg.argmax(1)+PLO
    return SR/lag, seg.max(1)

def harm_ratio(V, f0s):
    """Comb-filter energy at each key's reference F0. Additive mixtures keep every speaker's
    harmonic series intact, so this survives overlap where a single-pitch tracker does not."""
    freqs = np.linspace(0, SR/2, V.shape[0])
    tot = V.sum(0)+1e-10
    out = np.zeros((len(f0s), V.shape[1]))
    for j, f0 in enumerate(f0s):
        if not np.isfinite(f0) or f0 <= 0: continue
        m = np.zeros(V.shape[0], bool)
        for h in range(1, int(SR/2/f0)+1):
            m |= np.abs(freqs-h*f0) < max(f0*0.12, 25.0)
        out[j] = V[m].sum(0)/tot
    return out

def zrow(a):
    return (a - a.mean(0, keepdims=True))/(a.std(0, keepdims=True)+1e-6)

_DICT = {}
def case_dict(row):
    """Per-meeting dictionary: RANK components per key plus a flat channel that absorbs the
    broadband floor so silence is not forced onto a voice."""
    k = row.case_id
    if k in _DICT: return _DICT[k]
    Ws, cep, mel, f0s = [], [], [], []
    for j in range(4):
        x = read_wav(row['voice_%d_path' % j])
        V = spec(x)
        Ws.append(nmf_dict(V, RANK, seed=SEED+j))
        cep.append(mfcc(V))
        M = np.log(MEL@V + 1e-8); M -= M.mean(0, keepdims=True)
        mel.append(M/(np.linalg.norm(M, axis=0, keepdims=True)+1e-9))
        f0, cl = f0_track(x)
        ok = cl > np.percentile(cl, 60)
        f0s.append(float(np.median(f0[ok])) if ok.any() else np.nan)
    flat = np.ones((NFFT//2+1, 1)); flat /= flat.sum()
    W = np.concatenate(Ws+[flat], 1)
    own = np.concatenate([[j]*RANK for j in range(4)]+[[4]])
    ctr = [c.mean(1)/(np.linalg.norm(c.mean(1))+1e-9) for c in cep]
    _DICT[k] = (W, own, cep, mel, ctr, np.array(f0s))
    return _DICT[k]

def row_feats(row):
    """(24, 4, D) relative features. Nothing absolute or meeting-specific survives here."""
    W, own, refcep, refmel, refctr, f0s = case_dict(row)
    x = read_wav(row.query_path)
    V = spec(x)
    T = min(V.shape[1], NBIN*FPB); V = V[:, :T]
    H = nmf_activate(V, W)
    cw = W.sum(0)
    E = np.stack([(cw[own == j][:, None]*H[own == j]).sum(0) for j in range(5)])   # (5,frames)
    tot = E.sum(0)+1e-10
    SH = E[:4]/tot                                        # per-frame energy share per key
    C = mfcc(V)
    COS = np.stack([(refcep[j].T@C).max(0) for j in range(4)])   # nearest ref frame per key
    M = np.log(MEL@V + 1e-8); M -= M.mean(0, keepdims=True)
    M /= np.linalg.norm(M, axis=0, keepdims=True)+1e-9
    # the log-mel cosine measured the strongest of any single cue (AUC .701); the cepstral
    # version alone read .668, so both are kept rather than one replacing the other
    CM = np.stack([(refmel[j].T@M).max(0) for j in range(4)])
    HR = harm_ratio(V, f0s)[:, :T]
    lt = np.log(tot); lt = (lt-np.median(lt))/(np.std(lt)+1e-9)
    arg = SH.argmax(0)
    A = np.zeros((NBIN, 4, 8), np.float64)                # cues to be z-scored across the row
    B_ = np.zeros((NBIN, 4, 7), np.float64)
    for b in range(NBIN):
        s = slice(b*FPB, min((b+1)*FPB, T))
        if s.start >= T:
            A[b], B_[b] = A[b-1], B_[b-1]; continue
        sh, co, hr, e = SH[:, s], COS[:, s], HR[:, s], E[:4, s]
        cm = CM[:, s]
        w = tot[s]/(tot[s].sum()+1e-12)
        emb = C[:, s]@w; emb /= np.linalg.norm(emb)+1e-9   # energy-weighted bin identity
        ms = np.array([float(refctr[j]@emb) for j in range(4)])
        vote = np.array([(arg[s] == j).mean() for j in range(4)])
        rk = np.argsort(np.argsort(-sh.mean(1)))/3.0
        for j in range(4):
            o = [k for k in range(4) if k != j]
            A[b, j] = [sh[j].mean(), co[j].mean()-co[o].mean(1).max(), hr[j].mean(),
                       ms[j]-ms[o].max(), co[j].max(), sh[j].mean()-sh[o].mean(0).max(),
                       cm[j].mean()-cm[o].mean(1).max(), cm[j].max()]
            B_[b, j] = [sh[j].max(), np.percentile(co[j], 90),
                        hr[j].mean()-hr[o].mean(1).max(), ms[j],
                        vote[j], rk[j], lt[s].mean()]
    return np.concatenate([A, zrow(A), B_], -1).astype(np.float32)

def build(df):
    X = None
    for i, (_, r) in enumerate(df.iterrows()):
        f = row_feats(r)
        if X is None: X = np.zeros((len(df), NBIN, 4, f.shape[-1]), np.float32)
        X[i] = f
        if (i+1) % 50 == 0: log('  features %d/%d' % (i+1, len(df)))
    return X

def decode_labels(df):
    Y = np.zeros((len(df), NBIN, 4), np.float32)
    for i, c in enumerate(df.activity_code.values):
        for b, ch in enumerate(c[1:]):
            m = int(ch, 16)
            for j in range(4): Y[i, b, j] = (m >> j) & 1
    return Y

# ---------------- the exact competition metric ----------------
def transition_f1(P, Tt):
    """Entry/exit events per key, matched within one boundary, nearest unused first."""
    def ev(a):
        z = np.concatenate([[0], a, [0]])
        d = np.diff(z)
        return [(i, 1) for i in np.where(d > 0)[0]], [(i, -1) for i in np.where(d < 0)[0]]
    matched = np_pred = np_true = 0
    for j in range(4):
        pe = sum(ev(P[:, j]), []); te = sum(ev(Tt[:, j]), [])
        np_pred += len(pe); np_true += len(te)
        used = [False]*len(te)
        for (pb, pt) in sorted(pe):
            best, bd = -1, 99
            for k, (tb, tt) in enumerate(te):
                if used[k] or tt != pt: continue
                d = abs(tb-pb)
                if d <= 1 and d < bd: best, bd = k, d
            if best >= 0: used[best] = True; matched += 1
    if np_pred+np_true == 0: return 1.0
    return 2.0*matched/(np_pred+np_true)

def row_score(P, Tt):
    tp = float((P*Tt).sum()); fp = float((P*(1-Tt)).sum()); fn = float(((1-P)*Tt).sum())
    ks = 1.0 if (P.sum() == 0 and Tt.sum() == 0) else (
        2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0.0)
    iou = []
    for b in range(NBIN):
        u = float(((P[b]+Tt[b]) > 0).sum()); i = float((P[b]*Tt[b]).sum())
        iou.append(1.0 if u == 0 else i/u)
    ex = 1.0 if (P == Tt).all() else 0.0
    return 0.42*ks + 0.28*np.mean(iou) + 0.20*transition_f1(P, Tt) + 0.10*ex

def axes(Tt):
    """The four robustness axes, computed exactly as the brief defines them."""
    path = [int(sum((1 << j) for j in range(4) if Tt[b, j])) for b in range(NBIN)]
    sing = [p for p in path if p in (1, 2, 4, 8)]
    ch = sum(1 for a, b in zip(sing, sing[1:]) if a != b)
    dens = 'calm' if ch <= 2 else ('exchange' if ch <= 5 else 'rapid')
    mult = 'clean' if (Tt.sum(1) <= 1).all() else 'overlap'
    ne = int((Tt.sum(1) > 0).sum())
    cov = ('moderate' if ne <= 17 else 'dense' if ne <= 21 else 'continuous') if ne >= 12 else 'sparse'
    return dens, mult, cov

def comp_score(P, Tt, cases):
    R = np.array([row_score(P[i], Tt[i]) for i in range(len(P))])
    A = [axes(Tt[i]) for i in range(len(Tt))]
    def worst(keys):
        g = {}
        for k, r in zip(keys, R): g.setdefault(k, []).append(r)
        q = [np.mean(v) for v in g.values() if len(v) >= 5]
        return min(q) if q else R.mean()
    w = [worst(list(cases)), worst([a[0] for a in A]),
         worst([a[1] for a in A]), worst([a[2] for a in A])]
    return 0.70*R.mean() + 0.075*sum(w), R.mean(), w

# ---------------- model ----------------
import torch, torch.nn as nn
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.manual_seed(SEED)

class Net(nn.Module):
    """Shared weights across the four keys, so the model learns "is this key present" rather
    than any identity. Coalition coupling enters as the mean hidden state over keys at each
    step -- permutation equivariant, and it lets a confident key suppress its rivals."""
    def __init__(s, d=NF, h=64):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        s.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        s.head = nn.Sequential(nn.Linear(4*h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(s, x):                       # x: (B,24,4,D)
        B = x.shape[0]
        z = s.inp(x).permute(0, 2, 1, 3).reshape(B*4, NBIN, -1)
        g, _ = s.gru(z)
        g = g.reshape(B, 4, NBIN, -1)
        ctx = g.mean(1, keepdim=True).expand_as(g)
        return s.head(torch.cat([g, ctx], -1)).squeeze(-1).permute(0, 2, 1)   # (B,24,4)

def viterbi(pr, persist):
    """Per-key 2-state decode. Speech is contiguous; a persistence bonus stops single-bin
    flicker that TransitionF1 punishes twice (a false entry AND a false exit)."""
    lp = np.log(np.clip(pr, 1e-6, 1-1e-6)); ln = np.log(np.clip(1-pr, 1e-6, 1-1e-6))
    out = np.zeros_like(pr, np.int8)
    for j in range(pr.shape[1]):
        dp = np.array([ln[0, j], lp[0, j]]); bk = np.zeros((NBIN, 2), int)
        for b in range(1, NBIN):
            for st in (0, 1):
                c = [dp[0]+(persist if st == 0 else 0), dp[1]+(persist if st == 1 else 0)]
                bk[b, st] = int(np.argmax(c))
                dp2 = max(c) + (lp[b, j] if st else ln[b, j])
                if st == 0: n0 = dp2
                else: n1 = dp2
            dp = np.array([n0, n1])
        st = int(np.argmax(dp))
        for b in range(NBIN-1, -1, -1):
            out[b, j] = st; st = bk[b, st]
    return out

def fit(X, Y, seed, epochs=EPOCHS, budget=None):
    torch.manual_seed(seed)
    m = Net()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs, pct_start=0.3)
    xt, yt = torch.tensor(X), torch.tensor(Y)
    n, bs = len(X), 32
    for ep in range(epochs):
        m.train()
        for k in range(0, n, bs):
            i = torch.randperm(n)[:bs] if k == 0 else torch.randint(0, n, (bs,))
            loss = nn.functional.binary_cross_entropy_with_logits(m(xt[i]), yt[i])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        sch.step()
        if budget and time.time() > budget: break
    m.eval(); return m

@torch.no_grad()
def prob(ms, X):
    return np.mean([torch.sigmoid(m(torch.tensor(X))).numpy() for m in ms], 0)

NF = 23
log('extracting features...')
Xtr = build(train); Ytr = decode_labels(train)
log('train features %s | positive rate %.3f' % (Xtr.shape, Ytr.mean()))

_A = ['sh', 'cosmarg', 'harm', 'idmarg', 'cosmax', 'shmarg', 'melmarg', 'melmax']
FN = _A + [a+'_z' for a in _A] + ['sh_max', 'cos_p90', 'harm_marg', 'id', 'vote',
                                  'rank', 'loud']
def auc(x, y):
    r = np.argsort(np.argsort(x)) + 1.0
    n1, n0 = y.sum(), (1-y).sum()
    return (r[y > 0].sum() - n1*(n1+1)/2)/(n1*n0) if n1 and n0 else 0.5
_y = Ytr.reshape(-1)
log('per-feature AUC vs presence: %s'
    % {FN[k]: round(float(auc(Xtr.reshape(-1, NF)[:, k], _y)), 3) for k in range(NF)})

# Reference points, so the model's number can be read as good or bad rather than in a vacuum.
# The metric gives partial credit generously (empty-vs-empty bins score 1), so a trivial
# prediction is NOT near zero and must be measured before anything is called an improvement.
CV = train.case_id.values
_b = {'all_empty': np.zeros_like(Ytr),
      'always_one_key': np.eye(4, dtype=np.float32)[Xtr[..., 1].argmax(-1)],
      'top_cue_thresholded': (zrow(Xtr[..., 1]) > 0.5).astype(np.float32)}
for k, v in _b.items():
    log('  baseline %-20s composite %.4f (mean R %.4f)'
        % (k, comp_score(v.astype(np.int8), Ytr, CV)[0], comp_score(v.astype(np.int8), Ytr, CV)[1]))

# ---------------- meeting-grouped validation ----------------
# Folds hold out WHOLE meetings, which is the actual test condition. A random row split would
# leak: 42 rows share each meeting's reference panel and its four voices.
cases = np.array(sorted(train.case_id.unique()))
folds = {c: i % 3 for i, c in enumerate(cases)}
fid = train.case_id.map(folds).values
OOF = np.zeros_like(Ytr)
tval = T0 + 0.45*TIME_LIMIT
for f in range(3):
    tr, va = np.where(fid != f)[0], np.where(fid == f)[0]
    ms = [fit(Xtr[tr], Ytr[tr], 10+f, budget=tval)]
    OOF[va] = prob(ms, Xtr[va])
    log('  fold %d: %d meetings held out' % (f, len(set(train.case_id.values[va]))))

def apply_decode(PR, th, ps):
    """Threshold, then optionally smooth. Re-centring on the threshold lets one calibration
    knob and one persistence knob be tuned together rather than fighting each other."""
    if not ps: return (PR > th).astype(np.int8)
    Q = np.clip(0.5 + (PR-th)/max(2*max(th, 1-th), 1e-6), 1e-4, 1-1e-4)
    return np.stack([viterbi(Q[i], ps) for i in range(len(PR))])

best = None
for th in np.arange(0.20, 0.71, 0.05):
    for ps in (0.0, 0.3, 0.6, 1.0, 1.5):
        s, mr, w = comp_score(apply_decode(OOF, th, ps), Ytr, train.case_id.values)
        if best is None or s > best[0]: best = (s, th, ps, mr, w)
S, TH, PS, MR, W = best
log('===== VALIDATION (meeting-grouped, send this back) =====')
log('COMPOSITE %.4f | mean R %.4f | worst case/trans/mult/cov %s' % (S, MR, np.round(W, 4)))
log('threshold %.2f  persistence %.1f' % (TH, PS))
log('===== END VALIDATION =====')

log('training submission models on all meetings...')
final = []
for sd in range(NSEED):
    if time.time()-T0 > TIME_LIMIT-180: break
    final.append(fit(Xtr, Ytr, 100+sd))
    log('  model %d done' % sd)
if not final: final = [fit(Xtr, Ytr, 100, epochs=30)]

Xte = build(test)
P = apply_decode(prob(final, Xte), TH, PS)

codes = ['H'+''.join('%X' % sum((1 << j) for j in range(4) if P[i, b, j])
                     for b in range(NBIN)) for i in range(len(P))]
sub = pd.DataFrame({'id': test.id.values, 'activity_code': codes})
import re
pat = re.compile(r'^H[0-9A-F]{24}$')
assert list(sub.columns) == ['id', 'activity_code']
assert len(sub) == len(test) and sub.id.nunique() == len(test)
assert sub.activity_code.str.match(pat).all()
assert not sub.activity_code.str.contains(' ').any()
sub.to_csv(OUT, index=False)
log('wrote %s %d rows | non-empty bin rate %.3f | multi-key %.3f'
    % (OUT, len(sub), (P.sum(-1) > 0).mean(), (P.sum(-1) > 1).mean()))
log('done')
