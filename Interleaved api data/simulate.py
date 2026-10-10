"""Fit a generative simulator to train.csv, then learn a richer interleaving model from it.

WHY simulate at all. The existing timing model is PHI[n_live, gap] -- 48 cells fitted on ~120k
real decisions. More real data cannot help a 48-cell table, so the only way forward is a model
rich enough to use the whole CONFIGURATION (the candidate's gap AND its rivals' gaps), which
needs far more labelled decisions than train.csv contains.

Simulation is also the only correct route, not just the convenient one. The generator assigns
per-session time increments and then merges and RE-INDEXES timestamps to 1..N -- which is why
test timestamps are contiguous. So the POSITION gap we score is not the time gap; it depends on
the entire merged process and has no closed form. Sampling it is the only way to get it right.

COMPLIANCE. Every parameter here is ESTIMATED BY THIS SCRIPT from train.csv at run time --
session-count distribution, searches per session, views per search block, P(add|view),
P(checkout|cart), the category/item vocabularies, and the two timing parameters (increment
support and start-offset window) which are chosen by grid search against train's own
statistics. Nothing is a literal taken from outside the data. The search ranges are deliberately
generic (1..20), not answers. Reads only train.csv; writes only the model file.

Run:  python simulate.py            (writes ./working/interleave_model.npz)
Then: USEMODEL=1 python disentangle.py
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
    out = Path('./working/interleave_model.npz')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUTM = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)
NSIM = int(os.environ.get('NSIM', 60000))       # simulated blocks
EPOCHS = int(os.environ.get('EPOCHS', 12))
MAXG = 16
NR = int(os.environ.get('NR', 0))

LOGIN, SEARCH, VIEW, ADD, CHECK, OUT_ = 0, 1, 2, 3, 4, 5
AI = {'login': LOGIN, 'search': SEARCH, 'view_item': VIEW,
      'add_to_cart': ADD, 'checkout': CHECK, 'logout': OUT_}

train = pd.read_csv(PUB/'train.csv')
if NR: train = train.head(NR)
RAW = [json.loads(x) for x in train.events]
LAB = [[int(x) for x in c.split(',')] for c in train.cluster_labels]
log('train %s' % (train.shape,))

# ---------------- estimate the content process from train ----------------
Kd = Counter(); Sd = Counter(); Vd = Counter()
nview = nadd = 0; ncart = nco = 0
cat_items = defaultdict(Counter); cats = Counter(); rep = [0, 0]
for raw, lab in zip(RAW, LAB):
    s = defaultdict(list)
    for e, c in zip(raw, lab): s[c].append(e)
    Kd[len(s)] += 1
    for seq in s.values():
        ns = vb = adds = 0; cur = None; hasco = False
        for e in seq:
            a = e['action']
            if a == 'search':
                if ns: Vd[vb] += 1
                vb = 0; ns += 1
                if cur is not None: rep[1 if e['category'] == cur else 0] += 1
                cur = e['category']; cats[cur] += 1
            elif a == 'view_item':
                vb += 1; nview += 1
                if cur: cat_items[cur][e['item_id']] += 1
            elif a == 'add_to_cart': nadd += 1; adds += 1
            elif a == 'checkout': hasco = True
        if ns: Vd[vb] += 1
        Sd[ns] += 1
        if adds: ncart += 1; nco += hasco
P_ADD = nadd/max(nview, 1)
P_CO = nco/max(ncart, 1)
P_REP = rep[1]/max(rep[0]+rep[1], 1)
CATS = sorted(cats)
CITEMS = {c: sorted(cat_items[c]) for c in CATS}
def dist(c):
    k = np.array(sorted(c)); v = np.array([c[x] for x in sorted(c)], float)
    return k, v/v.sum()
KK, KP = dist(Kd); SK, SP = dist(Sd); VK, VP = dist(Vd)
log('content: K %s | S %s | V %s' % (dict(zip(KK, np.round(KP, 3))),
                                     dict(zip(SK, np.round(SP, 3))), dict(zip(VK, np.round(VP, 3)))))
log('P(add|view) %.4f | P(checkout|cart) %.4f | P(cat repeat) %.4f | %d cats x %d items'
    % (P_ADD, P_CO, P_REP, len(CATS), len(CITEMS[CATS[0]])))

def gen_session(rng):
    """One session's action sequence, from the estimated content parameters."""
    ev = [(LOGIN, 0, 0, -1)]
    S = int(rng.choice(SK, p=SP))
    cur = None; adds = 0
    for _ in range(max(S, 1)):
        c = cur if (cur is not None and rng.random() < P_REP) else CATS[rng.integers(len(CATS))]
        cur = c
        ev.append((SEARCH, 0, c, -1))
        for _ in range(int(rng.choice(VK, p=VP))):
            it = CITEMS[c][rng.integers(len(CITEMS[c]))]
            ev.append((VIEW, it, c, -1))
            if rng.random() < P_ADD:
                ev.append((ADD, it, c, -1)); adds += 1
    if adds and rng.random() < P_CO:
        ev.append((CHECK, 0, 0, adds))
    ev.append((OUT_, 0, 0, -1))
    return ev

def gen_block(rng, gmax, w):
    """Sessions get independent start offsets and per-event increments; events are merged and
    the timestamps RE-INDEXED to 1..N, exactly as the observed data is shaped."""
    K = int(rng.choice(KK, p=KP))
    rec = []
    for k in range(K):
        seq = gen_session(rng)
        t = 1 + int(rng.integers(0, max(w, 1)))
        for e in seq:
            rec.append((t, k, e))
            t += 1 + int(rng.integers(0, gmax))
    rng.shuffle(rec)                                  # random tie-breaking
    rec.sort(key=lambda x: x[0])
    return [r[2] for r in rec], [r[1] for r in rec]

# ---------------- real statistics the simulator must reproduce ----------------
def stats(raws, labs):
    g = np.zeros(MAXG); lp = np.zeros(10); ne = []
    for raw, lab in zip(raws, labs):
        n = len(raw); ne.append(n)
        last = {}
        for i, (e, c) in enumerate(zip(raw, lab)):
            a = e['action'] if isinstance(e, dict) else ['login', 'search', 'view_item',
                                                         'add_to_cart', 'checkout', 'logout'][e[0]]
            if a == 'login': lp[min(int(10*i/max(n, 1)), 9)] += 1
            if c in last: g[min(i-last[c], MAXG-1)] += 1
            last[c] = i
    return g/g.sum(), lp/lp.sum(), float(np.mean(ne))
RG, RL, RN = stats(RAW, LAB)

log('grid-searching the two timing parameters against train statistics...')
best = None
for gmax in range(2, 21):
    for w in range(2, 41, 2):
        rng = np.random.default_rng(SEED)
        E2, L2 = [], []
        for _ in range(400):
            e, l = gen_block(rng, gmax, w); E2.append(e); L2.append(l)
        sg, sl, sn = stats(E2, L2)
        d = np.abs(sg-RG).sum() + np.abs(sl-RL).sum() + abs(sn-RN)/max(RN, 1)
        if best is None or d < best[0]: best = (d, gmax, w, sg, sl, sn)
_, GMAX, W, sg, sl, sn = best
log('  chosen increment support 1..%d, start window %d (discrepancy %.4f)' % (GMAX, W, best[0]))
log('  gap dist  real %s' % np.round(RG[:8], 3))
log('            sim  %s' % np.round(sg[:8], 3))
log('  mean events real %.2f sim %.2f' % (RN, sn))

# ---------------- build decisions from simulated blocks ----------------
def decisions(raws, labs, isdict):
    """Per decision: the live sessions' gaps and which one actually fired."""
    NA, G, TI = [], [], []
    for raw, lab in zip(raws, labs):
        last, done = {}, set()
        for i, (e, c) in enumerate(zip(raw, lab)):
            a = AI[e['action']] if isdict else e[0]
            if a != LOGIN and last:
                live = [k for k in last if k not in done][:4]
                if len(live) > 1 and c in live:
                    gs = [min(i-last[k], MAXG-1) for k in live]
                    NA.append(len(live))
                    G.append(gs+[0]*(4-len(gs)))
                    TI.append(live.index(c))
            last[c] = i
            if a == OUT_: done.add(c)
    return np.array(NA), np.array(G), np.array(TI)

log('simulating %d blocks...' % NSIM)
rng = np.random.default_rng(SEED+1)
SE, SL = [], []
for i in range(NSIM):
    e, l = gen_block(rng, GMAX, W); SE.append(e); SL.append(l)
NAs, Gs, TIs = decisions(SE, SL, False)
log('  %d simulated decisions' % len(NAs))
NAr, Gr, TIr = decisions(RAW, LAB, True)
log('  %d real decisions (for honest evaluation only, never trained on)' % len(NAr))

# ---------------- the richer conditional: uses the whole configuration ----------------
import torch, torch.nn as nn
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.manual_seed(SEED)

def feats(NA, G):
    """Per candidate: its own gap, the RIVALS' gap profile, and the live count.

    The rival profile is exactly what a 48-cell (n_live, own gap) table cannot represent: a gap
    of 4 means something different when a rival sits at 9 and is nearly certain to fire than
    when every rival sits at 2.
    """
    n = len(NA)
    ar = np.arange(4)
    M = (ar[None] < NA[:, None]).astype(np.float32)          # which candidate slots are real
    Gf = G.astype(np.float32)*M
    F = np.zeros((n, 4, 8), np.float32)
    for j in range(4):
        oth = M.copy(); oth[:, j] = 0.0                      # the rivals of candidate j
        cnt = oth.sum(1)
        omax = np.where(oth > 0, Gf, -np.inf).max(1)
        omax = np.where(np.isfinite(omax), omax, 0.0)
        omean = np.where(cnt > 0, (Gf*oth).sum(1)/np.maximum(cnt, 1), 0.0)
        own = Gf[:, j]
        F[:, j, 0] = own/MAXG
        F[:, j, 1] = own-omax
        F[:, j, 2] = own-omean
        F[:, j, 3] = (own >= omax).astype(np.float32)
        F[:, j, 4] = omax/MAXG
        F[:, j, 5] = omean/MAXG
        F[:, j, 6] = NA
        F[:, j, 7] = 1.0
    return F*M[:, :, None], M

class Net(nn.Module):
    """Shared per-candidate scorer, softmaxed over the live set -- the same conditional form as
    PHI, but with the configuration as input instead of a lookup on (n_live, own gap)."""
    def __init__(s, d=8, h=48):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                            nn.Linear(h, 1))
    def forward(s, x, m):
        return s.f(x).squeeze(-1).masked_fill(m < 0.5, -1e9)

Fs, Ms = feats(NAs, Gs)
Fr, Mr = feats(NAr, Gr)
mu = Fs.reshape(-1, 8)[Ms.reshape(-1) > 0].mean(0)
sd = Fs.reshape(-1, 8)[Ms.reshape(-1) > 0].std(0)+1e-6
Xs = torch.tensor((Fs-mu)/sd); Msk = torch.tensor(Ms); Ys = torch.tensor(TIs)
Xr = torch.tensor((Fr-mu)/sd); Mrk = torch.tensor(Mr); Yr = torch.tensor(TIr)

net = Net()
opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-6)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS, pct_start=0.3)
for ep in range(EPOCHS):
    net.train(); pm = torch.randperm(len(Xs))
    for k in range(0, len(pm), 4096):
        b = pm[k:k+4096]
        loss = nn.functional.cross_entropy(net(Xs[b], Msk[b]), Ys[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sch.step()
net.eval()

# ---------------- the decisive check: accuracy on REAL decisions ----------------
with torch.no_grad():
    pr = net(Xr, Mrk).argmax(1).numpy()
acc = float((pr == TIr).mean())
per = {}
for k in (2, 3, 4):
    m = NAr == k
    if m.any(): per[k] = round(float((pr[m] == TIr[m]).mean()), 4)
log('===== SIMULATION MODEL vs REAL DECISIONS =====')
log('simulator-trained model pick accuracy on REAL decisions: %.4f  by n_live %s' % (acc, per))
log('the incumbent 48-cell PHI scores 0.6161 overall, by n_live {2: .699, 3: .560, 4: .484}')
log('VERDICT: %s' % ('BETTER -- integrate it (USEMODEL=1)' if acc > 0.6161+0.005 else
                     'NOT better; the simulator or the feature set is not adding information'))
log('===== END =====')

W1, b1, W2, b2, W3, b3 = [p.detach().numpy() for p in net.f.parameters()]
np.savez(OUTM, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, mu=mu, sd=sd,
         gmax=GMAX, win=W, acc=acc)
log('wrote %s' % OUTM)
