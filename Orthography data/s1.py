"""Drinov orthography safe repair: character-level contextual tagger, conservatively decoded.

The metric is 0.50*repair_gain + 0.30*safe_edit_precision + 0.20*heritage_glyph_f1, where
safe_edit_precision = progress/introduced_edits. Copying the OCR scores ZERO on all three
terms -- not just the first two -- because the OCR contains no heritage glyphs at all
(0 in ocr vs 8601 in the targets), so a copy has no glyphs to align and F1 is 0.

Measured shape of the task, from aligning the training pairs:
  keep 90.1%, substitute 7.4%, delete 1.6%, insert 0.9%
  so a per-character classifier over OUTPUT STRINGS covers every operation: '' is a deletion,
  a single char a substitution, two chars a substitution plus insertion.
  Only three glyphs ever occur: ѣ 6190, ѫ 1991, ѭ 420.
  The substitutions are patterned but context-dependent -- е->ѣ 526 times while 'е' usually
  stays 'е' -- which is why this has to be contextual rather than a confusion map.

Because safe_edit_precision divides by the number of edits made, an uncertain edit costs more
than it gains. The tagger therefore only applies a change when its probability clears a
threshold tuned on held-out documents against the real metric.
"""
import os, sys, time, math, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

T0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time()-T0), *a, flush=True)

def parse_paths():
    if len(sys.argv) > 2 and not sys.argv[1].startswith('-f'):
        return Path(sys.argv[1]), Path(sys.argv[2])
    cands = ['./dataset/public',
             '/kaggle/input/datasets/devansh66/drinov-orthography',
             '/kaggle/input/drinov-orthography', '/kaggle/input', '.']
    pub = next((Path(c) for c in cands if (Path(c)/'train.csv').exists()),
               Path('./dataset/public'))
    out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)
NFOLD = int(os.environ.get('NFOLD', 5))
EPOCHS = int(os.environ.get('EPOCHS', 30))
NCLS = int(os.environ.get('NCLS', 160))
EMB = int(os.environ.get('EMB', 64))
HID = int(os.environ.get('HID', 192))
NSEED = int(os.environ.get('NSEED', 2))
NSWEEP = int(os.environ.get('NSWEEP', 400))
TIME_LIMIT = float(os.environ.get('TIME_LIMIT', 9000))
GLYPHS = 'ѣѫѧѭѩѯѱѳѵ'

train = pd.read_csv(PUB/'train.csv'); test = pd.read_csv(PUB/'test.csv')
train['ocr_text'] = train.ocr_text.astype(str)
train['repaired_text'] = train.repaired_text.astype(str)
test['ocr_text'] = test.ocr_text.astype(str)
log('train %s test %s' % (train.shape, test.shape))

# ---------------- metric ----------------
AR = np.arange(4096)

def _row(prev, ca, bb, idx):
    """One Levenshtein row, vectorised.

    The (cur[j-1]+1) dependency is a min-plus scan, which np.minimum.accumulate solves after
    subtracting the position index -- so a row costs a handful of numpy ops instead of a Python
    loop. That matters because the threshold sweep needs thousands of these.
    """
    cur = np.minimum(prev[1:]+1, prev[:-1]+(bb != ca))
    cur = np.concatenate(([prev[0]+1], cur))
    return np.minimum.accumulate(cur-idx)+idx

def lev(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    bb = np.frombuffer(b.encode('utf-32-le'), np.uint32)
    aa = np.frombuffer(a.encode('utf-32-le'), np.uint32)
    idx = AR[:len(b)+1] if len(b)+1 <= len(AR) else np.arange(len(b)+1)
    prev = idx.astype(np.int32).copy()
    for i in range(len(aa)):
        prev = _row(prev, aa[i], bb, idx)
        prev[0] = i+1
        prev = np.minimum.accumulate(prev-idx)+idx
    return int(prev[-1])

def align_ops(a, b):
    """Monotonic edit path; insertions attach to the preceding source character."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return ([('d', i, '') for i in range(m)] if n == 0
                else [('i', 0, c) for c in b])
    aa = np.frombuffer(a.encode('utf-32-le'), np.uint32)
    bb = np.frombuffer(b.encode('utf-32-le'), np.uint32)
    idx = np.arange(n+1)
    D = np.zeros((m+1, n+1), np.int32)
    D[0] = idx
    for i in range(m):
        r = _row(D[i], aa[i], bb, idx)
        r[0] = i+1
        D[i+1] = np.minimum.accumulate(r-idx)+idx
    i, j, out = m, n, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i-1, j-1]+(aa[i-1] != bb[j-1]):
            out.append(('m', i-1, b[j-1])); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i-1, j]+1:
            out.append(('d', i-1, '')); i -= 1
        else:
            out.append(('i', i, b[j-1])); j -= 1
    return out[::-1]

def targets(a, b):
    """Output string emitted by each source character."""
    seg = ['']*len(a)
    pend = ''
    for t, i, c in align_ops(a, b):
        if t == 'i':
            pend += c
        elif t == 'd':
            seg[i] = pend; pend = ''
        else:
            seg[i] = pend+c; pend = ''
    if pend and len(a): seg[-1] = seg[-1]+pend
    return seg

def glyph_f1(pred, truth):
    tp = fp = fn = 0
    ops = align_ops(pred, truth)
    pm = set(); tm = set()
    for t, i, c in ops:
        if t == 'm' and i < len(pred) and pred[i] == c and c in GLYPHS:
            tp += 1; pm.add(i)
    fp = sum(1 for k, ch in enumerate(pred) if ch in GLYPHS and k not in pm)
    fn = sum(1 for ch in truth if ch in GLYPHS) - tp
    if tp+fp+fn == 0: return 1.0
    return 2.0*tp/(2.0*tp+fp+fn)

def score_rows(ocr, pred, truth):
    rg, sp, gf = [], [], []
    for o, p, t in zip(ocr, pred, truth):
        ie = lev(o, t); re_ = lev(p, t); ed = lev(o, p)
        prog = ie-re_
        rg.append(min(max(prog/max(ie, 1), 0.0), 1.0))
        sp.append(min(max(prog/max(ed, 1), 0.0), 1.0))
        gf.append(glyph_f1(p, t))
    return (0.50*np.mean(rg)+0.30*np.mean(sp)+0.20*np.mean(gf),
            np.mean(rg), np.mean(sp), np.mean(gf))

# ---------------- labels ----------------
log('aligning training pairs...')
SEGS = [targets(o, r) for o, r in zip(train.ocr_text, train.repaired_text)]
cnt = Counter()
for o, sg in zip(train.ocr_text, SEGS):
    for ch, s in zip(o, sg):
        if s != ch: cnt[s] += 1
keep_first = ['\x00']                          # class 0 = emit the source character unchanged
cls = keep_first + [s for s, _ in cnt.most_common(NCLS)]
CID = {s: i for i, s in enumerate(cls)}
covered = sum(c for s, c in cnt.items() if s in CID)
log('label classes %d | edit coverage %.4f | total edits %d'
    % (len(cls), covered/max(sum(cnt.values()), 1), sum(cnt.values())))

chars = Counter()
for t in list(train.ocr_text)+list(test.ocr_text): chars.update(t)
VOC = {c: i+1 for i, (c, _) in enumerate(chars.most_common())}
log('char vocab %d' % len(VOC))

def encode(t): return np.array([VOC.get(c, 0) for c in t], np.int64)
def labels(o, sg):
    y = np.zeros(len(o), np.int64)
    for k, (ch, s) in enumerate(zip(o, sg)):
        y[k] = 0 if s == ch else CID.get(s, 0)
    return y

X = [encode(t) for t in train.ocr_text]
Yl = [labels(o, sg) for o, sg in zip(train.ocr_text, SEGS)]
XT = [encode(t) for t in test.ocr_text]

import torch, torch.nn as nn
torch.set_num_threads(max(1, os.cpu_count() or 1))

class Net(nn.Module):
    """Dilated residual CNN over characters: a ~31-character receptive field, which is ample for
    orthographic context, and far cheaper on CPU than a recurrent stack."""
    def __init__(s, nv, nc, emb=EMB, h=HID):
        super().__init__()
        s.e = nn.Embedding(nv+1, emb)
        s.inp = nn.Conv1d(emb, h, 3, padding=1)
        s.bl = nn.ModuleList([nn.Conv1d(h, h, 3, padding=d, dilation=d)
                              for d in (1, 2, 4, 8, 16)])
        s.nm = nn.ModuleList([nn.GroupNorm(8, h) for _ in range(5)])
        s.out = nn.Conv1d(h, nc, 1)
        s.do = nn.Dropout(0.1)
    def forward(s, x):
        z = torch.nn.functional.gelu(s.inp(s.e(x).transpose(1, 2)))
        for c, n in zip(s.bl, s.nm):
            z = z + s.do(torch.nn.functional.gelu(n(c(z))))
        return s.out(z).transpose(1, 2)

def pad_batch(idx, Xs, Ys=None):
    L = max(len(Xs[i]) for i in idx)
    xb = np.zeros((len(idx), L), np.int64)
    yb = np.full((len(idx), L), -100, np.int64)
    for r, i in enumerate(idx):
        xb[r, :len(Xs[i])] = Xs[i]
        if Ys is not None: yb[r, :len(Ys[i])] = Ys[i]
    return torch.from_numpy(xb), torch.from_numpy(yb)

def train_model(idx, seed, w, budget):
    torch.manual_seed(seed)
    m = Net(len(VOC), len(cls))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=max(EPOCHS, 1),
                                              pct_start=0.3)
    wt = torch.tensor(w, dtype=torch.float32)
    for ep in range(EPOCHS):
        m.train()
        order = np.random.permutation(idx)
        for k in range(0, len(order), 16):
            b = order[k:k+16]
            if len(b) < 2: continue
            xb, yb = pad_batch(b, X, Yl)
            lo = m(xb)
            loss = nn.functional.cross_entropy(lo.reshape(-1, len(cls)), yb.reshape(-1),
                                               weight=wt, ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        sch.step()
        if time.time() > budget:
            log('    stopped at epoch %d (time)' % ep); break
    m.eval(); return m

@torch.no_grad()
def probs(models, Xs):
    out = []
    for i in range(0, len(Xs), 16):
        b = list(range(i, min(i+16, len(Xs))))
        xb, _ = pad_batch(b, Xs)
        p = None
        for m in models:
            q = torch.softmax(m(xb), -1)
            p = q if p is None else p+q
        p = (p/len(models)).numpy()
        for r, j in enumerate(b): out.append(p[r, :len(Xs[j])])
    return out

def apply_pred(text, P, th):
    """Emit the model's class only when it is confident; otherwise keep the source character."""
    o = []
    for k, ch in enumerate(text):
        row = P[k]
        j = int(np.argmax(row[1:]))+1 if len(row) > 1 else 0
        if row[j] >= th and cls[j] != '\x00':
            o.append(cls[j])
        else:
            o.append(ch)
    return ''.join(o)

# class weights: KEEP is 90% of positions, so it is down-weighted, but only mildly -- the
# metric punishes over-editing, so the model should stay biased toward keeping.
freq = np.zeros(len(cls))
for y in Yl:
    for v in y: freq[v] += 1
W = (freq.sum()/np.maximum(freq, 1.0))**0.35
W = W/W.mean()

# contiguous-block folds: consecutive chunks continue the same document, so blocks keep
# documents together where a random split would not
fold = (np.arange(len(train))*NFOLD)//len(train)
OOFP = [None]*len(train)
tval = T0 + 0.55*TIME_LIMIT
for f in range(NFOLD):
    tr_i = np.where(fold != f)[0]; va_i = np.where(fold == f)[0]
    m = train_model(tr_i, 100+f, W, min(tval, T0+TIME_LIMIT))
    for j, p in zip(va_i, probs([m], [X[j] for j in va_i])): OOFP[j] = p
    log('  fold %d/%d done' % (f+1, NFOLD))

sw = np.random.RandomState(SEED).permutation(len(train))[:NSWEEP]
so = [train.ocr_text[i] for i in sw]; st = [train.repaired_text[i] for i in sw]
best = None
for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    preds = [apply_pred(train.ocr_text[i], OOFP[i], th) for i in sw]
    s = score_rows(so, preds, st)
    log('  th %.2f -> score %.4f (gain %.4f, safe %.4f, glyph %.4f)' % ((th,)+s))
    if best is None or s[0] > best[0][0]: best = (s, th)
(S, RG, SP, GF), TH = best
cp = score_rows(so, so, st)
log('===== VALIDATION (contiguous-block OOF) =====')
log('copy-OCR baseline score %.4f (gain %.4f safe %.4f glyph %.4f)' % cp)
log('BEST score %.4f at threshold %.2f | gain %.4f safe %.4f glyph %.4f'
    % (S, TH, RG, SP, GF))
log('===== END VALIDATION =====')

log('training final models on all chunks...')
models = []
for sd in range(NSEED):
    if time.time()-T0 > TIME_LIMIT-300: break
    models.append(train_model(np.arange(len(train)), 7+sd, W, T0+TIME_LIMIT-120))
    log('  model %d done' % sd)
if not models: models = [train_model(np.arange(len(train)), 7, W, T0+TIME_LIMIT-60)]

PT = probs(models, XT)
rep = [apply_pred(o, PT[i], TH) for i, o in enumerate(test.ocr_text)]
rep = [r if r.strip() else o for r, o in zip(rep, test.ocr_text)]
sub = pd.DataFrame({'id': test.id.values, 'repaired_text': rep})
assert list(sub.columns) == ['id', 'repaired_text'] and len(sub) == len(test)
assert sub.id.tolist() == test.id.tolist()
assert sub.repaired_text.str.len().min() > 0
sub.to_csv(OUT, index=False, encoding='utf-8')
ed = [lev(o, r) for o, r in zip(test.ocr_text, rep)]
ng = sum(sum(r.count(g) for g in GLYPHS) for r in rep)
log('wrote %s %d rows | mean edits/chunk %.1f | glyphs restored %d' % (OUT, len(sub), np.mean(ed), ng))
log('done')
