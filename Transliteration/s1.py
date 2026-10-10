"""Dakshina dual-view transliteration: cross-script gap filling with a learned lexicon + char model.

Structure recovered from train:
  gaps on the two sides are disjoint, so every gap has an INTACT counterpart token at the same
  cell -- the task is per-token transliteration in whichever direction is missing.
  99.7% of gaps are a single token, so skeleton tokens map 1:1 onto alignment cells.
  |A_pred| == |A_gold| is a HARD GATE: a wrong count zeroes the whole row. C = len(native
  tokens) satisfies it on 97.17% of rows, against 96.38% for the roman count and 95.10% for
  max(Tn,Tr). The residual failures are roman-side merges, e.g. Hindi "इसमें एक" -> "ismeanek",
  where two native tokens share one roman cell.

Scoring shape: an exact gap string earns 0.80+0.20*F1 while a near miss earns only
0.20*F1^2, and a full sentence earns 1.0 only if every gap in that view is exact. So exactness
is worth far more than closeness, which is why a lexicon of observed token pairs is the primary
predictor and the character model only handles what the lexicon has never seen.
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
             '/kaggle/input/datasets/devansh66/dakshina-dual-view-transliteration',
             '/kaggle/input/dakshina-dual-view-transliteration', '/kaggle/input', '.']
    pub = next((Path(c) for c in cands if (Path(c)/'train.csv').exists()),
               Path('./dataset/public'))
    out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)
NVAL = int(os.environ.get('NVAL', 4000))
NGRAM = int(os.environ.get('NGRAM', 3))
NEIGH = int(os.environ.get('NEIGH', 40))

train = pd.read_csv(PUB/'train.csv'); test = pd.read_csv(PUB/'test.csv')
log('train %s test %s' % (train.shape, test.shape))

def toks(s): return str(s).split()
def hid_of(t):
    return t[1:-1] if len(t) > 2 and t[0] == '<' and t[-1] == '>' else None

# hold out the last rows for an honest check; the lexicon must not see them
VA = np.arange(len(train)-NVAL, len(train))
FT = np.arange(0, len(train)-NVAL)

def build_lex(rows):
    n2r = defaultdict(Counter); r2n = defaultdict(Counter)
    for i in rows:
        L = train.language.values[i]
        for a in json.loads(train.target_json.values[i])['alignments']:
            nn, rr = a.get('native', ''), a.get('roman', '')
            if nn and rr:
                n2r[(L, nn)][rr] += 1
                r2n[(L, rr)][nn] += 1
    return ({k: c.most_common(1)[0][0] for k, c in n2r.items()},
            {k: c.most_common(1)[0][0] for k, c in r2n.items()})

def build_index(lex_keys):
    """Character-n-gram postings, so an unseen token can be matched to the nearest seen one."""
    idx = defaultdict(list)
    for (L, w) in lex_keys:
        s = '^'+w+'$'
        for j in range(max(len(s)-NGRAM+1, 1)):
            idx[(L, s[j:j+NGRAM])].append(w)
    return idx

def nearest(idx, lex, L, w):
    s = '^'+w+'$'
    cand = Counter()
    for j in range(max(len(s)-NGRAM+1, 1)):
        for c in idx.get((L, s[j:j+NGRAM]), ())[:200]: cand[c] += 1
    if not cand: return None
    best, bs = None, -1.0
    for c, ov in cand.most_common(NEIGH):
        sc = 2.0*ov/(len(s)+len(c)+2)
        if sc > bs: best, bs = c, sc
    return lex.get((L, best)) if best is not None else None

def predict_row(L, nskel, rskel, holes, N2R, R2N, IN, IR):
    nt, rt = toks(nskel), toks(rskel)
    C = len(nt)
    nf, rf = list(nt), list(rt)
    ng, rg = [], []
    for k, t in enumerate(nt):
        h = hid_of(t)
        if h and h.startswith('n'):
            src = rt[k] if k < len(rt) and not hid_of(rt[k]) else ''
            out = R2N.get((L, src)) if src else None
            if out is None and src: out = nearest(IR, R2N, L, src)
            if not out: out = src or t
            nf[k] = out; ng.append({'hole_id': h, 'text': out})
    for k, t in enumerate(rt):
        h = hid_of(t)
        if h and h.startswith('r'):
            src = nt[k] if k < len(nt) and not hid_of(nt[k]) else ''
            out = N2R.get((L, src)) if src else None
            if out is None and src: out = nearest(IN, N2R, L, src)
            if not out: out = src or t
            rf[k] = out; rg.append({'hole_id': h, 'text': out})
    want = [h for h in holes]
    ng = [g for g in ng if g['hole_id'] in want]
    rg = [g for g in rg if g['hole_id'] in want]
    have = {g['hole_id'] for g in ng} | {g['hole_id'] for g in rg}
    for h in want:                                   # never drop a required hole: the gate
        if h not in have:                            # checks key sets, not just contents
            (ng if h.startswith('n') else rg).append({'hole_id': h, 'text': ''})
    # alignment cells follow the native token count; pad or trim the roman side to match
    al = []
    for k in range(C):
        al.append({'native': nf[k] if k < len(nf) else '',
                   'roman': rf[k] if k < len(rf) else ''})
    return {'alignments': al, 'native_gaps': ng, 'native_text': ' '.join(nf),
            'roman_gaps': rg, 'romanized_text': ' '.join(rf)}

def f1char(p, g):
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    cp, cg = Counter(p), Counter(g)
    inter = sum((cp & cg).values())
    if inter == 0: return 0.0
    pr, rc = inter/len(p), inter/len(g)
    return 2*pr*rc/(pr+rc)

def score_row(pred, gold):
    pn = {g['hole_id'] for g in pred['native_gaps']}
    gn = {g['hole_id'] for g in gold['native_gaps']}
    pr_ = {g['hole_id'] for g in pred['roman_gaps']}
    gr = {g['hole_id'] for g in gold['roman_gaps']}
    if pn != gn or pr_ != gr or len(pred['alignments']) != len(gold['alignments']):
        return 0.0
    def gapscore(P, G):
        gd = {g['hole_id']: g['text'] for g in G}
        pd_ = {g['hole_id']: g['text'] for g in P}
        if not gd: return 1.0
        s = 0.0
        for h, gt in gd.items():
            pt = pd_.get(h, '')
            f = f1char(pt, gt)
            s += (0.80+0.20*f) if pt == gt else 0.20*f*f
        return s/len(gd)
    sn = gapscore(pred['native_gaps'], gold['native_gaps'])
    sr = gapscore(pred['roman_gaps'], gold['roman_gaps'])
    fn = 1.0 if pred['native_text'] == gold['native_text'] else 0.10*f1char(pred['native_text'], gold['native_text'])
    fr = 1.0 if pred['romanized_text'] == gold['romanized_text'] else 0.10*f1char(pred['romanized_text'], gold['romanized_text'])
    ap = Counter((a.get('native', ''), a.get('roman', '')) for a in pred['alignments'])
    ag = Counter((a.get('native', ''), a.get('roman', '')) for a in gold['alignments'])
    inter = sum((ap & ag).values())
    if inter == 0: fa = 0.0
    else:
        p_, r_ = inter/sum(ap.values()), inter/sum(ag.values())
        fa = 2*p_*r_/(p_+r_)
    return 0.25*sn+0.25*sr+0.20*fn+0.20*fr+0.10*fa

log('building lexicon from %d rows...' % len(FT))
N2R, R2N = build_lex(FT)
IN = build_index(N2R.keys()); IR = build_index(R2N.keys())
log('lexicon: native->roman %d | roman->native %d entries' % (len(N2R), len(R2N)))

log('validating on %d held-out rows...' % len(VA))
sc = []; gate = 0
for i in VA:
    row = train.iloc[i]
    gold = json.loads(row.target_json)
    p = predict_row(row.language, row.native_skeleton, row.roman_skeleton,
                    json.loads(row.hole_ids_json), N2R, R2N, IN, IR)
    s = score_row(p, gold); sc.append(s)
    gate += (len(p['alignments']) == len(gold['alignments']))
log('===== VALIDATION (held-out tail) =====')
log('MEAN SCORE %.4f | gate pass %.4f | zero rows %.4f'
    % (np.mean(sc), gate/len(VA), np.mean([s == 0 for s in sc])))
log('===== END VALIDATION =====')

log('rebuilding lexicon on ALL train rows...')
N2R, R2N = build_lex(np.arange(len(train)))
IN = build_index(N2R.keys()); IR = build_index(R2N.keys())
log('lexicon: native->roman %d | roman->native %d entries' % (len(N2R), len(R2N)))

preds = []
for i, row in enumerate(test.itertuples()):
    p = predict_row(row.language, row.native_skeleton, row.roman_skeleton,
                    json.loads(row.hole_ids_json), N2R, R2N, IN, IR)
    preds.append(json.dumps(p, ensure_ascii=False))
    if (i+1) % 4000 == 0: log('  test %d/%d' % (i+1, len(test)))

sub = pd.DataFrame({'id': test.id.values, 'prediction': preds})
assert list(sub.columns) == ['id', 'prediction'] and len(sub) == len(test)
assert sub.id.tolist() == test.id.tolist()
for s in sub.prediction:
    o = json.loads(s)
    assert set(o) == {'alignments', 'native_gaps', 'native_text', 'roman_gaps', 'romanized_text'}
    assert isinstance(o['alignments'], list) and o['alignments']
    for a in o['alignments']: assert set(a) == {'native', 'roman'}
    for g in o['native_gaps']+o['roman_gaps']: assert set(g) == {'hole_id', 'text'}
sub.to_csv(OUT, index=False, encoding='utf-8')
log('wrote %s %d rows | mean cells %.2f'
    % (OUT, len(sub), np.mean([len(json.loads(s)['alignments']) for s in sub.prediction])))
log('done')
