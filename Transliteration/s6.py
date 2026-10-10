"""Dakshina dual-view transliteration: cross-script gap filling with lexicon + JST graphone + analogy transducer ensemble.

Structure & Metric:
  gaps are disjoint across native and roman views, so every gap has an intact counterpart token.
  |A_pred| == |A_gold| is a HARD GATE (zeroes row if violated).
  Exact match per gap earns 0.80+0.20*F1; non-exact earns only 0.20*F1^2. Exact match is the single key driver.

Pipeline:
  1. Lexicon Lookup (most_common(1) from train alignments): serves ~64-70% of test gaps instantly.
  2. For OOV gaps: Confidence-gated Ensemble of Joint-Sequence Character Transducer (JST graphone n-gram model with EM alignment)
     and Analogy Suffix Transducer (mining suffix rewrite rules from near lexicon neighbours).
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys, json, time, math, random, bisect
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

T0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

def parse_paths():
    if len(sys.argv) > 2 and not sys.argv[1].startswith('-f'):
        return Path(sys.argv[1]), Path(sys.argv[2])
    cands = ['./dataset/public',
             '/kaggle/input/datasets/devansh66/dakshina-dual-view-transliteration',
             '/kaggle/input/dakshina-dual-view-transliteration', '/kaggle/input',
             'C:/Users/Devansh Rajput/Downloads/Dakshina Dual-View Transliteration', '.']
    pub = next((Path(c) for c in cands if (Path(c)/'train.csv').exists()),
               Path('./dataset/public'))
    out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED)
NVAL = int(os.environ.get('NVAL', 4000))

train = pd.read_csv(PUB/'train.csv'); test = pd.read_csv(PUB/'test.csv')
log('train %s test %s' % (train.shape, test.shape))

def toks(s): return str(s).split()
def hid_of(t): return t[1:-1] if len(t) > 2 and t[0] == '<' and t[-1] == '>' else None

VA = np.arange(len(train) - NVAL, len(train))
FT = np.arange(0, len(train) - NVAL)

def build_common(rows):
    cn, cr = defaultdict(Counter), defaultdict(Counter)
    for i in rows:
        L = train.language.values[i]
        for a in json.loads(train.target_json.values[i])['alignments']:
            if a.get('native'): cn[L][a['native']] += 1
            if a.get('roman'): cr[L][a['roman']] += 1
    return ({L: c.most_common(1)[0][0] for L, c in cn.items()},
            {L: c.most_common(1)[0][0] for L, c in cr.items()})

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
            {k: c.most_common(1)[0][0] for k, c in r2n.items()},
            n2r, r2n)

W, PMIN, TOPR, NC, KNN = 8, 2, 5, 10, 20

def mk(lex):
    fwd, rev = defaultdict(list), defaultdict(list)
    for (L, w) in lex: fwd[L].append(w); rev[L].append(w[::-1])
    for L in fwd: fwd[L].sort(); rev[L].sort()
    return fwd, rev

def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i

def prefix_cands(arr, w, k=KNN):
    j = bisect.bisect_left(arr, w)
    return arr[max(0, j-k):j+k]

def edist(a, b):
    lb = len(b); prev = list(range(lb+1))
    for i in range(1, len(a)+1):
        cur = [i]+[0]*lb; ai = a[i-1]
        for j in range(1, lb+1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ai != b[j-1]))
        prev = cur
    return prev[lb]

def mine_rules(lex, fwd):
    R = defaultdict(Counter)
    for L, arr in fwd.items():
        for i in range(len(arr)):
            v1 = arr[i]; t1 = lex[(L, v1)]
            for j in range(i+1, min(i+W, len(arr))):
                v2 = arr[j]; t2 = lex[(L, v2)]
                p, q = lcp(v1, v2), lcp(t1, t2)
                if p < PMIN or q < 2: continue
                a, b, c, d = v1[p:], v2[p:], t1[q:], t2[q:]
                if len(a) > 7 or len(b) > 7 or len(c) > 9 or len(d) > 9: continue
                R[(L, a, b)][(c, d)] += 1
                R[(L, b, a)][(d, c)] += 1
    # Memory optimization: Prune rules with count < 2
    R_pruned = defaultdict(Counter)
    for k, C in R.items():
        for pair, count in C.items():
            if count >= 2:
                R_pruned[k][pair] = count
    return R_pruned

def make_analogy(lex, fwd, rev, RUL, tag, CACHE={}):
    def f(L, s):
        key = (tag, L, s)
        if key in CACHE: return CACHE[key]
        arr = fwd.get(L, []); r = None; max_conf = 0.0
        if arr:
            cands = set(prefix_cands(arr, s))
            for c in prefix_cands(rev.get(L, []), s[::-1]): cands.add(c[::-1])
            scored = sorted(cands, key=lambda c: (edist(s, c), -lcp(s, c)))[:NC]
            V = Counter()
            for rank, v in enumerate(scored):
                p = lcp(s, v)
                if p < PMIN: continue
                a, b = v[p:], s[p:]; t = lex.get((L, v))
                if t is None: continue
                if a == b:
                    V[t] += 6.0/(1+rank); continue
                cc = RUL.get((L, a, b))
                if not cc: continue
                tot = sum(cc.values())
                for (cf, ct), n in cc.most_common(TOPR):
                    if t.endswith(cf) and (len(t)-len(cf)) >= 1:
                        weight = (n/tot)*(3.0/(1+rank))
                        V[t[:len(t)-len(cf)]+ct] += weight
                        max_conf = max(max_conf, weight)
                        break
            if V: r = V.most_common(1)[0][0]
            elif scored: r = lex.get((L, scored[0]))
        CACHE[key] = (r, max_conf)
        return r, max_conf
    return f

# JST (Joint-Sequence Transducer)
MPL = 8000; EMIT = 4; BEAM = 12; NCAND = 8

def select_pairs(cnt_lex, top_lex):
    per = defaultdict(list)
    for (L, w), C in cnt_lex.items():
        per[L].append((sum(C.values()), w))
    pairs = []
    langs = []
    for L, v in per.items():
        v.sort(reverse=True)
        for _, w in v[:MPL]:
            pairs.append((w, top_lex[(L, w)]))
            langs.append(L)
    return pairs, langs

def align_sym(s, t, LP, max_src=3, max_tgt=3, first=False):
    Ls, Lt = len(s), len(t)
    if Ls == 0 or Lt == 0: return None
    NEG = -1e17
    dp = [NEG] * (Lt + 1)
    dp[0] = 0.0
    bp = [[(0, 0)] * (Lt + 1) for _ in range(Ls + 1)]
    ratio = Lt / Ls
    
    for i in range(Ls):
        for j in range(Lt + 1):
            v = dp[j]
            if v < NEG / 2: continue
            for ls in range(1, min(max_src, Ls - i) + 1):
                s_chunk = s[i:i+ls]
                for lt in range(0, min(max_tgt, Lt - j) + 1):
                    t_chunk = t[j:j+lt]
                    k = j + lt
                    if first:
                        sc = v - abs(lt - ls * ratio)
                    else:
                        prob = LP.get(s_chunk, {}).get(t_chunk, -11.0)
                        sc = v + prob
                    if sc > dp[k]:
                        dp[k] = sc
                        bp[i+ls][k] = (i, j)
                        
    if dp[Lt] < NEG / 2: return None
    out = []
    i, j = Ls, Lt
    while (i, j) != (0, 0):
        pi, pj = bp[i][j]
        if (pi, pj) == (i, j): return None
        out.append((s[pi:i], t[pj:j]))
        i, j = pi, pj
    out.reverse()
    return out

def train_align(pairs, max_src=3, max_tgt=3, iters=EMIT):
    LP = None
    for it in range(iters):
        cnt = defaultdict(Counter)
        nfail = 0
        for (s, t) in pairs:
            a = align_sym(s, t, LP, max_src, max_tgt, first=(it == 0))
            if a is None:
                nfail += 1
                continue
            for sc, tc in a:
                cnt[sc][tc] += 1
        LP = {}
        for sc, C in cnt.items():
            tot = sum(C.values()) + len(C) * 0.1
            LP[sc] = {k: math.log((v + 0.1) / tot) for k, v in C.items()}
    return LP

def build_graphone_model(pairs, langs, LP, max_src=3, max_tgt=3):
    C1 = defaultdict(Counter)
    C2 = defaultdict(Counter)
    C3 = defaultdict(Counter)
    for (s, t), L in zip(pairs, langs):
        a = align_sym(s, t, LP, max_src, max_tgt)
        if a is None: continue
        pc = '<s>'
        pos = 0
        for i, (sc, tc) in enumerate(a):
            p = s[pos-1] if pos > 0 else '^'
            nx = s[pos+len(sc)] if pos+len(sc) < len(s) else '$'
            k2 = (L, p, sc, nx)
            C1[(L, sc)][tc] += 1
            C2[k2][tc] += 1
            C3[k2 + (pc,)][tc] += 1
            pc = tc
            pos += len(sc)
    return dict(C1), dict(C2), dict(C3)

def decode_graphone(s, L, C1, C2, C3, max_src=3, CACHE={}):
    key = (L, s, id(C1))
    if key in CACHE: return CACHE[key]
    n = len(s)
    if n == 0: return ''
    st = defaultdict(list)
    st[0] = [(0.0, '<s>', '')]
    
    for i in range(n):
        if not st[i]: continue
        cur = sorted(st[i], key=lambda x: -x[0])[:BEAM]
        st[i] = cur
        
        for l in range(1, min(max_src, n - i) + 1):
            k = i + l
            sc = s[i:k]
            p = s[i-1] if i > 0 else '^'
            nx = s[k] if k < n else '$'
            k2 = (L, p, sc, nx)
            
            cd = C2.get(k2) or C1.get((L, sc))
            if cd is None:
                if l > 1: continue
                cd = Counter({sc: 1})
                
            tot = sum(cd.values())
            cl = cd.most_common(NCAND)
            buck = st[k]
            
            for (score, pc, out) in cur:
                c3 = C3.get(k2 + (pc,))
                t3 = sum(c3.values()) if c3 else 0
                for tc, count in cl:
                    p2 = count / tot
                    pr = 0.35 * p2 + (0.65 * c3.get(tc, 0) / t3 if c3 else 0.65 * p2)
                    buck.append((score + math.log(pr + 1e-9), tc, out + tc))
                    
    fin = st.get(n)
    res = sorted(fin, key=lambda x: -x[0])[0][2] if fin else ''
    CACHE[key] = res
    return res

def cell_align(nt, rt, L, N2R, R2N):
    n, m = len(nt), len(rt)
    NEG = -1e9
    D = [[NEG]*(m+1) for _ in range(n+1)]
    B = [[None]*(m+1) for _ in range(n+1)]
    D[0][0] = 0.0

    def cell_score(ns, rs):
        gn = any(hid_of(t) for t in ns); gr = any(hid_of(t) for t in rs)
        if gn and gr: return NEG
        if gn or gr: return 0.0
        a, b = ' '.join(ns), ' '.join(rs)
        pa, pb = R2N.get((L, b)), N2R.get((L, a))
        if pa == a or pb == b: return 1.0
        if pa is not None or pb is not None: return -0.25
        return 0.0

    for i in range(n+1):
        for j in range(m+1):
            if D[i][j] == NEG: continue
            for di, dj, pen in ((1, 1, 0.0), (2, 1, -0.4), (1, 2, -0.4)):
                if i+di > n or j+dj > m: continue
                cs = cell_score(nt[i:i+di], rt[j:j+dj])
                if cs == NEG: continue
                v = D[i][j] + cs + pen
                if v > D[i+di][j+dj]:
                    D[i+di][j+dj] = v; B[i+di][j+dj] = (i, j)
    if D[n][m] == NEG: return None
    cells, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj = B[i][j]
        cells.append((list(range(pi, i)), list(range(pj, j))))
        i, j = pi, pj
    return cells[::-1]

def predict_row(L, nskel, rskel, holes, N2R, R2N, AN_n, AN_r, JST_n, JST_r):
    nt, rt = toks(nskel), toks(rskel)
    cells = cell_align(nt, rt, L, N2R, R2N) if (nt and rt and len(nt) != len(rt)) else None
    if cells is None:
        cells = [([k], [k]) for k in range(min(len(nt), len(rt)))]
    if not cells:
        cells = [(list(range(len(nt))), list(range(len(rt))))]
    if not (4 <= len(cells) <= 32):
        alt = [([k], [k]) for k in range(min(len(nt), len(rt)))]
        if alt and 4 <= len(alt) <= 32: cells = alt
    C = len(cells)
    nf, rf = list(nt), list(rt)
    ng, rg = [], []
    
    C1_n, C2_n, C3_n = JST_n
    C1_r, C2_r, C3_r = JST_r

    for ni, ri in cells:
        nspan = [nt[k] for k in ni]; rspan = [rt[k] for k in ri]
        for k, t in zip(ni, nspan):
            h = hid_of(t)
            if h and h.startswith('n'):
                src = ' '.join(x for x in rspan if not hid_of(x))
                out = R2N.get((L, src)) if src else None
                if out is None and src:
                    p_an, conf_an = AN_n(L, src)
                    p_jst = decode_graphone(src, L, C1_n, C2_n, C3_n, max_src=4)
                    if p_jst and p_an:
                        out = p_jst if p_jst == p_an or conf_an < 1.5 else p_an
                    else:
                        out = p_jst or p_an
                if not out: out = src or t
                nf[k] = out; ng.append({'hole_id': h, 'text': out})
        for k, t in zip(ri, rspan):
            h = hid_of(t)
            if h and h.startswith('r'):
                src = ' '.join(x for x in nspan if not hid_of(x))
                out = N2R.get((L, src)) if src else None
                if out is None and src:
                    p_an, conf_an = AN_r(L, src)
                    p_jst = decode_graphone(src, L, C1_r, C2_r, C3_r, max_src=1)
                    if p_jst and p_an:
                        out = p_jst if p_jst == p_an or conf_an < 1.5 else p_an
                    else:
                        out = p_jst or p_an
                if not out: out = src or t
                rf[k] = out; rg.append({'hole_id': h, 'text': out})
    want = [h for h in holes]
    ng = [g for g in ng if g['hole_id'] in want]
    rg = [g for g in rg if g['hole_id'] in want]
    have = {g['hole_id'] for g in ng} | {g['hole_id'] for g in rg}
    for h in want:
        if h not in have:
            (ng if h.startswith('n') else rg).append({'hole_id': h, 'text': ''})
    al = []
    for ni, ri in cells:
        al.append({'native': ' '.join(nf[k] for k in ni),
                   'roman': ' '.join(rf[k] for k in ri)})
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
N2R, R2N, n2r, r2n = build_lex(FT)
COMMON_N, COMMON_R = build_common(FT)
log('lexicon: native->roman %d | roman->native %d entries' % (len(N2R), len(R2N)))
FN, RN = mk(N2R); FR, RR = mk(R2N)
RUL_N2R = mine_rules(N2R, FN); RUL_R2N = mine_rules(R2N, FR)
AN_n = make_analogy(R2N, FR, RR, RUL_R2N, 'n')
AN_r = make_analogy(N2R, FN, RN, RUL_N2R, 'r')
log('mined rules: n2r %d | r2n %d' % (len(RUL_N2R), len(RUL_R2N)))

log('training Joint-Sequence Transducers (JST)...')
pairs_r, langs_r = select_pairs(n2r, N2R)
LP_r = train_align(pairs_r, max_src=1, max_tgt=4, iters=EMIT)
JST_r = build_graphone_model(pairs_r, langs_r, LP_r, max_src=1, max_tgt=4)

pairs_n, langs_n = select_pairs(r2n, R2N)
LP_n = train_align(pairs_n, max_src=4, max_tgt=2, iters=EMIT)
JST_n = build_graphone_model(pairs_n, langs_n, LP_n, max_src=4, max_tgt=2)

log('validating on %d held-out rows...' % len(VA))
sc = []; gate = 0
for i in VA:
    row = train.iloc[i]
    gold = json.loads(row.target_json)
    p = predict_row(row.language, row.native_skeleton, row.roman_skeleton,
                    json.loads(row.hole_ids_json), N2R, R2N, AN_n, AN_r, JST_n, JST_r)
    s = score_row(p, gold); sc.append(s)
    gate += (len(p['alignments']) == len(gold['alignments']))
log('===== VALIDATION (held-out tail) =====')
log('MEAN SCORE %.4f | gate pass %.4f | zero rows %.4f'
    % (np.mean(sc), gate/len(VA), np.mean([s == 0 for s in sc])))
log('===== END VALIDATION =====')

log('rebuilding lexicon and JST models on ALL train rows...')
N2R, R2N, n2r, r2n = build_lex(np.arange(len(train)))
COMMON_N, COMMON_R = build_common(np.arange(len(train)))
log('lexicon: native->roman %d | roman->native %d entries' % (len(N2R), len(R2N)))
FN, RN = mk(N2R); FR, RR = mk(R2N)
RUL_N2R = mine_rules(N2R, FN); RUL_R2N = mine_rules(R2N, FR)
AN_n = make_analogy(R2N, FR, RR, RUL_R2N, 'n2')
AN_r = make_analogy(N2R, FN, RN, RUL_N2R, 'r2')
log('mined rules: n2r %d | r2n %d' % (len(RUL_N2R), len(RUL_R2N)))

pairs_r, langs_r = select_pairs(n2r, N2R)
LP_r = train_align(pairs_r, max_src=1, max_tgt=4, iters=EMIT)
JST_r = build_graphone_model(pairs_r, langs_r, LP_r, max_src=1, max_tgt=4)

pairs_n, langs_n = select_pairs(r2n, R2N)
LP_n = train_align(pairs_n, max_src=4, max_tgt=2, iters=EMIT)
JST_n = build_graphone_model(pairs_n, langs_n, LP_n, max_src=4, max_tgt=2)

preds = []
for i, row in enumerate(test.itertuples()):
    p = predict_row(row.language, row.native_skeleton, row.roman_skeleton,
                    json.loads(row.hole_ids_json), N2R, R2N, AN_n, AN_r, JST_n, JST_r)
    preds.append(json.dumps(p, ensure_ascii=False))
    if (i+1) % 4000 == 0: log('  test %d/%d' % (i+1, len(test)))

sub = pd.DataFrame({'id': test.id.values, 'prediction': preds})
assert list(sub.columns) == ['id', 'prediction'] and len(sub) == len(test)
assert sub.id.tolist() == test.id.tolist()
for s in sub.prediction:
    o = json.loads(s)
    assert isinstance(o.get('native_text'), str) and isinstance(o.get('romanized_text'), str)
    assert set(o) == {'alignments', 'native_gaps', 'native_text', 'roman_gaps', 'romanized_text'}
    assert isinstance(o['alignments'], list) and o['alignments']
    for a in o['alignments']: assert set(a) == {'native', 'roman'}
    for g in o['native_gaps']+o['roman_gaps']: assert set(g) == {'hole_id', 'text'}
sub.to_csv(OUT, index=False, encoding='utf-8')
log('wrote %s %d rows | mean cells %.2f'
    % (OUT, len(sub), np.mean([len(json.loads(s)['alignments']) for s in sub.prediction])))
log('done')
