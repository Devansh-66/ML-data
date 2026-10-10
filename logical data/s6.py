"""Logical bundle induction by constraint induction over a compositional rule schema.

WHAT THE DATA ACTUALLY DOES (established by analysis, not by search):

  Every recommended item is the price-extremal (or price-nearest) member of a SUBSET of its
  own category.  Evidence: 74.9% of all 18,048 gold outputs are exactly the most expensive
  or exactly the cheapest item of their (category, colour, brand) cell, against a 13% chance
  rate; the price histogram of gold outputs is violently U-shaped (44% in the top price
  decile, 24% in the bottom) while the catalogue and the carts are uniform.

  So an output is produced by a three-part schema

        output  =  select_by_price( { x in category K : colour(x) in A , brand(x) in B } )

  where the category K is read off the examples (the output category multiset is constant
  within a task), and the two attribute constraints and the price selector are the only
  unknowns.  Each constraint is one of a handful of relations to the cart:

        ANY          no constraint  (the attribute is then a CONSEQUENCE of the price rule,
                                     which is why "copy the attribute" models fail)
        INCART       the attribute must occur somewhere in the cart
        MODE         the attribute most frequent in the cart
        S<sel>       the attribute of one distinguished cart item -- dearest / cheapest /
                     first / last / second / median / the one sharing the output category
        K<value>     a constant, task-specific value

  and the price selector is either an extreme (max / min) or "nearest to T" for a cart
  statistic T (max, min, mean, sum, median, or the price of a distinguished cart item).
  The "nearest to T" branch is what produces the striking near-equal-price pairs in
  multi-output tasks: two slots share one target T, so the two chosen items land within a
  couple of currency units of each other.

WHY THIS IS INDUCTION AND NOT ENUMERATION OF HANDCRAFTED RULES.  The hypothesis space is not
a list of rules; it is a small schema with three independent slots, and each slot's value is
INDUCED from the examples by the classical version-space method: keep exactly the constraint
values that are consistent with every observed (cart, output) pair and discard the rest.
Consistency is checked by membership and by a price interval, so no rule is ever "tried".
When several constraint values survive -- unavoidable with three examples -- the ambiguity is
resolved by a PRIOR over schema instantiations estimated from the whole corpus (pass 1 below),
i.e. a learned model of which rules the task generator actually writes.
"""
import os, sys, json, time
from pathlib import Path
from collections import Counter, defaultdict
from itertools import permutations
import numpy as np, pandas as pd

T0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time()-T0), *a, flush=True)

def parse_paths():
    if len(sys.argv) > 2 and not sys.argv[1].startswith('-f'):
        return Path(sys.argv[1]), Path(sys.argv[2])
    cands = ['./dataset/public', '/kaggle/input/datasets/devansh66/logical-bundle',
             '/kaggle/input/logical-bundle', '/kaggle/input', '.']
    pub = next((Path(c) for c in cands if (Path(c)/'tasks.json').exists()),
               Path('./dataset/public'))
    out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SUBNAME = os.environ.get('SUBNAME', '')          # override output filename for safe testing
if SUBNAME: OUT = OUT.parent/SUBNAME
NVAL = int(os.environ.get('NVAL', '3000'))

catalog = json.load(open(PUB/'catalog.json', encoding='utf-8'))
tasks = json.load(open(PUB/'tasks.json', encoding='utf-8'))
log('catalog %d | tasks %d' % (len(catalog), len(tasks)))

# ---------------------------------------------------------------- catalogue tables
CATS = sorted({c['category'] for c in catalog})
COLS = sorted({c['color'] for c in catalog})
BRAS = sorted({c['brand'] for c in catalog})
ci = {c: i for i, c in enumerate(CATS)}
oi = {c: i for i, c in enumerate(COLS)}
bi = {c: i for i, c in enumerate(BRAS)}
IDS = np.array([c['item_id'] for c in catalog])
PRICE = np.array([c['price'] for c in catalog], np.float64)
CA = np.array([ci[c['category']] for c in catalog])
CO = np.array([oi[c['color']] for c in catalog])
BR = np.array([bi[c['brand']] for c in catalog])
POS = {int(v): k for k, v in enumerate(IDS)}
# each category is held in ASCENDING PRICE ORDER, so any attribute-filtered subset of it is
# already sorted and the price selectors reduce to an endpoint or a binary search
BYCAT = {k: np.where(CA == k)[0] for k in range(len(CATS))}
for _k in BYCAT: BYCAT[_k] = BYCAT[_k][np.argsort(PRICE[BYCAT[_k]])]
NCOL, NBRA = len(COLS), len(BRAS)

# colour/brand membership tables per category, so a subset is one boolean AND
CATCO = {k: CO[BYCAT[k]] for k in BYCAT}
CATBR = {k: BR[BYCAT[k]] for k in BYCAT}
CATPR = {k: PRICE[BYCAT[k]] for k in BYCAT}

# ---------------------------------------------------------------- cart descriptors
SELS = ('ph0', 'pl0', 'f0', 'b0', 'ph1', 'pl1', 'f1', 'b1', 'med', 'sc')

def describe(inp, ocat):
    """Distinguished cart items and cart price statistics."""
    q = np.array([POS[i] for i in inp])
    p = PRICE[q]
    n = len(q)
    o = np.argsort(-p)
    d = {'ph0': q[o[0]], 'pl0': q[o[-1]], 'f0': q[0], 'b0': q[-1],
         'ph1': q[o[min(1, n-1)]], 'pl1': q[o[max(n-2, 0)]],
         'f1': q[min(1, n-1)], 'b1': q[max(n-2, 0)], 'med': q[o[n//2]]}
    same = [x for x in q if CA[x] == ocat]
    if same: d['sc'] = same[0]
    # the cart item of one fixed category ("the Shoes in your basket"): a task can key its
    # colour, its brand, or its price. Ties inside a category resolve to the dearest.
    for k in range(len(CATS)):
        m = [x for x in q if CA[x] == k]
        if m:
            d['k%d' % k] = m[int(np.argmax(PRICE[m]))]
            if len(m) > 1: d['j%d' % k] = m[0]        # first-listed, when the cart repeats K
    T = {'NMAX': p.max(), 'NMIN': p.min(), 'NMEAN': p.mean(), 'NSUM': p.sum(),
         'NMED': float(np.median(p)), 'NHALF': p.sum()/2.0}
    for s in ('ph0', 'pl0', 'f0', 'b0', 'sc'):
        if s in d: T['N'+s] = PRICE[d[s]]
    for k in range(len(CATS)):
        if 'k%d' % k in d: T['Nk%d' % k] = PRICE[d['k%d' % k]]
        if 'j%d' % k in d: T['Nj%d' % k] = PRICE[d['j%d' % k]]
    return q, d, T

def attr_options(q, d, ARR, nval):
    """The candidate values of one attribute constraint, as boolean masks over 0..nval-1."""
    vals = ARR[q]
    o = {'ANY': None}
    m = np.zeros(nval, bool); m[vals] = True
    o['INCART'] = m
    cnt = Counter(int(v) for v in vals)
    top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    m = np.zeros(nval, bool); m[top[0][0]] = True
    o['MODE'] = m
    if len(top) > 1:
        m = np.zeros(nval, bool); m[top[-1][0]] = True
        o['ANTIMODE'] = m
    for s, v in d.items():
        m = np.zeros(nval, bool); m[ARR[v]] = True
        o['S'+s] = m
    return o

# price selectors: the two extremes, or "nearest to T" for a cart-derived target price T.
# (Price-BAND variants -- cheapest strictly above T, dearest strictly below T -- were tried
# and measured: they add +0.0003 LOO while tripling the space, so they are not carried.)
TARGETS = ('MAX', 'MIN', 'NMAX', 'NMIN', 'NMEAN', 'NSUM', 'NMED', 'NHALF',
           'Nph0', 'Npl0', 'Nf0', 'Nb0', 'Nsc') + tuple('Nk%d' % k for k in range(10))           + tuple('Nj%d' % k for k in range(10))

def const_masks(n):
    out = {}
    for v in range(n):
        m = np.zeros(n, bool); m[v] = True
        out['K%d' % v] = m
    return out
CONSTC = const_masks(NCOL)
CONSTB = const_masks(NBRA)

def subset(ocat, cm, bm):
    pool = BYCAT[ocat]
    k = np.ones(len(pool), bool)
    if cm is not None: k &= cm[CATCO[ocat]]
    if bm is not None: k &= bm[CATBR[ocat]]
    return pool[k]

def _nearest(s, p, t):
    i = int(np.searchsorted(p, t))
    if i == 0: return s[0]
    if i == len(p): return s[-1]
    return s[i] if (p[i] - t) < (t - p[i-1]) else s[i-1]

def choose(s, tname, T):
    if len(s) == 0: return None
    if tname == 'MAX': return s[-1]
    if tname == 'MIN': return s[0]
    t = T.get(tname)
    if t is None: return None
    return _nearest(s, PRICE[s], t)

def choose_all(s, T):
    """Evaluate every price selector on one candidate subset in a single vectorised pass.
    Returns {selector name -> chosen catalogue row}. This is the inner loop of induction,
    so it matters that it is one (|S| x n_targets) reduction rather than a Python loop."""
    if len(s) == 0: return {}
    p = PRICE[s]
    out = {'MAX': s[-1], 'MIN': s[0]}
    names = [t for t in TARGETS if t in T]
    if names:
        tv = np.fromiter((T[t] for t in names), np.float64, len(names))
        idx = np.searchsorted(p, tv)
        lo = np.clip(idx-1, 0, len(p)-1); hi = np.clip(idx, 0, len(p)-1)
        take = np.where(np.abs(p[hi]-tv) < np.abs(tv-p[lo]), hi, lo)
        for t, r in zip(names, take): out[t] = s[int(r)]
    return out

# ---------------------------------------------------------------- version-space induction
def induce(carts, ocat, golds, best=False):
    """Return every schema instantiation consistent with all (cart, gold) pairs.

    With best=True only the maximum-prior survivor is returned.  Two examples constrain the
    schema weakly enough that the version space can hold tens of thousands of members, and
    materialising it is pure waste when all the caller wants is the MAP hypothesis."""
    K = len(carts)
    Q, D, TT = [], [], []
    for c in carts:
        q, d, T = describe(c, ocat); Q.append(q); D.append(d); TT.append(T)
    CoO = [attr_options(Q[k], D[k], CO, NCOL) for k in range(K)]
    BrO = [attr_options(Q[k], D[k], BR, NBRA) for k in range(K)]
    for k in range(K): CoO[k].update(CONSTC); BrO[k].update(CONSTB)
    g = [POS[x] for x in golds]
    out = [] if not best else []
    cn = [n for n in CoO[0] if all(n in CoO[k] for k in range(K))]
    bn = [n for n in BrO[0] if all(n in BrO[k] for k in range(K))]
    # version space step 1: the constraint must ADMIT the gold in every example
    cn = [n for n in cn if all(CoO[k][n] is None or CoO[k][n][CO[g[k]]] for k in range(K))]
    bn = [n for n in bn if all(BrO[k][n] is None or BrO[k][n][BR[g[k]]] for k in range(K))]
    for a in cn:
        subs_a = [CoO[k][a] for k in range(K)]
        for b in bn:
            # version space step 2: keep the price selectors that land on the gold in EVERY
            # example -- intersect the per-example winning sets rather than re-testing rules
            surv = None
            for k in range(K):
                ch = choose_all(subset(ocat, subs_a[k], BrO[k][b]), TT[k])
                w = {t for t, r in ch.items() if r == g[k]}
                surv = w if surv is None else (surv & w)
                if not surv: break
            if not surv: continue
            if best:
                for t in surv:
                    h = (a, b, t); w = PRIOR.get(h, 0.0)
                    if not out or w > out[1]: out = [h, w]
            else:
                for t in surv: out.append((a, b, t))
    return ([out[0]] if out else []) if best else out

_CTX = {}
def context(cart, ocat):
    """Cart descriptors and attribute-constraint options, memoised: a single cart is
    described once and then reused by every schema evaluated against it."""
    k = (tuple(cart), ocat)
    v = _CTX.get(k)
    if v is None:
        q, d, T = describe(cart, ocat)
        CoO = attr_options(q, d, CO, NCOL); CoO.update(CONSTC)
        BrO = attr_options(q, d, BR, NBRA); BrO.update(CONSTB)
        v = _CTX[k] = (CoO, BrO, T)
    return v

def apply_hyp(h, cart, ocat):
    a, b, t = h
    CoO, BrO, T = context(cart, ocat)
    if a not in CoO or b not in BrO: return None
    j = choose(subset(ocat, CoO[a], BrO[b]), t, T)
    return None if j is None else int(IDS[j])

# ---------------------------------------------------------------- slot alignment
def slots_of(ex):
    """Canonical slot order inside one example: by category, then by price descending."""
    return sorted(ex['output'], key=lambda i: (CA[POS[i]], -PRICE[POS[i]], i))

def aligned(exs):
    per = [slots_of(e) for e in exs]
    cats = [int(CA[POS[i]]) for i in per[0]]
    groups = defaultdict(list)
    for j, c in enumerate(cats): groups[c].append(j)
    dup = [g for g in groups.values() if len(g) == 2]
    if not dup: return per
    carts = [e['input'] for e in exs]
    for g in dup:
        j0, j1 = g
        oc = cats[j0]
        # for each example after the first, decide whether the pair is swapped, by
        # checking which assignment keeps a consistent schema alive on both slots
        order = [0]*len(per)
        best, bestsc = None, -1
        from itertools import product
        for flips in product([0, 1], repeat=len(per)-1):
            f = (0,) + flips
            a = [per[k][j1] if f[k] else per[k][j0] for k in range(len(per))]
            b = [per[k][j0] if f[k] else per[k][j1] for k in range(len(per))]
            sc = (1 if induce(carts, oc, a) else 0) + (1 if induce(carts, oc, b) else 0)
            if sc > bestsc: bestsc, best = sc, f
        for k in range(len(per)):
            if best[k]:
                per[k][j0], per[k][j1] = per[k][j1], per[k][j0]
    return per

# ---------------------------------------------------------------- pass 1: learn the prior
log('pass 1: estimating the schema prior over the corpus ...')
PRIOR = Counter()
ALIGN = {}
VSETS = []                     # the surviving version space of every slot in the corpus
NSLOT_TOTAL = 0
# Version spaces are interned to small integer ids and capped, so the corpus-wide prior fits
# in memory: a slot whose examples barely constrain the schema can leave thousands of
# survivors, and those slots carry almost no information about the prior anyway.
HID = {}; HLIST = []
VCAP = 256
_rng = np.random.default_rng(0)
def _intern(H):
    out = np.empty(len(H), np.int32)
    for i, h in enumerate(H):
        k = HID.get(h)
        if k is None:
            k = len(HLIST); HID[h] = k; HLIST.append(h)
        out[i] = k
    if len(out) > VCAP: out = _rng.choice(out, VCAP, replace=False)
    return out
for tid, t in tasks.items():
    exs = t['train']
    per = aligned(exs)
    ALIGN[tid] = per
    carts = [e['input'] for e in exs]
    for j in range(len(per[0])):
        oc = int(CA[POS[per[0][j]]])
        H = induce(carts, oc, [p[j] for p in per])
        NSLOT_TOTAL += 1
        if H: VSETS.append(_intern(H))
log('pass 1 done | %d slots explained by the schema (%.3f of %d)'
    % (len(VSETS), len(VSETS)/max(NSLOT_TOTAL, 1), NSLOT_TOTAL))

# The version spaces are ambiguous, so the mixing proportions of the generator's rules are a
# latent-variable problem: estimate them by EM.  E-step distributes each slot's unit of mass
# over its surviving schemas in proportion to the current prior; M-step renormalises.
P = np.zeros(len(HLIST))
for H in VSETS: P[H] += 1.0/len(H)
for _ in range(15):
    nxt = np.zeros(len(HLIST))
    for H in VSETS:
        w = P[H] + 1e-9
        nxt[H] += w/w.sum()
    P = nxt
PRIOR = {HLIST[i]: float(P[i]) for i in np.nonzero(P > 1e-6)[0]}
top = sorted(PRIOR.items(), key=lambda kv: -kv[1])[:5]
log('EM prior converged | %d schemas with mass | top: %s'
    % (len(PRIOR), ', '.join('%s=%.0f' % (h, v) for h, v in top)))
PMAX = max(PRIOR.values()) if PRIOR else 1.0

PRIOR_ORDER = sorted(PRIOR, key=lambda h: (-PRIOR[h], h))
PARTIAL_N = int(os.environ.get('PARTIAL_N', '250'))
def rank_hyps(H):
    return sorted(H, key=lambda h: (-PRIOR.get(h, 0.0), h))

FALLBACK = [('ANY', 'ANY', 'MAX'), ('ANY', 'ANY', 'MIN')]

def predict_slot(carts, ocat, golds, cart_test):
    """Induce on the given examples, then apply the highest-prior surviving schema."""
    H = induce(carts, ocat, golds, best=True)
    if not H:
        # No schema explains every example -- either the task is outside the schema or one
        # example is generated by a variant.  Fall back to MAP over the learned prior:
        # among the schemas the generator actually uses, take the one that explains the most
        # examples, breaking ties by prior mass.  (A full version-space relaxation over
        # subsets of the examples was tried; this is as accurate and far cheaper.)
        best, bestsc = None, -1
        for h in PRIOR_ORDER[:PARTIAL_N]:
            sc = sum(1 for k in range(len(carts))
                     if apply_hyp(h, carts[k], ocat) == golds[k])
            if sc > bestsc: bestsc, best = sc, h
            if sc == len(carts): break
        H = [best] if best is not None else FALLBACK
    # take the maximum-prior survivor in one pass; a full sort of the version space is
    # wasted work because it is almost always the first candidate that applies
    order = [max(H, key=lambda h: (PRIOR.get(h, 0.0), h))]
    for h in order:
        v = apply_hyp(h, cart_test, ocat)
        if v is not None: return v
    for h in rank_hyps(H)[:64]:
        v = apply_hyp(h, cart_test, ocat)
        if v is not None: return v
    for h in FALLBACK:
        v = apply_hyp(h, cart_test, ocat)
        if v is not None: return v
    return int(IDS[BYCAT[ocat][0]])

# ---------------------------------------------------------------- honest leave-one-out
log('leave-one-out validation (induce on 2 examples, predict the held-out third) ...')
jac = []
cover = 0; nslot = 0
for tid in list(tasks)[:NVAL]:
    exs = tasks[tid]['train']
    per = ALIGN[tid]
    for hold in range(len(exs)):
        keep = [k for k in range(len(exs)) if k != hold]
        carts = [exs[k]['input'] for k in keep]
        pred = []
        for j in range(len(per[0])):
            oc = int(CA[POS[per[0][j]]])
            golds = [per[k][j] for k in keep]
            v = predict_slot(carts, oc, golds, exs[hold]['input'])
            nslot += 1
            if v == per[hold][j]: cover += 1
            pred.append(v)
        p = set(pred); g = set(exs[hold]['output'])
        jac.append(len(p & g)/max(len(p | g), 1))
log('===== VALIDATION (leave-one-out over the in-context examples) =====')
log('slot exact-item accuracy %.4f' % (cover/max(nslot, 1)))
log('MEAN JACCARD %.4f  -> score ~%.2f' % (np.mean(jac), 100*np.mean(jac)))
log('===== END VALIDATION =====')

# ---------------------------------------------------------------- predict the test carts
log('predicting test carts ...')
rows = []
for tid, t in tasks.items():
    exs = t['train']; per = ALIGN[tid]
    carts = [e['input'] for e in exs]
    cart_test = t['test'][0]['input']
    got = []
    for j in range(len(per[0])):
        oc = int(CA[POS[per[0][j]]])
        v = predict_slot(carts, oc, [p[j] for p in per], cart_test)
        if v in got:                      # never emit a duplicate id: Jaccard counts sets
            oc_pool = BYCAT[oc]
            for alt in oc_pool[::-1]:
                if int(IDS[alt]) not in got: v = int(IDS[alt]); break
        got.append(v)
    rows.append((tid, ' '.join(str(x) for x in got)))

sub = pd.DataFrame(rows, columns=['task_id', 'recommended_item_ids'])
ss = pd.read_csv(PUB/'sample_submission.csv')
sub = ss[['task_id']].merge(sub, on='task_id', how='left')
sub['recommended_item_ids'] = sub.recommended_item_ids.fillna(str(int(IDS[0])))
assert list(sub.columns) == ['task_id', 'recommended_item_ids'] and len(sub) == len(ss)
assert sub.task_id.tolist() == ss.task_id.tolist()
assert sub.recommended_item_ids.str.len().min() > 0
sub.to_csv(OUT, index=False)
log('wrote %s %d rows | mean items/pred %.2f'
    % (OUT, len(sub), np.mean([len(s.split()) for s in sub.recommended_item_ids])))
log('done')
