"""Signal hunt for the disentanglement task. Standalone: reads train.csv only, writes nothing.

Each section tests ONE hypothesis about the generator that the current solver does not use,
and prints a verdict. Run:  python signals.py [nrows]
"""
import os, sys, json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

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
# a bare int argument sets the row count; a notebook's "-f kernel.json" must not be read as one
NR = next((int(a) for a in sys.argv[1:] if a.isdigit()), 2000)
NR = int(os.environ.get('NR', NR))
print('reading %s | %d rows' % (PUB, NR))
tr = pd.read_csv(PUB/'train.csv').head(NR)
E = [json.loads(x) for x in tr.events]
L = [[int(x) for x in c.split(',')] for c in tr.cluster_labels]
CAT = {1: 'electronics', 2: 'books', 3: 'clothing', 4: 'home', 5: 'sports'}
print('blocks: %d\n' % len(E))
def hdr(s): print('\n=== %s ===' % s)

# ---------------------------------------------------------------- A: category collisions
# If the generator gives concurrent sessions DISTINCT categories, category alone nearly
# determines assignment and the whole problem collapses. If they collide freely, category is
# only a weak filter. This bounds how much category can ever contribute.
hdr('A. do concurrent sessions share search categories?')
disj = 0; ncoll = Counter()
for ev, lab in zip(E, L):
    s = defaultdict(set)
    for e, c in zip(ev, lab):
        if e['action'] == 'search': s[c].add(e['category'])
    cats = list(s.values())
    allc = [x for v in cats for x in v]
    disj += (len(allc) == len(set(allc)))
    ncoll[len(allc)-len(set(allc))] += 1
print('blocks where every session uses a DISTINCT category set: %.3f' % (disj/len(E)))
print('distribution of category collisions per block:', dict(sorted(ncoll.items())[:6]))
print('VERDICT: %s' % ('categories are near-disjoint -> strong signal, exploit hard'
                       if disj/len(E) > 0.6 else
                       'categories collide freely -> category is only a filter, as assumed'))

# ---------------------------------------------------------------- B: is interleaving uniform?
# If the next event is chosen uniformly among live sessions, the gap prior already extracts
# ALL interleaving information and no cleverer temporal model can beat it. Deviation from
# 1/n_active is exploitable structure.
hdr('B. is the next event uniform among live sessions?')
same = Counter(); tot = Counter()
for ev, lab in zip(E, L):
    live, done = set(), set()
    for i, (e, c) in enumerate(zip(ev, lab)):
        live.add(c)
        na = len(live)-len(done)
        if i and na > 1:
            tot[na] += 1; same[na] += (lab[i] == lab[i-1])
        if e['action'] == 'logout': done.add(c)
for na in sorted(tot):
    if tot[na] > 50:
        print('  live=%d: P(next is same session as previous)=%.3f  uniform would be %.3f  (n=%d)'
              % (na, same[na]/tot[na], 1.0/na, tot[na]))
print('VERDICT: compare the two columns. Close => gap prior is already sufficient;')
print('         systematically higher => sessions come in BURSTS, model run-length.')

# ---------------------------------------------------------------- C: per-session rate spread
# If sessions differ in pace, per-session rate identifies them. Earlier crude check said no;
# this measures it as events per unit span, which is the quantity that would matter.
hdr('C. do sessions inside one block differ in pace?')
rel = []
for ev, lab in zip(E, L):
    s = defaultdict(list)
    for i, c in enumerate(lab): s[c].append(i)
    r = [(len(v)-1)/max(v[-1]-v[0], 1) for v in s.values() if len(v) > 2]
    if len(r) > 1: rel.append(np.std(r)/(np.mean(r)+1e-9))
print('relative spread of per-session rate within a block: median %.3f' % np.median(rel))
print('VERDICT: %s' % ('>0.35 => pace is session-identifying, add a per-session rate feature'
                       if np.median(rel) > 0.35 else 'low => pace carries little identity'))

# ---------------------------------------------------------------- D: item identity links events
# THE HIGH-VALUE TEST. If a repeated item_id inside a block almost always belongs to ONE
# session, then two events sharing an item_id are near-certainly the same session -- a direct
# pairwise link the current solver only uses via the weak add_to_cart-was-viewed constraint.
hdr('D. does a repeated item_id imply the same session?')
pair_same = pair_tot = 0
for ev, lab in zip(E, L):
    by = defaultdict(list)
    for e, c in zip(ev, lab):
        if 'item_id' in e: by[e['item_id']].append(c)
    for it, cs in by.items():
        for i in range(len(cs)):
            for j in range(i+1, len(cs)):
                pair_tot += 1; pair_same += (cs[i] == cs[j])
# chance baseline: two random events in a block landing in the same session
ch_s = ch_t = 0
for lab in L:
    cnt = Counter(lab); n = len(lab)
    ch_s += sum(v*(v-1) for v in cnt.values()); ch_t += n*(n-1)
print('P(same session | same item_id) = %.4f   vs chance %.4f  (pairs=%d)'
      % (pair_same/max(pair_tot, 1), ch_s/max(ch_t, 1), pair_tot))
lift = (pair_same/max(pair_tot, 1))/(ch_s/max(ch_t, 1)+1e-9)
print('lift %.2fx vs unconditional chance' % lift)
# CONTROL: same item implies same category, and category is ALREADY a hard constraint. The
# honest question is whether item identity adds anything beyond category, so compare against
# same-category-different-item pairs rather than against unconditional chance.
cs_s = cs_t = 0
for ev, lab in zip(E, L):
    it = [(e['item_id'], c) for e, c in zip(ev, lab) if 'item_id' in e]
    for i in range(len(it)):
        for j in range(i+1, len(it)):
            if it[i][0] != it[j][0] and it[i][0]//100 == it[j][0]//100:
                cs_t += 1; cs_s += (it[i][1] == it[j][1])
base = cs_s/max(cs_t, 1)
print('CONTROL P(same session | same CATEGORY, different item) = %.4f  (pairs=%d)' % (base, cs_t))
print('item-identity lift BEYOND category: %.2fx' % ((pair_same/max(pair_tot, 1))/(base+1e-9)))
print('VERDICT: %s' % ('item identity adds signal beyond category -> add a same-item affinity'
                       if (pair_same/max(pair_tot, 1))/(base+1e-9) > 1.15 else
                       'item identity adds nothing beyond category; the 2x lift was category'))

# ---------------------------------------------------------------- E: session length structure
hdr('E. session length: predictable, and does it depend on start?')
ln, st = [], []
for ev, lab in zip(E, L):
    s = defaultdict(list)
    for i, c in enumerate(lab): s[c].append(i)
    for v in s.values(): ln.append(len(v)); st.append(v[0])
ln, st = np.array(ln), np.array(st)
print('session length: mean %.2f sd %.2f min %d max %d' % (ln.mean(), ln.std(), ln.min(), ln.max()))
print('corr(length, start position) = %.3f' % np.corrcoef(ln, st)[0, 1])
print('length distribution:', dict(sorted(Counter(ln).items())[:12]))
print('VERDICT: a tight length prior lets the decoder balance session sizes;')
print('         |corr|>0.2 means early sessions are systematically longer/shorter.')

# ---------------------------------------------------------------- F: how often is truth the
# smallest-gap candidate, and what is the per-decision ceiling for a STRONGER model?
hdr('F. capacity probe on the ambiguous decisions')
def legal(stt, e):
    a = e['action']
    if stt['done']: return False
    if stt['last'] == 'checkout' and a != 'logout': return False
    if a == 'search': return stt['last'] in ('login', 'view_item', 'add_to_cart')
    if a == 'view_item':
        return (stt['last'] in ('search', 'view_item', 'add_to_cart')
                and stt['cat'] is not None and CAT[e['item_id']//100] == stt['cat'])
    if a == 'add_to_cart':
        return stt['last'] == 'view_item' and e['item_id'] in stt['viewed']
    if a == 'checkout':
        return stt['last'] in ('view_item', 'add_to_cart') and e['cart_size'] == stt['nadd']
    if a == 'logout': return stt['last'] in ('view_item', 'add_to_cart', 'checkout')
    return False

X, Y, G = [], [], []
mingap_right = amb = 0
for ev, lab in zip(E, L):
    S, nlog = {}, 0
    sl = [0]*(len(ev)+1)
    for i in range(len(ev)-1, -1, -1):
        sl[i] = sl[i+1] + (ev[i]['action'] == 'logout')
    for pos, (e, c) in enumerate(zip(ev, lab)):
        if e['action'] == 'login':
            S[c] = dict(last='login', cat=None, viewed=[], nadd=0, done=False, pos=pos,
                        lpos=pos, cnt=1)
            continue
        ks = [k for k in sorted(S) if legal(S[k], e)]
        if len(ks) > 1:
            amb += 1
            gaps = [pos-S[k]['pos'] for k in ks]
            mingap_right += (ks[int(np.argmin(gaps))] == c)
            na = sum(1 for k in S if not S[k]['done'])
            for k in ks:
                s = S[k]
                g = pos-s['pos']
                X.append([g, g-min(gaps), 1.0*(g == min(gaps)), na, len(ks),
                          len(s['viewed']), s['nadd'], s['cnt'], pos-s['lpos'], pos,
                          len(ev)-pos, sl[pos]-na,
                          1.0*(e['action'] == 'view_item' and e.get('item_id') in s['viewed']),
                          1.0*(e['action'] == 'search' and e.get('category') == s['cat']),
                          s['cnt']/max(pos-s['lpos'], 1),
                          1.0*(s['last'] == 'checkout'), s['nadd'] > 0,
                          len(set(s['viewed'])), len(s['viewed'])-len(set(s['viewed']))])
                Y.append(1 if k == c else 0); G.append(len(G) and 0)
            G[-len(ks)] = 1   # mark group start
        s = S[c]
        if e['action'] == 'search': s['cat'] = e['category']
        elif e['action'] == 'view_item': s['viewed'].append(e['item_id'])
        elif e['action'] == 'add_to_cart': s['nadd'] += 1
        elif e['action'] == 'logout': s['done'] = True
        s['last'] = e['action']; s['pos'] = pos; s['cnt'] += 1

X = np.array(X, np.float32); Y = np.array(Y); G = np.array(G)
print('ambiguous decisions: %d | picking the smallest-gap candidate is right %.4f of the time'
      % (amb, mingap_right/max(amb, 1)))
starts = np.where(G == 1)[0]
ends = np.append(starts[1:], len(X))
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    ntr = int(0.8*len(starts))
    trm = np.zeros(len(X), bool)
    for a, b in zip(starts[:ntr], ends[:ntr]): trm[a:b] = True
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                         max_depth=None, l2_regularization=1.0)
    clf.fit(X[trm], Y[trm])
    p = clf.predict_proba(X)[:, 1]
    ok = n = 0
    for a, b in zip(starts[ntr:], ends[ntr:]):
        n += 1; ok += (Y[a:b][int(np.argmax(p[a:b]))] == 1)
    print('STRONGER model held-out per-decision accuracy: %.4f  (current ranker .7483)' % (ok/max(n, 1)))
    print('VERDICT: %s' % ('a stronger model gains -> the ranker is underpowered, upgrade it'
                           if ok/max(n, 1) > 0.77 else
                           'plateau -> near the information limit of these features; find NEW '
                           'features or accept the ceiling'))
    im = ['gap', 'gap-min', 'is_min', 'n_live', 'ncand', 'nviewed', 'nadd', 'cnt', 'age',
          'pos', 'remaining', 'slack', 'view_repeat', 'same_cat', 'rate', 'was_checkout',
          'has_cart', 'uniq_viewed', 'dup_viewed']
    print('(features: %s)' % ', '.join(im))
except ImportError:
    print('sklearn unavailable; skipped the capacity probe')

# ---------------------------------------------------------------- G: crude ceiling estimate
# Per-decision accuracy p over the ambiguous fraction f gives a rough event error rate, and
# ARI degrades roughly with misassigned events. Ties the two together so the ceiling is
# expressed in the metric we are actually scored on.
hdr('G. what per-decision accuracy would 0.79 require?')
f = amb/max(sum(len(e) for e in E), 1)
print('ambiguous fraction of all events: %.3f' % f)
for p in (0.75, 0.80, 0.85, 0.90, 0.95, 1.0):
    print('  per-decision %.2f -> misassigned events %.3f' % (p, f*(1-p)))
print('Current: per-decision ~.75 -> ~%.3f misassigned, and honest ARI is .760.' % (f*0.25))
print('Use the ratio to project what accuracy the 0.79 target implies.')
