"""Dataset probes for the dual-view transliteration task. Standalone; writes nothing.

Only one thing moves the score: per-gap EXACT match (the full-sentence terms are a
deterministic function of it). So every section here asks where exactness is being lost and
whether the loss is systematic enough to fix cheaply.

Run:  python analyze.py [nrows]
"""
import os, sys, json, bisect
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

def parse_paths():
    cands = ['./dataset/public',
             '/kaggle/input/datasets/devansh66/dakshina-dual-view-transliteration',
             '/kaggle/input/dakshina-dual-view-transliteration', '/kaggle/input', '.']
    return next((Path(c) for c in cands if (Path(c)/'train.csv').exists()), Path('.'))

PUB = parse_paths()
NR = next((int(a) for a in sys.argv[1:] if a.isdigit()), 30000)
NR = int(os.environ.get('NR', NR))
tr = pd.read_csv(PUB/'train.csv', nrows=NR)
te = pd.read_csv(PUB/'test.csv')
print('train rows %d | test rows %d' % (len(tr), len(te)))
def hdr(s): print('\n=== %s ===' % s)

FIT = int(len(tr)*0.85)
n2r = defaultdict(Counter); r2n = defaultdict(Counter)
for i in range(FIT):
    L = tr.language.values[i]
    for a in json.loads(tr.target_json.values[i])['alignments']:
        if a.get('native') and a.get('roman'):
            n2r[(L, a['native'])][a['roman']] += 1
            r2n[(L, a['roman'])][a['native']] += 1
N2R = {k: c.most_common(1)[0][0] for k, c in n2r.items()}
R2N = {k: c.most_common(1)[0][0] for k, c in r2n.items()}
print('lexicon built from %d rows: n2r %d, r2n %d' % (FIT, len(N2R), len(R2N)))

# ---------------------------------------------------------------- A: per-language difficulty
# Effort is only worth spending where the mass is. If a few languages carry most of the loss,
# a language-specific remedy beats a global one.
hdr('A. per-language gap exactness and OOV rate')
rows = []
for L in sorted(set(tr.language)):
    ex_n = tot_n = oov_n = ex_r = tot_r = oov_r = 0
    for i in range(FIT, len(tr)):
        if tr.language.values[i] != L: continue
        tj = json.loads(tr.target_json.values[i])
        ns = tr.native_skeleton.values[i].split(); rs = tr.roman_skeleton.values[i].split()
        for g in tj['native_gaps']:
            k = next((j for j, t in enumerate(ns) if t == '<%s>' % g['hole_id']), None)
            if k is None or k >= len(rs): continue
            src = rs[k]; tot_n += 1
            if (L, src) in R2N: ex_n += (R2N[(L, src)] == g['text'])
            else: oov_n += 1
        for g in tj['roman_gaps']:
            k = next((j for j, t in enumerate(rs) if t == '<%s>' % g['hole_id']), None)
            if k is None or k >= len(ns): continue
            src = ns[k]; tot_r += 1
            if (L, src) in N2R: ex_r += (N2R[(L, src)] == g['text'])
            else: oov_r += 1
    if tot_n and tot_r:
        rows.append((L, tot_n+tot_r, ex_n/tot_n, oov_n/tot_n, ex_r/tot_r, oov_r/tot_r))
print('%-4s %7s %9s %8s %9s %8s' % ('lang', 'gaps', 'nat_exact', 'nat_oov', 'rom_exact', 'rom_oov'))
for L, n, a, b, c, d in sorted(rows, key=lambda x: x[2]+x[4]):
    print('%-4s %7d %9.3f %8.3f %9.3f %8.3f' % (L, n, a, b, c, d))
print('VERDICT: languages at the top of this list are where exactness is lost. If the spread is')
print('         wide, a per-language model or per-language decoding budget is justified.')

# ---------------------------------------------------------------- B: case-only mismatches
# The brief says romanizations are CASED. If a prediction is right except for capitalisation it
# scores as a MISS (0.20*F1^2 instead of 0.80+), so a systematic casing rule would be a cheap
# and large win. This measures how much of the roman loss is purely casing.
hdr('B. is any roman loss purely capitalisation?')
case_only = wrong = right = 0
first_pos_cap = first_pos_tot = 0
for i in range(FIT, len(tr)):
    L = tr.language.values[i]
    tj = json.loads(tr.target_json.values[i])
    rs = tr.roman_skeleton.values[i].split(); ns = tr.native_skeleton.values[i].split()
    for g in tj['roman_gaps']:
        k = next((j for j, t in enumerate(rs) if t == '<%s>' % g['hole_id']), None)
        if k is None or k >= len(ns): continue
        p = N2R.get((L, ns[k]))
        if p is None: continue
        if p == g['text']: right += 1
        else:
            wrong += 1
            if p.lower() == g['text'].lower(): case_only += 1
        if k == 0:
            first_pos_tot += 1; first_pos_cap += g['text'][:1].isupper()
print('in-lexicon roman gaps: right %d, wrong %d, of which CASE-ONLY %d (%.4f of wrong)'
      % (right, wrong, case_only, case_only/max(wrong, 1)))
print('gold roman token at sentence position 0 is capitalised: %.4f (%d samples)'
      % (first_pos_cap/max(first_pos_tot, 1), first_pos_tot))
print('VERDICT: if case-only is a large share of wrong, apply a casing rule (capitalise at')
print('         position 0, else match the source casing) before scoring exactness.')

# ---------------------------------------------------------------- C: counterpart index
# The pipeline reads the counterpart at the SAME token index. That must break when the two
# skeletons have different token counts. This measures how often, and whether a monotonic
# alignment that skips merged cells would recover it.
hdr('C. is the same-index counterpart correct?')
same_ok = same_tot = 0
by_eq = Counter()
for i in range(FIT, len(tr)):
    tj = json.loads(tr.target_json.values[i])
    ns = tr.native_skeleton.values[i].split(); rs = tr.roman_skeleton.values[i].split()
    al = tj['alignments']
    eq = 'Tn==Tr' if len(ns) == len(rs) else 'Tn!=Tr'
    for g in tj['native_gaps']:
        k = next((j for j, t in enumerate(ns) if t == '<%s>' % g['hole_id']), None)
        if k is None or k >= len(al): continue
        same_tot += 1
        hit = (k < len(rs) and rs[k] == al[k].get('roman'))
        same_ok += hit; by_eq[(eq, hit)] += 1
print('same-index counterpart matches the gold cell: %.4f' % (same_ok/max(same_tot, 1)))
for e in ('Tn==Tr', 'Tn!=Tr'):
    a, b = by_eq[(e, True)], by_eq[(e, False)]
    if a+b: print('   %s: correct %.4f  (n=%d)' % (e, a/(a+b), a+b))
print('VERDICT: the Tn!=Tr row tells you the ceiling for fixing the index by proper alignment.')

# ---------------------------------------------------------------- D: test-side coverage
# Label-free and directly relevant: how much of the ACTUAL test set can the lexicon serve?
hdr('D. lexicon coverage on the real test set')
N2Rf = {}; R2Nf = {}
for i in range(len(tr)):
    L = tr.language.values[i]
    for a in json.loads(tr.target_json.values[i])['alignments']:
        if a.get('native') and a.get('roman'):
            N2Rf.setdefault((L, a['native']), a['roman'])
            R2Nf.setdefault((L, a['roman']), a['native'])
cov_n = tot_n = cov_r = tot_r = 0
for row in te.itertuples():
    L = row.language
    ns = row.native_skeleton.split(); rs = row.roman_skeleton.split()
    for k, t in enumerate(ns):
        if t.startswith('<n') and k < len(rs) and not rs[k].startswith('<'):
            tot_n += 1; cov_n += ((L, rs[k]) in R2Nf)
    for k, t in enumerate(rs):
        if t.startswith('<r') and k < len(ns) and not ns[k].startswith('<'):
            tot_r += 1; cov_r += ((L, ns[k]) in N2Rf)
print('TEST native gaps: counterpart in lexicon %.4f  (n=%d)' % (cov_n/max(tot_n, 1), tot_n))
print('TEST roman  gaps: counterpart in lexicon %.4f  (n=%d)' % (cov_r/max(tot_r, 1), tot_r))
print('VERDICT: 1 - coverage is the share of gaps that MUST come from the OOV model. Multiply')
print('         by ~0.0021 per pp of OOV accuracy to size the remaining headroom.')

# ---------------------------------------------------------------- E: ambiguity ceiling
hdr('E. how much in-lexicon loss is irreducible ambiguity?')
for nm, D in (('native<-roman', r2n), ('roman<-native', n2r)):
    amb = [c for c in D.values() if sum(c.values()) >= 3]
    if not amb: continue
    top = np.mean([c.most_common(1)[0][1]/sum(c.values()) for c in amb])
    multi = np.mean([len(c) > 1 for c in amb])
    print('  %s: entries seen >=3 times -> %.3f have >1 target; top choice holds %.3f of mass'
          % (nm, multi, top))
print('VERDICT: "top choice holds X" is the ceiling for any context-free lexicon in that')
print('         direction. If X is already near the observed exactness, the lexicon is done.')

# ---------------------------------------------------------------- F: token shape
hdr('F. what do the hard tokens look like?')
Ls = defaultdict(list)
for i in range(FIT, len(tr)):
    L = tr.language.values[i]
    tj = json.loads(tr.target_json.values[i])
    ns = tr.native_skeleton.values[i].split(); rs = tr.roman_skeleton.values[i].split()
    for g in tj['roman_gaps']:
        k = next((j for j, t in enumerate(rs) if t == '<%s>' % g['hole_id']), None)
        if k is None or k >= len(ns): continue
        p = N2R.get((L, ns[k]))
        Ls['len_gold'].append(len(g['text']))
        Ls['hit' if p == g['text'] else 'miss'].append(len(g['text']))
if Ls['hit'] and Ls['miss']:
    print('roman gold length: exact-hit mean %.2f | miss mean %.2f'
          % (np.mean(Ls['hit']), np.mean(Ls['miss'])))
    print('VERDICT: if misses are much longer, the OOV model should get extra beam width on long')
    print('         tokens, where a single character error already costs the whole exact match.')
