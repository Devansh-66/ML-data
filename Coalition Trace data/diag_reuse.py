"""Is the query built from the SAME source audio as the references, or merely the same voices?

If segments are reused, normalised waveform cross-correlation between a bin and the true key's
reference peaks near 1.0 and this is an exact-matching problem. If the references are only
other recordings of the same speakers, the peak stays low and it is a speaker-ID problem.
Everything here reads only supplied public files.
"""
import sys, wave
from pathlib import Path
import numpy as np, pandas as pd

PUB = Path('.')
tr = pd.read_csv(PUB/'train.csv')
NROW = int(sys.argv[1]) if len(sys.argv) > 1 else 8

def rd(p):
    with wave.open(str(PUB/p), 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)

def ncc_max(seg, ref):
    """Peak normalised cross-correlation of seg against every offset in ref."""
    n, m = len(seg), len(ref)
    L = 1 << (n+m-1).bit_length()
    c = np.fft.irfft(np.fft.rfft(ref, L)*np.conj(np.fft.rfft(seg, L)), L)[:m-n+1]
    cs = np.concatenate([[0.0], np.cumsum(ref*ref)])
    en = cs[n:m+1]-cs[:m-n+1]
    d = np.sqrt(en*(seg@seg)) + 1e-9
    return float(np.max(np.abs(c)/d))

pres, absent = [], []
for _, r in tr.head(NROW).iterrows():
    q = rd(r.query_path)
    refs = [rd(r['voice_%d_path' % j]) for j in range(4)]
    code = r.activity_code[1:]
    for b, ch in enumerate(code):
        seg = q[b*4000:(b+1)*4000]
        if seg@seg < 1e-6: continue
        m = int(ch, 16)
        for j in range(4):
            v = ncc_max(seg, refs[j])
            (pres if (m >> j) & 1 else absent).append(v)

pres, absent = np.array(pres), np.array(absent)
print('present  n=%4d  NCC mean %.3f  median %.3f  p90 %.3f  max %.3f'
      % (len(pres), pres.mean(), np.median(pres), np.percentile(pres, 90), pres.max()))
print('absent   n=%4d  NCC mean %.3f  median %.3f  p90 %.3f  max %.3f'
      % (len(absent), absent.mean(), np.median(absent), np.percentile(absent, 90), absent.max()))
allv = np.concatenate([pres, absent]); y = np.concatenate([np.ones(len(pres)), np.zeros(len(absent))])
rk = np.argsort(np.argsort(allv))+1.0
print('AUC of raw waveform NCC: %.4f'
      % ((rk[y > 0].sum()-len(pres)*(len(pres)+1)/2)/(len(pres)*len(absent))))
print('\nreuse verdict: %s' % ('SEGMENTS REUSED -> exact matching problem'
                               if np.median(pres) > 0.5 else
                               'no reuse -> genuine speaker-ID problem'))
