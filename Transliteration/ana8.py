import os, json, time, bisect
from collections import Counter, defaultdict
import numpy as np, pandas as pd
T0=time.time()
def log(*a): print('[%6.1fs]'%(time.time()-T0),*a,flush=True)
D='C:/Users/Devansh Rajput/Downloads/Dakshina Dual-View Transliteration/'
train=pd.read_csv(D+'train.csv')
NVAL=int(os.environ.get('NVAL',4000))
VA=np.arange(len(train)-NVAL,len(train)); FT=np.arange(0,len(train)-NVAL)
lang=train.language.values; tj=train.target_json.values
nsk=train.native_skeleton.values; rsk=train.roman_skeleton.values
def toks(s): return str(s).split()
def hid_of(t): return t[1:-1] if len(t)>2 and t[0]=='<' and t[-1]=='>' else None
n2r=defaultdict(Counter); r2n=defaultdict(Counter)
for i in FT:
    L=lang[i]
    for a in json.loads(tj[i])['alignments']:
        nn,rr=a['native'],a['roman']
        if nn and rr: n2r[(L,nn)][rr]+=1; r2n[(L,rr)][nn]+=1
N2R={k:c.most_common(1)[0][0] for k,c in n2r.items()}
R2N={k:c.most_common(1)[0][0] for k,c in r2n.items()}
log('lex',len(N2R),len(R2N))
# sorted keys per language, forward and reversed
def mk(lex):
    fwd=defaultdict(list); rev=defaultdict(list)
    for (L,w) in lex: fwd[L].append(w); rev[L].append(w[::-1])
    for L in fwd: fwd[L].sort(); rev[L].sort()
    return fwd,rev
FN,RN=mk(N2R); FR,RR=mk(R2N)
def lcp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i
def prefix_cands(arr,w,K=8):
    j=bisect.bisect_left(arr,w); out=[]
    for k in range(max(0,j-K),min(len(arr),j+K)): out.append(arr[k])
    return out
def edist(a,b,cap=99):
    la,lb=len(a),len(b)
    prev=list(range(lb+1))
    for i in range(1,la+1):
        cur=[i]+[0]*lb
        for j in range(1,lb+1):
            cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(a[i-1]!=b[j-1]))
        prev=cur
    return prev[lb]

# collect OOV gap cases with gold
cases={'n':[],'r':[]}
for i in VA:
    L=lang[i]; g=json.loads(tj[i]); A=g['alignments']
    nt,rt=toks(nsk[i]),toks(rsk[i])
    gapn={x['hole_id']:x['text'] for x in g['native_gaps']}
    gapr={x['hole_id']:x['text'] for x in g['roman_gaps']}
    ncell=[];rcell=[]
    for ci,a in enumerate(A):
        for w in a['native'].split(): ncell.append(ci)
        for w in a['roman'].split(): rcell.append(ci)
    def walk(skel,cells,gaps):
        out=[];p=0
        for t in skel:
            h=hid_of(t)
            if h and h in gaps:
                nw=len(gaps[h].split()); out.append((h,cells[p] if p<len(cells) else -1)); p+=nw
            else: p+=1
        return out
    for (h,ci) in walk(nt,ncell,gapn):
        if ci<0: continue
        src=A[ci]['roman']
        if (L,src) not in R2N: cases['n'].append((L,src,gapn[h]))
    for (h,ci) in walk(rt,rcell,gapr):
        if ci<0: continue
        src=A[ci]['native']
        if (L,src) not in N2R: cases['r'].append((L,src,gapr[h]))
log('OOV cases: n=%d r=%d'%(len(cases['n']),len(cases['r'])))

def eval_method(d,fn,name):
    ok=0; tot=0; f1s=0
    for (L,src,gold) in cases[d]:
        p=fn(L,src) or src
        ok+= (p==gold); tot+=1
    log('  %-22s dir=%s exact %.4f (%d/%d)'%(name,d,ok/tot,ok,tot))

def copy_src(L,s): return s
for d in 'nr': eval_method(d,copy_src,'copy-source')

def make_nn(lex,fwd,rev,useprefix=True):
    def f(L,s):
        cands=set(prefix_cands(fwd.get(L,[]),s))
        if useprefix:
            for c in prefix_cands(rev.get(L,[]),s[::-1]): cands.add(c[::-1])
        if not cands: return None
        best=min(cands,key=lambda c:(edist(s,c),-lcp(s,c)))
        return lex.get((L,best))
    return f
eval_method('n',make_nn(R2N,FR,RR),'prefix/suffix-NN')
eval_method('r',make_nn(N2R,FN,RN),'prefix/suffix-NN')

# analogy: nearest neighbour by prefix, then splice suffix using a learned char-suffix map
def build_suffix_map(lex):
    # from lexicon: pairs of keys sharing a long prefix -> (nsuf, rsuf) rewrite
    sm=defaultdict(Counter)
    return sm
# measure how often gold == lex[nn] vs how often they share prefix
def diag(d,lex,fwd,rev):
    same=0; tot=0; shr=0
    for (L,src,gold) in cases[d][:2000]:
        cands=set(prefix_cands(fwd.get(L,[]),src))|set(c[::-1] for c in prefix_cands(rev.get(L,[]),src[::-1]))
        if not cands: continue
        best=min(cands,key=lambda c:(edist(src,c),-lcp(src,c)))
        out=lex.get((L,best)); tot+=1
        same+= (out==gold)
        shr+= (edist(src,best)<=2)
    log('  dir=%s NN within edit<=2 of source: %.4f ; exact %.4f'%(d,shr/tot,same/tot))
diag('n',R2N,FR,RR); diag('r',N2R,FN,RN)
