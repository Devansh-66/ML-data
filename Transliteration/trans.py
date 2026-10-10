"""Joint-sequence character transducer: hard-EM 1-to-many char alignment + context graphone model."""
import os, json, time, math, bisect, pickle
from collections import Counter, defaultdict
import numpy as np, pandas as pd
T0=time.time()
def log(*a): print('[%6.1fs]'%(time.time()-T0),*a,flush=True)
D='C:/Users/Devansh Rajput/Downloads/Dakshina Dual-View Transliteration/'
MAXCH=int(os.environ.get('MAXCH',5)); BAND=int(os.environ.get('BAND',5))
EMIT=int(os.environ.get('EMIT',3)); MPL=int(os.environ.get('MPL',6000))
BEAM=int(os.environ.get('BEAM',8)); NCAND=int(os.environ.get('NCAND',6))
DEF=-11.0

def align(s,t,LP,first=False):
    La,Lb=len(s),len(t)
    if La==0 or Lb>La*MAXCH: return None
    NEG=-1e17
    dp=[NEG]*(Lb+1); dp[0]=0.0
    bp=[[0]*(Lb+1) for _ in range(La)]
    r=Lb/La
    for i in range(La):
        c=s[i]; tab=None if first else LP.get(c)
        nd=[NEG]*(Lb+1); bpi=bp[i]
        ctr=(i+1)*r; lo=max(0,int(ctr)-BAND); hi=min(Lb,int(ctr)+BAND+1)
        for j in range(min(Lb,int(i*r)+BAND+1)+1):
            v=dp[j]
            if v<NEG/2: continue
            mx=min(MAXCH,Lb-j)
            for l in range(mx+1):
                k=j+l
                if k<lo or k>hi: continue
                if first: sc=v-abs(l-r)
                else: sc=v+(tab.get(t[j:k],DEF) if tab is not None else DEF)
                if sc>nd[k]: nd[k]=sc; bpi[k]=j
        dp=nd
    if dp[Lb]<NEG/2: return None
    out=[]; j=Lb
    for i in range(La-1,-1,-1):
        pj=bp[i][j]; out.append((s[i],t[pj:j])); j=pj
    if j!=0: return None
    out.reverse(); return out

def train_align(pairs,iters=3):
    LP=None
    for it in range(iters):
        cnt=defaultdict(Counter); nfail=0
        for (s,t) in pairs:
            a=align(s,t,LP,first=(it==0))
            if a is None: nfail+=1; continue
            for c,ch in a: cnt[c][ch]+=1
        LP={}
        for c,C in cnt.items():
            tot=sum(C.values())+len(C)*0.1
            LP[c]={k:math.log((v+0.1)/tot) for k,v in C.items()}
        log('   EM iter %d  chars=%d fail=%d'%(it,len(LP),nfail))
    return LP

def build_ctx(pairs,LP,langs):
    C1=defaultdict(Counter); C2=defaultdict(Counter); C3=defaultdict(Counter)
    for (s,t),L in zip(pairs,langs):
        a=align(s,t,LP)
        if a is None: continue
        pc='<s>'
        for i,(c,ch) in enumerate(a):
            p=s[i-1] if i>0 else '^'; nx=s[i+1] if i+1<len(s) else '$'
            k2=(L,p,c,nx)
            C1[(L,c)][ch]+=1; C2[k2][ch]+=1; C3[k2+(pc,)][ch]+=1
            pc=ch
    return dict(C1),dict(C2),dict(C3)

def decode(s,L,C1,C2,C3):
    n=len(s)
    if n==0: return ''
    states=[(0.0,'<s>','')]
    for i in range(n):
        c=s[i]; p=s[i-1] if i>0 else '^'; nx=s[i+1] if i+1<n else '$'
        k2=(L,p,c,nx)
        cd=C2.get(k2)
        if cd is None: cd=C1.get((L,c))
        if cd is None: cd=Counter({c:1})
        tot2=sum(cd.values()); cl=cd.most_common(NCAND)
        new={}
        for (sc,pc,out) in states:
            c3=C3.get(k2+(pc,)); t3=sum(c3.values()) if c3 else 0
            for ch,nn in cl:
                p2=nn/tot2
                pr=0.35*p2+(0.65*(c3.get(ch,0)/t3) if c3 else 0.65*p2)
                ns=sc+math.log(pr+1e-9); o=out+ch; key=(ch,o)
                if key not in new or new[key][0]<ns: new[key]=(ns,ch,o)
        states=sorted(new.values(),key=lambda x:-x[0])[:BEAM]
    return states[0][2]

# ---------------- data ----------------
exec(open(D+'ana8.py').read().split("def eval_method")[0])
log('lexicons + OOV cases ready: n=%d r=%d'%(len(cases['n']),len(cases['r'])))

def select(cnt_lex, top_lex):
    """cnt_lex: (L,src)->Counter(tgt). Keep MPL most-frequent srcs per language."""
    per=defaultdict(list)
    for (L,w),C in cnt_lex.items(): per[L].append((sum(C.values()),w))
    pairs=[]; langs=[]
    for L,v in per.items():
        v.sort(reverse=True)
        for _,w in v[:MPL]:
            pairs.append((w,top_lex[(L,w)])); langs.append(L)
    return pairs,langs

MODELS={}
for tag,cnt_lex,top_lex in (('r',n2r,N2R),('n',r2n,R2N)):
    pairs,langs=select(cnt_lex,top_lex)
    log('dir=%s training on %d pairs'%(tag,len(pairs)))
    LP=train_align(pairs,iters=EMIT)
    C1,C2,C3=build_ctx(pairs,LP,langs)
    log('   ctx sizes %d %d %d'%(len(C1),len(C2),len(C3)))
    MODELS[tag]=(C1,C2,C3)

for d in 'nr':
    C1,C2,C3=MODELS[d]
    ok=0; t=time.time()
    for (L,src,gold) in cases[d]:
        ok+= (decode(src,L,C1,C2,C3)==gold)
    log('TRANSDUCER dir=%s exact %.4f  (%.1fs for %d words)'%(d,ok/len(cases[d]),time.time()-t,len(cases[d])))
pickle.dump(MODELS,open(D+'models.pkl','wb'))
log('saved models.pkl')
