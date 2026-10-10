"""Joint-sequence transducer, v2.
One hard-EM alignment per language pair set (native char -> roman chunk), reused both ways:
  dir 'r' (native->roman): emit a roman chunk per native char.
  dir 'n' (roman->native): segment the roman word, emit a native string per roman chunk.
"""
import os, json, time, math, bisect, pickle
from collections import Counter, defaultdict
import numpy as np, pandas as pd
T0=time.time()
def log(*a): print('[%6.1fs]'%(time.time()-T0),*a,flush=True)
D='C:/Users/Devansh Rajput/Downloads/Dakshina Dual-View Transliteration/'
MAXCH=int(os.environ.get('MAXCH',5)); BAND=int(os.environ.get('BAND',5))
EMIT=int(os.environ.get('EMIT',3)); MPL=int(os.environ.get('MPL',6000))
BEAM=int(os.environ.get('BEAM',10)); NCAND=int(os.environ.get('NCAND',6))
LB=float(os.environ.get('LB',1.5))
DEF=-11.0

def align(s,t,LP,first=False):
    La,Lb=len(s),len(t)
    if La==0 or Lb>La*MAXCH: return None
    NEG=-1e17; dp=[NEG]*(Lb+1); dp[0]=0.0
    bp=[[0]*(Lb+1) for _ in range(La)]; r=Lb/La
    for i in range(La):
        c=s[i]; tab=None if first else LP.get(c)
        nd=[NEG]*(Lb+1); bpi=bp[i]
        ctr=(i+1)*r; lo=max(0,int(ctr)-BAND); hi=min(Lb,int(ctr)+BAND+1)
        for j in range(min(Lb,int(i*r)+BAND+1)+1):
            v=dp[j]
            if v<NEG/2: continue
            for l in range(min(MAXCH,Lb-j)+1):
                k=j+l
                if k<lo or k>hi: continue
                sc=v-abs(l-r) if first else v+(tab.get(t[j:k],DEF) if tab is not None else DEF)
                if sc>nd[k]: nd[k]=sc; bpi[k]=j
        dp=nd
    if dp[Lb]<NEG/2: return None
    out=[]; j=Lb
    for i in range(La-1,-1,-1):
        pj=bp[i][j]; out.append((s[i],t[pj:j])); j=pj
    if j!=0: return None
    out.reverse(); return out

def train_align(pairs,iters):
    LP=None
    for it in range(iters):
        cnt=defaultdict(Counter); nf=0
        for (s,t) in pairs:
            a=align(s,t,LP,first=(it==0))
            if a is None: nf+=1; continue
            for c,ch in a: cnt[c][ch]+=1
        LP={c:{k:math.log((v+0.1)/(sum(C.values())+len(C)*0.1)) for k,v in C.items()} for c,C in cnt.items()}
        log('   EM %d chars=%d fail=%d'%(it,len(LP),nf))
    return LP

def segs_of(a):
    """[(nchar, rchunk)] -> [(rchunk, nstr)] with every rchunk non-empty."""
    S=[]
    for c,ch in a:
        if ch=='':
            if S: S[-1]=(S[-1][0],S[-1][1]+c)
            else: S.append(('',c))
        else:
            if S and S[0][0]=='' and len(S)==1: S=[(ch,S[0][1]+c)]
            else: S.append((ch,c))
    return [x for x in S if x[0]]

def build(pairs,langs,LP):
    A1=defaultdict(Counter); A2=defaultdict(Counter); A3=defaultdict(Counter)
    B1=defaultdict(Counter); B2=defaultdict(Counter); B3=defaultdict(Counter)
    for (s,t),L in zip(pairs,langs):
        a=align(s,t,LP)
        if a is None: continue
        pc='<s>'
        for i,(c,ch) in enumerate(a):
            p=s[i-1] if i>0 else '^'; nx=s[i+1] if i+1<len(s) else '$'
            k2=(L,p,c,nx); A1[(L,c)][ch]+=1; A2[k2][ch]+=1; A3[k2+(pc,)][ch]+=1; pc=ch
        S=segs_of(a); pos=0; pv='<s>'
        for (ch,ns) in S:
            p=t[pos-1] if pos>0 else '^'; nx=t[pos+len(ch)] if pos+len(ch)<len(t) else '$'
            k2=(L,p,ch,nx); B1[(L,ch)][ns]+=1; B2[k2][ns]+=1; B3[k2+(pv,)][ns]+=1
            pv=ns; pos+=len(ch)
    return (dict(A1),dict(A2),dict(A3)),(dict(B1),dict(B2),dict(B3))

def decodeA(s,L,M):
    A1,A2,A3=M; n=len(s)
    if n==0: return ''
    st=[(0.0,'<s>','')]
    for i in range(n):
        c=s[i]; p=s[i-1] if i>0 else '^'; nx=s[i+1] if i+1<n else '$'
        k2=(L,p,c,nx); cd=A2.get(k2) or A1.get((L,c)) or Counter({c:1})
        tot=sum(cd.values()); cl=cd.most_common(NCAND); new={}
        for (sc,pc,out) in st:
            c3=A3.get(k2+(pc,)); t3=sum(c3.values()) if c3 else 0
            for ch,nn in cl:
                p2=nn/tot
                pr=0.35*p2+(0.65*c3.get(ch,0)/t3 if c3 else 0.65*p2)
                ns=sc+math.log(pr+1e-9); o=out+ch; k=(ch,o)
                if k not in new or new[k][0]<ns: new[k]=(ns,ch,o)
        st=sorted(new.values(),key=lambda x:-x[0])[:BEAM]
    return st[0][2]

def decodeB(r,L,M):
    B1,B2,B3=M; n=len(r)
    if n==0: return ''
    st=defaultdict(list); st[0]=[(0.0,'<s>','')]
    for j in range(n):
        if not st[j]: continue
        cur=sorted(st[j],key=lambda x:-x[0])[:BEAM]; st[j]=cur
        for l in range(1,MAXCH+1):
            k=j+l
            if k>n: break
            ch=r[j:k]; p=r[j-1] if j>0 else '^'; nx=r[k] if k<n else '$'
            k2=(L,p,ch,nx); cd=B2.get(k2) or B1.get((L,ch))
            if cd is None:
                if l>1: continue
                cd=Counter({ch:1})
            tot=sum(cd.values()); cl=cd.most_common(NCAND); buck=st[k]
            for (sc,pv,out) in cur:
                c3=B3.get(k2+(pv,)); t3=sum(c3.values()) if c3 else 0
                for ns_,cnt in cl:
                    p2=cnt/tot
                    pr=0.35*p2+(0.65*c3.get(ns_,0)/t3 if c3 else 0.65*p2)
                    buck.append((sc+math.log(pr+1e-9)+LB,ns_,out+ns_))
    fin=st.get(n)
    return sorted(fin,key=lambda x:-x[0])[0][2] if fin else r

exec(open(D+'ana8.py').read().split("def eval_method")[0])
log('OOV cases n=%d r=%d'%(len(cases['n']),len(cases['r'])))
per=defaultdict(list)
for (L,w),C in n2r.items(): per[L].append((sum(C.values()),w))
pairs=[]; langs=[]
for L,v in per.items():
    v.sort(reverse=True)
    for _,w in v[:MPL]: pairs.append((w,N2R[(L,w)])); langs.append(L)
log('training on %d pairs (MPL=%d, EMIT=%d)'%(len(pairs),MPL,EMIT))
LP=train_align(pairs,EMIT)
MA,MB=build(pairs,langs,LP)
log('A sizes %d %d %d | B sizes %d %d %d'%(len(MA[0]),len(MA[1]),len(MA[2]),len(MB[0]),len(MB[1]),len(MB[2])))
for d,fn,M in (('r',decodeA,MA),('n',decodeB,MB)):
    ok=0; t=time.time()
    for (L,src,gold) in cases[d]: ok+= (fn(src,L,M)==gold)
    log('TRANSDUCER dir=%s exact %.4f  (%.1fs / %d words)'%(d,ok/len(cases[d]),time.time()-t,len(cases[d])))
pickle.dump((LP,MA,MB),open(D+'models2.pkl','wb'))
