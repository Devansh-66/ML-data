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
nsk=train.native_skeleton.values; rsk=train.roman_skeleton.values; hj=train.hole_ids_json.values
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
def prefix_cands(arr,w,K=20):
    j=bisect.bisect_left(arr,w)
    return arr[max(0,j-K):j+K]
def edist(a,b):
    lb=len(b); prev=list(range(lb+1))
    for i in range(1,len(a)+1):
        cur=[i]+[0]*lb; ai=a[i-1]
        for j in range(1,lb+1):
            cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(ai!=b[j-1]))
        prev=cur
    return prev[lb]
W=8;PMIN=2;TOPR=5;NC=10
def mine_rules(lex,fwd):
    R=defaultdict(Counter)
    for L,arr in fwd.items():
        for i in range(len(arr)):
            v1=arr[i]; t1=lex[(L,v1)]
            for j in range(i+1,min(i+W,len(arr))):
                v2=arr[j]; t2=lex[(L,v2)]
                p=lcp(v1,v2); q=lcp(t1,t2)
                if p<PMIN or q<2: continue
                a,b=v1[p:],v2[p:]; c,d=t1[q:],t2[q:]
                if len(a)>7 or len(b)>7 or len(c)>9 or len(d)>9: continue
                R[(L,a,b)][(c,d)]+=1; R[(L,b,a)][(d,c)]+=1
    return R
log('mining rules'); RUL_N2R=mine_rules(N2R,FN); RUL_R2N=mine_rules(R2N,FR); log('rules done')
CACHE={}
def make_analogy(lex,fwd,rev,RUL,tag):
    def f(L,s):
        key=(tag,L,s)
        if key in CACHE: return CACHE[key]
        arr=fwd.get(L,[]); r=None
        if arr:
            cands=set(prefix_cands(arr,s))
            for c in prefix_cands(rev.get(L,[]),s[::-1]): cands.add(c[::-1])
            scored=sorted(cands,key=lambda c:(edist(s,c),-lcp(s,c)))[:NC]
            V=Counter()
            for rank,v in enumerate(scored):
                p=lcp(s,v)
                if p<PMIN: continue
                a,b=v[p:],s[p:]; t=lex.get((L,v))
                if t is None: continue
                if a==b:
                    V[t]+=6.0/(1+rank); continue
                cc=RUL.get((L,a,b))
                if not cc: continue
                tot=sum(cc.values())
                for (cf,ct),n in cc.most_common(TOPR):
                    if t.endswith(cf) and (len(t)-len(cf))>=1:
                        V[t[:len(t)-len(cf)]+ct]+=(n/tot)*(3.0/(1+rank)); break
            if V: r=V.most_common(1)[0][0]
            if r is None and scored: r=lex.get((L,scored[0]))
        CACHE[key]=r; return r
    return f
AN_n=make_analogy(R2N,FR,RR,RUL_R2N,'n'); AN_r=make_analogy(N2R,FN,RN,RUL_N2R,'r')

def predict_row(L,nskel,rskel,holes,Cmode='min',useAN=True):
    nt,rt=toks(nskel),toks(rskel)
    C=min(len(nt),len(rt)) if Cmode=='min' else len(nt)
    nf,rf=list(nt),list(rt); ng,rg=[],[]
    for k,t in enumerate(nt):
        h=hid_of(t)
        if h and h.startswith('n'):
            src=rt[k] if k<len(rt) and not hid_of(rt[k]) else ''
            out=R2N.get((L,src)) if src else None
            if out is None and src and useAN: out=AN_n(L,src)
            if not out: out=src or t
            nf[k]=out; ng.append({'hole_id':h,'text':out})
    for k,t in enumerate(rt):
        h=hid_of(t)
        if h and h.startswith('r'):
            src=nt[k] if k<len(nt) and not hid_of(nt[k]) else ''
            out=N2R.get((L,src)) if src else None
            if out is None and src and useAN: out=AN_r(L,src)
            if not out: out=src or t
            rf[k]=out; rg.append({'hole_id':h,'text':out})
    want=list(holes)
    ng=[g for g in ng if g['hole_id'] in want]; rg=[g for g in rg if g['hole_id'] in want]
    have={g['hole_id'] for g in ng}|{g['hole_id'] for g in rg}
    for h in want:
        if h not in have: (ng if h.startswith('n') else rg).append({'hole_id':h,'text':''})
    al=[{'native':nf[k] if k<len(nf) else '','roman':rf[k] if k<len(rf) else ''} for k in range(C)]
    return {'alignments':al,'native_gaps':ng,'native_text':' '.join(nf),
            'roman_gaps':rg,'romanized_text':' '.join(rf)}
def f1char(p,g):
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    cp,cg=Counter(p),Counter(g); inter=sum((cp&cg).values())
    if inter==0: return 0.0
    pr,rc=inter/len(p),inter/len(g); return 2*pr*rc/(pr+rc)
def score_row(pred,gold):
    pn={g['hole_id'] for g in pred['native_gaps']}; gn={g['hole_id'] for g in gold['native_gaps']}
    pr_={g['hole_id'] for g in pred['roman_gaps']}; gr={g['hole_id'] for g in gold['roman_gaps']}
    if pn!=gn or pr_!=gr or len(pred['alignments'])!=len(gold['alignments']): return 0.0
    def gs(P,G):
        gd={g['hole_id']:g['text'] for g in G}; pd_={g['hole_id']:g['text'] for g in P}
        if not gd: return 1.0
        s=0.0
        for h,gt in gd.items():
            pt=pd_.get(h,''); f=f1char(pt,gt)
            s+=(0.80+0.20*f) if pt==gt else 0.20*f*f
        return s/len(gd)
    sn=gs(pred['native_gaps'],gold['native_gaps']); sr=gs(pred['roman_gaps'],gold['roman_gaps'])
    fn=1.0 if pred['native_text']==gold['native_text'] else 0.10*f1char(pred['native_text'],gold['native_text'])
    fr=1.0 if pred['romanized_text']==gold['romanized_text'] else 0.10*f1char(pred['romanized_text'],gold['romanized_text'])
    ap=Counter((a['native'],a['roman']) for a in pred['alignments']); ag=Counter((a['native'],a['roman']) for a in gold['alignments'])
    inter=sum((ap&ag).values())
    fa=0.0 if inter==0 else (lambda p_,r_:2*p_*r_/(p_+r_))(inter/sum(ap.values()),inter/sum(ag.values()))
    return 0.25*sn+0.25*sr+0.20*fn+0.20*fr+0.10*fa
for cfg in [('min',True)]:
    sc=[]; exN=[0,0]; exR=[0,0]
    for i in VA:
        gold=json.loads(tj[i])
        p=predict_row(lang[i],nsk[i],rsk[i],json.loads(hj[i]),cfg[0],cfg[1])
        sc.append(score_row(p,gold))
        gn={g['hole_id']:g['text'] for g in gold['native_gaps']}; gr={g['hole_id']:g['text'] for g in gold['roman_gaps']}
        for g in p['native_gaps']: exN[0]+=(g['text']==gn.get(g['hole_id'])); exN[1]+=1
        for g in p['roman_gaps']: exR[0]+=(g['text']==gr.get(g['hole_id'])); exR[1]+=1
    log('C=%-3s analogy=%-5s  SCORE %.4f  exactN %.4f exactR %.4f'%(cfg[0],cfg[1],np.mean(sc),exN[0]/exN[1],exR[0]/exR[1]))
