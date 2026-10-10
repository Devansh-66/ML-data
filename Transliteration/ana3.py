import os, sys, json, time
from collections import Counter, defaultdict
import numpy as np, pandas as pd
T0=time.time()
def log(*a): print('[%6.1fs]'%(time.time()-T0),*a,flush=True)
D='C:/Users/Devansh Rajput/Downloads/Dakshina Dual-View Transliteration/'
train=pd.read_csv(D+'train.csv')
NVAL=int(os.environ.get('NVAL',4000)); NGRAM=3; NEIGH=40
VA=np.arange(len(train)-NVAL,len(train)); FT=np.arange(0,len(train)-NVAL)
def toks(s): return str(s).split()
def hid_of(t): return t[1:-1] if len(t)>2 and t[0]=='<' and t[-1]=='>' else None
lang=train.language.values; nsk=train.native_skeleton.values; rsk=train.roman_skeleton.values
tj=train.target_json.values; hj=train.hole_ids_json.values

def build_lex(rows):
    n2r=defaultdict(Counter); r2n=defaultdict(Counter)
    for i in rows:
        L=lang[i]
        for a in json.loads(tj[i])['alignments']:
            nn,rr=a.get('native',''),a.get('roman','')
            if nn and rr: n2r[(L,nn)][rr]+=1; r2n[(L,rr)][nn]+=1
    return ({k:c.most_common(1)[0][0] for k,c in n2r.items()},
            {k:c.most_common(1)[0][0] for k,c in r2n.items()})
def build_index(keys):
    idx=defaultdict(list)
    for (L,w) in keys:
        s='^'+w+'$'
        for j in range(max(len(s)-NGRAM+1,1)): idx[(L,s[j:j+NGRAM])].append(w)
    return idx
def nearest(idx,lex,L,w):
    s='^'+w+'$'; cand=Counter()
    for j in range(max(len(s)-NGRAM+1,1)):
        for c in idx.get((L,s[j:j+NGRAM]),())[:200]: cand[c]+=1
    if not cand: return None
    best,bs=None,-1.0
    for c,ov in cand.most_common(NEIGH):
        sc=2.0*ov/(len(s)+len(c)+2)
        if sc>bs: best,bs=c,sc
    return lex.get((L,best)) if best is not None else None

def predict_row(L,nskel,rskel,holes,N2R,R2N,IN,IR,Cmode='Tn'):
    nt,rt=toks(nskel),toks(rskel)
    C=len(nt) if Cmode=='Tn' else min(len(nt),len(rt))
    nf,rf=list(nt),list(rt); ng,rg=[],[]
    for k,t in enumerate(nt):
        h=hid_of(t)
        if h and h.startswith('n'):
            src=rt[k] if k<len(rt) and not hid_of(rt[k]) else ''
            out=R2N.get((L,src)) if src else None
            if out is None and src: out=nearest(IR,R2N,L,src)
            if not out: out=src or t
            nf[k]=out; ng.append({'hole_id':h,'text':out})
    for k,t in enumerate(rt):
        h=hid_of(t)
        if h and h.startswith('r'):
            src=nt[k] if k<len(nt) and not hid_of(nt[k]) else ''
            out=N2R.get((L,src)) if src else None
            if out is None and src: out=nearest(IN,N2R,L,src)
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
def gapscore(P,G):
    gd={g['hole_id']:g['text'] for g in G}; pd_={g['hole_id']:g['text'] for g in P}
    if not gd: return 1.0,0,0
    s=0.0; ex=0
    for h,gt in gd.items():
        pt=pd_.get(h,''); f=f1char(pt,gt)
        s+=(0.80+0.20*f) if pt==gt else 0.20*f*f
        ex+=(pt==gt)
    return s/len(gd),ex,len(gd)
def components(pred,gold):
    pn={g['hole_id'] for g in pred['native_gaps']}; gn={g['hole_id'] for g in gold['native_gaps']}
    pr_={g['hole_id'] for g in pred['roman_gaps']}; gr={g['hole_id'] for g in gold['roman_gaps']}
    gate = not(pn!=gn or pr_!=gr or len(pred['alignments'])!=len(gold['alignments']))
    sn,exn,tn=gapscore(pred['native_gaps'],gold['native_gaps'])
    sr,exr,tr_=gapscore(pred['roman_gaps'],gold['roman_gaps'])
    fn=1.0 if pred['native_text']==gold['native_text'] else 0.10*f1char(pred['native_text'],gold['native_text'])
    fr=1.0 if pred['romanized_text']==gold['romanized_text'] else 0.10*f1char(pred['romanized_text'],gold['romanized_text'])
    ap=Counter((a.get('native',''),a.get('roman','')) for a in pred['alignments'])
    ag=Counter((a.get('native',''),a.get('roman','')) for a in gold['alignments'])
    inter=sum((ap&ag).values())
    fa=0.0 if inter==0 else (lambda p_,r_:2*p_*r_/(p_+r_))(inter/sum(ap.values()),inter/sum(ag.values()))
    tot=(0.25*sn+0.25*sr+0.20*fn+0.20*fr+0.10*fa) if gate else 0.0
    return dict(gate=gate,sn=sn,sr=sr,fn=fn,fr=fr,fa=fa,tot=tot,exn=exn,tn=tn,exr=exr,tr=tr_,
                nfull=float(pred['native_text']==gold['native_text']),
                rfull=float(pred['romanized_text']==gold['romanized_text']),
                gatecnt=float(len(pred['alignments'])==len(gold['alignments'])),
                gatekey=float(pn==gn and pr_==gr))

log('building lex'); N2R,R2N=build_lex(FT); IN=build_index(N2R.keys()); IR=build_index(R2N.keys())
log('lex',len(N2R),len(R2N))
for Cmode in ['Tn','min']:
    rec=defaultdict(list); EXN=EXR=TN=TR=0
    for i in VA:
        gold=json.loads(tj[i])
        p=predict_row(lang[i],nsk[i],rsk[i],json.loads(hj[i]),N2R,R2N,IN,IR,Cmode)
        c=components(p,gold)
        for k in ['gate','sn','sr','fn','fr','fa','tot','nfull','rfull','gatecnt','gatekey']: rec[k].append(c[k])
        EXN+=c['exn']; TN+=c['tn']; EXR+=c['exr']; TR+=c['tr']
    m={k:np.mean(v) for k,v in rec.items()}
    g=np.array(rec['gate'],bool)
    log('--- Cmode=%s  TOTAL %.4f'%(Cmode,m['tot']))
    log('  gate %.4f (count-ok %.4f, keys-ok %.4f)'%(m['gate'],m['gatecnt'],m['gatekey']))
    log('  among gated-in rows: sn=%.4f sr=%.4f fn=%.4f fr=%.4f fa=%.4f'%(
        np.mean(np.array(rec['sn'])[g]),np.mean(np.array(rec['sr'])[g]),
        np.mean(np.array(rec['fn'])[g]),np.mean(np.array(rec['fr'])[g]),np.mean(np.array(rec['fa'])[g])))
    log('  weighted contributions: 0.25sn=%.4f 0.25sr=%.4f 0.20fn=%.4f 0.20fr=%.4f 0.10fa=%.4f'%(
        0.25*np.mean(np.array(rec['sn'])[g])*m['gate'],0.25*np.mean(np.array(rec['sr'])[g])*m['gate'],
        0.20*np.mean(np.array(rec['fn'])[g])*m['gate'],0.20*np.mean(np.array(rec['fr'])[g])*m['gate'],
        0.10*np.mean(np.array(rec['fa'])[g])*m['gate']))
    log('  gap exact-match: native %d/%d=%.4f roman %d/%d=%.4f'%(EXN,TN,EXN/TN,EXR,TR,EXR/TR))
    log('  full-sentence exact: native %.4f roman %.4f (over all rows)'%(m['nfull'],m['rfull']))
