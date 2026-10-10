import os, json, time
from collections import Counter, defaultdict
import numpy as np, pandas as pd
T0=time.time()
def log(*a): print('[%6.1fs]'%(time.time()-T0),*a,flush=True)
D='C:/Users/Devansh Rajput/Downloads/Dakshina Dual-View Transliteration/'
train=pd.read_csv(D+'train.csv')
NVAL=int(os.environ.get('NVAL',4000))
VA=np.arange(len(train)-NVAL,len(train)); FT=np.arange(0,len(train)-NVAL)
lang=train.language.values; nsk=train.native_skeleton.values; rsk=train.roman_skeleton.values
tj=train.target_json.values; hj=train.hole_ids_json.values
def toks(s): return str(s).split()
def hid_of(t): return t[1:-1] if len(t)>2 and t[0]=='<' and t[-1]=='>' else None

n2r=defaultdict(Counter); r2n=defaultdict(Counter)
for i in FT:
    L=lang[i]
    for a in json.loads(tj[i])['alignments']:
        nn,rr=a['native'],a['roman']
        if nn and rr: n2r[(L,nn)][rr]+=1; r2n[(L,rr)][nn]+=1
log('lex built',len(n2r),len(r2n))

# For each gap, recover the GOLD cell it belongs to, and gold counterpart.
stats=Counter(); dirstats={'n':Counter(),'r':Counter()}
srcidx_ok={'n':Counter(),'r':Counter()}
for i in VA:
    g=json.loads(tj[i]); A=g['alignments']; L=lang[i]
    nt=toks(nsk[i]); rt=toks(rsk[i])
    # map: native full tokens from cells, with cell index per token
    ncell=[]; rcell=[]
    for ci,a in enumerate(A):
        for w in a['native'].split(): ncell.append((w,ci))
        for w in a['roman'].split(): rcell.append((w,ci))
    # reconstruct native gap positions: skeleton tokens align to full native tokens with gaps expanding
    gapn={x['hole_id']:x['text'] for x in g['native_gaps']}
    gapr={x['hole_id']:x['text'] for x in g['roman_gaps']}
    # walk skeleton
    def walk(skel,cells,gaps):
        out=[]; p=0
        for t in skel:
            h=hid_of(t)
            if h and h in gaps:
                nw=len(gaps[h].split()); out.append((h,p,cells[p][1] if p<len(cells) else -1,nw)); p+=nw
            else: p+=1
        return out
    nn=walk(nt,ncell,gapn); rr=walk(rt,rcell,gapr)
    for (h,p,ci,nw) in nn:
        d='n'; gold=gapn[h]
        # gold counterpart = roman of that cell
        src_gold=A[ci]['roman'] if ci>=0 else ''
        k=nt.index('<'+h+'>')
        src_pred=rt[k] if k<len(rt) and not hid_of(rt[k]) else ''
        dirstats[d]['total']+=1
        dirstats[d]['srcidx_correct']+= (src_pred==src_gold)
        c=r2n.get((L,src_gold))
        dirstats[d]['goldsrc_in_lex']+= (c is not None)
        if c is not None:
            dirstats[d]['goldsrc_top1']+= (c.most_common(1)[0][0]==gold)
            dirstats[d]['goldsrc_inlist']+= (gold in c)
            dirstats[d]['goldsrc_top3']+= (gold in [x for x,_ in c.most_common(3)])
        c2=r2n.get((L,src_pred)) if src_pred else None
        dirstats[d]['predsrc_in_lex']+= (c2 is not None)
        if c2 is not None: dirstats[d]['predsrc_top1']+= (c2.most_common(1)[0][0]==gold)
        dirstats[d]['multitok']+= (nw>1)
    for (h,p,ci,nw) in rr:
        d='r'; gold=gapr[h]
        src_gold=A[ci]['native'] if ci>=0 else ''
        k=rt.index('<'+h+'>')
        src_pred=nt[k] if k<len(nt) and not hid_of(nt[k]) else ''
        dirstats[d]['total']+=1
        dirstats[d]['srcidx_correct']+= (src_pred==src_gold)
        c=n2r.get((L,src_gold))
        dirstats[d]['goldsrc_in_lex']+= (c is not None)
        if c is not None:
            dirstats[d]['goldsrc_top1']+= (c.most_common(1)[0][0]==gold)
            dirstats[d]['goldsrc_inlist']+= (gold in c)
            dirstats[d]['goldsrc_top3']+= (gold in [x for x,_ in c.most_common(3)])
        c2=n2r.get((L,src_pred)) if src_pred else None
        dirstats[d]['predsrc_in_lex']+= (c2 is not None)
        if c2 is not None: dirstats[d]['predsrc_top1']+= (c2.most_common(1)[0][0]==gold)
        dirstats[d]['multitok']+= (nw>1)
for d in 'nr':
    s=dirstats[d]; T=s['total']
    log('=== direction %s gaps (T=%d)'%(d,T))
    for k in ['srcidx_correct','multitok','goldsrc_in_lex','goldsrc_top1','goldsrc_top3','goldsrc_inlist','predsrc_in_lex','predsrc_top1']:
        log('   %-18s %6d  %.4f'%(k,s[k],s[k]/T))
