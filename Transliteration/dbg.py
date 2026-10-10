import pickle,os,json
from collections import Counter,defaultdict
D='C:/Users/Devansh Rajput/Downloads/Dakshina Dual-View Transliteration/'
exec(open(D+'trans.py').read().split('# ---------------- data')[0])
M=pickle.load(open(D+'models.pkl','rb'))
exec(open(D+'ana8.py').read().split("def eval_method")[0])
for d in 'nr':
    C1,C2,C3=M[d]
    print('=== dir',d)
    for (L,s,g) in cases[d][:12]:
        print('  %s | src=%s | gold=%s | pred=%s'%(L,s,g,decode(s,L,C1,C2,C3)))
    per=defaultdict(lambda:[0,0])
    for (L,s,g) in cases[d]:
        p=decode(s,L,C1,C2,C3); per[L][0]+=(p==g); per[L][1]+=1
    print('  per-lang:',{k:'%.3f(%d)'%(v[0]/v[1],v[1]) for k,v in sorted(per.items())})
