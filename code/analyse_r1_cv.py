"""analyse_r1_cv.py -- summary statistics and paired tests for Table X (writes results/cv_summary.json).
Author: Md Tabrez"""
import json, os, numpy as np, sys
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
from sim.r1_clf import corrected_resampled_t, mcnemar_exact, mean_ci
R=json.load(open(os.path.join(RES, 'r1_clf_cv.json'))); y=np.array(R['y']); sev=np.array(R['sev'])
rows=R['rows']; names=list(rows[0]['acc'].keys())
from sklearn.metrics import precision_recall_fscore_support
print(len(rows))
summ={}
for n in names:
    a=[r['acc'][n] for r in rows]; f1=[]
    for r in rows:
        te=np.array(r['test_idx']); p=np.array(r['preds'][n]); f1.append(precision_recall_fscore_support(y[te],p,average='macro',zero_division=0)[2])
    m,s,lo,hi=mean_ci(a); summ[n]=dict(mean=m,sd=s,lo=lo,hi=hi,f1=float(np.mean(f1)),f1sd=float(np.std(f1,ddof=1)))
    print(f"{n:40s} {m*100:6.2f} ± {s*100:4.2f}  CI [{lo*100:.2f},{hi*100:.2f}]  F1 {np.mean(f1):.3f}")
ntr=np.mean([r['n_train']+r['n_val'] for r in rows]); nte=np.mean([r['n_test'] for r in rows])
ref='Student, distilled + INT8'
comps=[('Student, distilled','Student, no distillation'),('Teacher','Student, distilled + INT8'),('RandomForest','Student, distilled + INT8'),
('Student, distilled + INT8','Student, distilled'),('Student, no distillation','Student, no distillation, no physics'),('RandomForest','RandomForest, no physics'),('RandomForest','Teacher'),('GRU','Student, distilled + INT8'),('CNN-LSTM','Student, distilled + INT8')]
out=[]
for a,b in comps:
    d,t,p=corrected_resampled_t([r['acc'][a] for r in rows],[r['acc'][b] for r in rows],ntr,nte)
    # McNemar pooled per repeat (each episode once)
    mp=[]
    for rep in (0,1):
        yy=[];pa=[];pb=[]
        for r in rows:
            if r['rep']!=rep: continue
            te=r['test_idx']; yy+=list(y[te]); pa+=r['preds'][a]; pb+=r['preds'][b]
        mp.append(mcnemar_exact(np.array(yy),np.array(pa),np.array(pb)))
    print(f"{a:28s} vs {b:36s} d={d*100:+.2f} t={t:.2f} p={p:.3f}  McNemar rep0 {mp[0]} rep1 {mp[1]}")
    out.append(dict(a=a,b=b,diff=d,t=t,p=p,mcnemar=mp))
# severity analysis on rep0 OOF for proposed and RF
for n in ['Student, distilled + INT8','RandomForest']:
    yy=[];pp=[];ss=[]
    for r in rows:
        if r['rep']!=0: continue
        te=r['test_idx']; yy+=list(y[te]); pp+=r['preds'][n]; ss+=list(sev[te])
    yy=np.array(yy);pp=np.array(pp);ss=np.array(ss); f=yy>0
    q=np.quantile(ss[f],[1/3,2/3]); bins=np.digitize(ss,q)
    print(n,'normal',round((pp[~f]==0).mean()*100,1),[ (round(ss[f&(bins==b)].min(),3),round(ss[f&(bins==b)].max(),3), round((pp[f&(bins==b)]==yy[f&(bins==b)]).mean()*100,1)) for b in range(3)])
    print(' per-class', [round((pp[yy==c]==c).mean()*100,1) for c in range(6)])
info=[r['info'] for r in rows]; print({k:[i[k] for i in info] for k in info[0]})
json.dump(dict(summary=summ,tests=out),open(os.path.join(RES, 'cv_summary.json'),'w'),indent=1)
