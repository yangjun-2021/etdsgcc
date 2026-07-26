import os, numpy as np, glob
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from sklearn.metrics import *

OD=r'D:\Project\ThiefElectricity\output'
y=np.load(os.path.join(OD,'v225_results_20260609_003553.npz'))['y']
oofs={}
for f in sorted(glob.glob(os.path.join(OD,'v*_results*.npz'))):
 try:
  d=np.load(f,allow_pickle=True); o=d.get('oof_final')
  if o is not None and len(o)==len(y): oofs[f.split('v')[-1].split('_')[0]]=o
 except: pass

names=sorted(oofs.keys())
print(f'OOF sources: {len(oofs)}\n')
for k in sorted(names,key=lambda x:-float(x) if x.isdigit() else 0):
 bf,bt=0,.5
 for t in np.arange(.05,.95,.005):
  p=(oofs[k]>t).astype(int)
  if p.sum()==0: continue
  f=f1_score(y,p,zero_division=0)
  if f>bf: bf,bt=f,t
 pred=(oofs[k]>bt).astype(int)
 print(f"V{k}: F1={bf:.4f} Rec={recall_score(y,pred):.4f} Prec={precision_score(y,pred):.4f} AUC={roc_auc_score(y,oofs[k]):.4f}")

eq=np.mean([oofs[k] for k in names],axis=0)
bf,bt=0,.5
for t in np.arange(.05,.95,.005):
 p=(eq>t).astype(int)
 if p.sum()==0: continue
 f=f1_score(y,p,zero_division=0)
 if f>bf: bf,bt=f,t
pred=(eq>bt).astype(int)
print(f'\nEqual avg (all 7): F1={bf:.4f} Rec={recall_score(y,pred):.4f} Prec={precision_score(y,pred):.4f} AUC={roc_auc_score(y,eq):.4f}')

n=len(names); w=np.ones(n)/n
def score(wt):
 wt=np.maximum(wt,0); wt=wt/wt.sum()
 prob=np.zeros(len(y))
 for i,nm in enumerate(names): prob+=wt[i]*oofs[nm]
 bf,_=0,.5
 for t in np.arange(.05,.95,.005):
  p=(prob>t).astype(int)
  if p.sum()==0: continue
  f=f1_score(y,p,zero_division=0)
  if f>bf: bf,_=f,t
 return bf

best_w=w.copy(); best_s=score(best_w)
for it in range(2000):
 improved=False
 for i in np.random.permutation(n):
  for d in [.003,-.003,.01,-.01,.03,-.03]:
   tw=best_w.copy(); tw[i]+=d; tw=np.maximum(tw,0); tw=tw/tw.sum()
   s=score(tw)
   if s>best_s: best_s=s; best_w=tw.copy(); improved=True
 if it%200==0: print(f'HC iter {it}: F1={best_s:.4f}')
 if not improved: break

print(f'\nHill-climb final: F1={best_s:.4f}')
for i,nm in enumerate(names):
 if best_w[i]>.01: print(f'  {nm}: {best_w[i]:.4f}')

prob=np.zeros(len(y))
for i,nm in enumerate(names): prob+=best_w[i]*oofs[nm]

print(f'\nRecall-constrained search:')
for tr in [0.90, 0.905, 0.91]:
 rc_f1,rc_th=0,.5
 for t in np.arange(.02,.95,.002):
  p=(prob>t).astype(int)
  r=recall_score(y,p,zero_division=0)
  if r<tr: continue
  f=f1_score(y,p,zero_division=0)
  if f>rc_f1: rc_f1,rc_th=f,t
 if rc_f1>0:
  p=(prob>rc_th).astype(int)
  print(f'  Recall>={tr}: F1={rc_f1:.4f} Rec={recall_score(y,p):.4f} Prec={precision_score(y,p):.4f} th={rc_th:.3f}')
 else:
  mr=max(recall_score(y,(prob>t).astype(int)) for t in np.arange(.01,.99,.002))
  print(f'  Recall>={tr}: IMPOSSIBLE (max recall={mr:.4f})')

print(f'\nTheoretical F1 ceiling at each recall level:')
pos_probs=prob[y==1]; neg_probs=prob[y==0]
sorted_pos=np.sort(pos_probs)[::-1]
for tr in [0.80,0.83,0.85,0.88,0.90,0.92,0.95]:
 k=int(len(pos_probs)*tr)
 th=sorted_pos[min(k,len(sorted_pos)-1)] if k>0 else 0
 fn=len(pos_probs)-k; fp=int((neg_probs>th).sum())
 f1=2*k/(2*k+fp+fn) if (2*k+fp+fn)>0 else 0
 prec=k/(k+fp) if (k+fp)>0 else 0
 print(f'  Rec={tr:.2f}: max F1={f1:.4f} Prec={prec:.4f} TP={k} FP={fp} FN={fn}')

print(f'\nV225 baseline:')
tp=((prob>.74)&(y==1)).sum(); fp=((prob>.74)&(y==0)).sum(); fn=((prob<=.74)&(y==1)).sum()
print(f'  TP={tp} FP={fp} FN={fn} F1={2*tp/(2*tp+fp+fn):.4f}')

gap_to_90 = int(len(pos_probs)*0.90) - tp
print(f'\nTo reach F1>=0.90 with Recall>=0.90: need to reclassify {gap_to_90} FN as TP without adding FP')
print(f'Current FN: {fn}. Need: {(y==1).sum()*0.9:.0f} TP, current best: {tp}')
