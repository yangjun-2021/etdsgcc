import os,sys,numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
sys.path.insert(0,r'D:\Project\ThiefElectricity')
from config import OUTPUT_DIR,SEED,N_FOLDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import xgboost as xgb
import glob

# Load all OOFs  
OOF_PATH=r'D:\Project\ThiefElectricity\output'
from dl_data import find_oof, best_f1
v225=find_oof('v225_results_20')
y=np.load(os.path.join(OOF_PATH,'v225_results_20260609_003553.npz'),allow_pickle=True)['y']

oofs={}
for f in sorted(glob.glob(f'{OOF_PATH}/v*_results*.npz')):
    try:
        d=np.load(f,allow_pickle=True); o=d.get('oof_final')
        if o is not None and len(o)==len(y): oofs[f.split('v')[-1].split('_')[0]]=o
    except: pass
oof_names=sorted(oofs.keys(),key=lambda k: best_f1(y,oofs[k])[0],reverse=True)
print(f'OOF sources: {len(oofs)}')
for nm in oof_names:
    bf,bt=best_f1(y,oofs[nm]); p=(oofs[nm]>bt).astype(int)
    print(f'  V{nm}: F1={bf:.4f} AUC={roc_auc_score(y,oofs[nm]):.4f} Rec={recall_score(y,p):.4f}')

# Load our 63 features
d=np.load(os.path.join(OUTPUT_DIR,'sgcc_preprocessed.npz'))
sf=np.nan_to_num(d['stat_features'],nan=0.0,posinf=0.0,neginf=0.0)
mr=d['impute_mask'].mean(axis=1).reshape(-1,1)
X_new=np.hstack([sf,mr]); X_new=np.clip(X_new,-1e6,1e6)

# Build mega feature matrix: all OOFs + all OOF diffs + new features
print(f'\nBuilding mega feature matrix...')
feat_list=[oofs[nm].reshape(-1,1) for nm in oof_names]

# OOF pairwise diffs (model disagreement)
for i in range(min(5,len(oof_names))):
    for j in range(i+1,min(5,len(oof_names))):
        feat_list.append(np.abs(oofs[oof_names[i]]-oofs[oof_names[j]]).reshape(-1,1))

# OOF vs mean disagreement
mean_oof=np.column_stack([feat_list[i] for i in range(len(oof_names))]).mean(axis=1)
for nm in oof_names:
    feat_list.append(np.abs(oofs[nm]-mean_oof).reshape(-1,1))

X_mega=np.column_stack(feat_list)
print(f'OOF features: {X_mega.shape}')

skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)

# Meta learner: XGBoost on OOF features only
print('\n--- Meta(OOF only) ---')
oof_meta1=np.zeros(len(y))
for fi,(ti,vi) in enumerate(skf.split(np.zeros(len(y)),y)):
    pw=(y[ti]==0).sum()/max((y[ti]==1).sum(),1)
    m=xgb.XGBClassifier(n_estimators=500,max_depth=4,learning_rate=0.03,
        scale_pos_weight=pw,subsample=0.8,tree_method='hist',random_state=SEED,verbosity=0)
    m.fit(X_mega[ti],y[ti]); oof_meta1[vi]=m.predict_proba(X_mega[vi])[:,1]
    bf,bt=best_f1(y[vi],oof_meta1[vi])
    print(f'Fold {fi+1}: F1={bf:.4f} AUC={roc_auc_score(y[vi],oof_meta1[vi]):.4f}')

bf1,bt1=best_f1(y,oof_meta1); p1=(oof_meta1>bt1).astype(int)
print(f'Meta(OOF): F1={bf1:.4f} Rec={recall_score(y,p1):.4f} Prec={precision_score(y,p1):.4f} AUC={roc_auc_score(y,oof_meta1):.4f}')

# Meta + new features (Quantile transformed)
print('\n--- Meta(OOF + 63 new feats) ---')
X_full=np.column_stack([X_mega,X_new])
oof_meta2=np.zeros(len(y))
for fi,(ti,vi) in enumerate(skf.split(np.zeros(len(y)),y)):
    pw=(y[ti]==0).sum()/max((y[ti]==1).sum(),1)
    qt=QuantileTransformer(output_distribution='normal',random_state=SEED+fi)
    Xt=np.column_stack([X_mega[ti],qt.fit_transform(X_new[ti])])
    Xv=np.column_stack([X_mega[vi],qt.transform(X_new[vi])])
    m=xgb.XGBClassifier(n_estimators=500,max_depth=4,learning_rate=0.03,
        scale_pos_weight=pw,subsample=0.8,tree_method='hist',random_state=SEED,verbosity=0)
    m.fit(Xt,y[ti]); oof_meta2[vi]=m.predict_proba(Xv)[:,1]
    bf,bt=best_f1(y[vi],oof_meta2[vi])
    print(f'Fold {fi+1}: F1={bf:.4f} AUC={roc_auc_score(y[vi],oof_meta2[vi]):.4f}')

bf2,bt2=best_f1(y,oof_meta2); p2=(oof_meta2>bt2).astype(int)
print(f'Meta(OOF+63feat): F1={bf2:.4f} Rec={recall_score(y,p2):.4f} Prec={precision_score(y,p2):.4f} AUC={roc_auc_score(y,oof_meta2):.4f}')

# Hill-climb of all: V225 + OOF_meta1 + OOF_meta2
sources=[v225, oof_meta1, oof_meta2]; snames=['V225','Meta_OOF','Meta_OOF+63']
def score(wt):
    wt=np.maximum(wt,0);wt=wt/wt.sum()
    p=np.zeros(len(y))
    for i,s in enumerate(sources): p+=wt[i]*s
    bf,_=best_f1(y,p); return bf

w=np.ones(3)/3; bs=score(w)
for it in range(500):
    imp=False
    for i in range(3):
        for d in [.005,-.005,.015,-.015,.03,-.03]:
            tw=w.copy();tw[i]+=d;tw=np.maximum(tw,0);tw=tw/tw.sum()
            s=score(tw)
            if s>bs+1e-6: bs=s;w=tw.copy();imp=True
    if not imp: break

p=np.zeros(len(y))
for i,s in enumerate(sources): p+=w[i]*s
bf,bt=best_f1(y,p); pf=(p>bt).astype(int)
tp=((pf==1)&(y==1)).sum();fp=((pf==1)&(y==0)).sum();fn=((pf==0)&(y==1)).sum()
print(f'\nFINAL HILL-CLIMB: F1={bf:.4f} Rec={recall_score(y,pf):.4f} Prec={precision_score(y,pf):.4f} AUC={roc_auc_score(y,p):.4f}')
print(f'TP/FP/FN: {tp}/{fp}/{fn}')
print(f'Weights: {dict(zip(snames,[f"{x:.4f}" for x in w]))}')

# Compare vs V225
delta=bf-0.8457
print(f'\nDelta vs V225: {delta:+.4f}')
if delta>0: print('*** IMPROVED! ***')
elif delta>-.001: print('Essentially unchanged')
else: print('V225 still leads')
