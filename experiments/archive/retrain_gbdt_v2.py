import os,numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from config import OUTPUT_DIR, SEED, N_FOLDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import sys; sys.path.insert(0, r'D:\Project\ThiefElectricity')
from dl_data import find_oof, best_f1

d=np.load(os.path.join(OUTPUT_DIR,'sgcc_preprocessed.npz'))
sf=np.nan_to_num(d['stat_features'],nan=0.0,posinf=0.0,neginf=0.0)
y=d['flags']; im=d['impute_mask']
mr=im.mean(axis=1).reshape(-1,1)
X=np.hstack([sf,mr]); X=np.clip(X,-1e6,1e6)
print(f'X: {X.shape}, range: [{np.min(X):.2f}, {np.max(X):.2f}]')

# Drop bad features (AUC<0.55 or constant)
meta_pkl=os.path.join(OUTPUT_DIR,'sgcc_meta.pkl')
import pickle
meta=pickle.load(open(meta_pkl,'rb'))
all_names=meta['stat_names']
# compute per-feature AUC to filter
from sklearn.metrics import roc_auc_score as au
feat_aucs=np.array([au(y,np.nan_to_num(X[:,i],nan=0)) for i in range(X.shape[1])])
good_idx=np.where((feat_aucs>0.53)|(feat_aucs<0.47))[0]
print(f'Keeping {len(good_idx)}/{X.shape[1]} features (AUC>0.53 or <0.47)')
X_filtered=X[:,good_idx]
if len(good_idx)<X.shape[1]:
    print(f'Dropped features:')
    for i in range(X.shape[1]):
        if i not in good_idx:
            print(f'  {all_names[i] if i<len(all_names) else f"feat{i}"}: AUC={feat_aucs[i]:.4f}')

skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
oof_ebm=np.zeros(len(y))

for fi,(ti,vi) in enumerate(skf.split(X,y)):
    Xt,Xv=X_filtered[ti],X_filtered[vi]; yt,yv=y[ti],y[vi]
    pw=(yt==0).sum()/max((yt==1).sum(),1)
    
    # QuantileTransform per fold
    qt=QuantileTransformer(output_distribution='normal',random_state=SEED+fi)
    Xt_qt=qt.fit_transform(Xt); Xv_qt=qt.transform(Xv)
    
    # XGBoost (best performer)
    m=xgb.XGBClassifier(
        n_estimators=3000,max_depth=8,learning_rate=0.02,
        subsample=0.7,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,
        min_child_weight=5,scale_pos_weight=pw,
        tree_method='hist',random_state=SEED+fi,verbosity=0
    )
    m.fit(Xt_qt,yt,eval_set=[(Xv_qt,yv)],verbose=False)
    oof_ebm[vi]=m.predict_proba(Xv_qt)[:,1]
    
    bf,bt=0,0.5
    for t in np.arange(.05,.95,.005):
        p=(oof_ebm[vi]>t).astype(int)
        if p.sum()==0: continue
        f=f1_score(yv,p,zero_division=0)
        if f>bf: bf,bt=f,t
    p=(oof_ebm[vi]>bt).astype(int)
    print(f'Fold {fi+1}: F1={bf:.4f} AUC={roc_auc_score(yv,oof_ebm[vi]):.4f} Rec={recall_score(yv,p):.4f}')

bf_o,bt_o=best_f1(y,oof_ebm)
p_o=(oof_ebm>bt_o).astype(int)
print(f'\nNEW GBDT (63feat+XGBoost+Quantile):')
print(f'  F1={bf_o:.4f} Rec={recall_score(y,p_o):.4f} Prec={precision_score(y,p_o):.4f} AUC={roc_auc_score(y,oof_ebm):.4f} th={bt_o:.3f}')

# Compare with V225
v225=find_oof('v225_results_20')
bf_v,_=best_f1(y,v225)
print(f'V225: F1={bf_v:.4f} AUC={roc_auc_score(y,v225):.4f}')

# Correlation between new OOF and V225
corr=np.corrcoef(oof_ebm, v225)[0,1]
print(f'Correlation(new, V225): {corr:.4f}')

# Ensemble
oof_avg=(oof_ebm+v225)/2
bf_a,_=best_f1(y,oof_avg)
p_a=(oof_avg>bt_o).astype(int) if bf_a>bf_o else p_o
print(f'Avg new+V225: F1={bf_a:.4f} AUC={roc_auc_score(y,oof_avg):.4f}')

np.savez_compressed(os.path.join(OUTPUT_DIR,'sgcc_gbdt_v2.npz'),
    oof_new=oof_ebm, oof_v225=v225, y=y)
print('\nSaved sgcc_gbdt_v2.npz')
