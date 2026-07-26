import os,numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from config import OUTPUT_DIR, SEED, N_FOLDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import pickle

d=np.load(os.path.join(OUTPUT_DIR,'sgcc_preprocessed.npz'))
X_raw=d['stat_features'].copy()
y=d['flags'].copy()
im=d['impute_mask']

miss_ratio=im.mean(axis=1).reshape(-1,1)
X=np.hstack([X_raw,miss_ratio])
X=np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X=np.clip(X, -1e6, 1e6)

print(f'X: {X.shape}, y: {len(y)}, theft: {y.sum()}/{len(y)}')
print(f'NaN in X: {np.isnan(X).sum()}, Inf: {np.isinf(X).sum()}')
print(f'Range: [{np.min(X):.2f}, {np.max(X):.2f}]')

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
oof_lgb=np.zeros(len(y)); oof_xgb=np.zeros(len(y)); oof_cb=np.zeros(len(y))

for fi,(ti,vi) in enumerate(skf.split(X,y)):
    Xt,Xv=X[ti],X[vi]; yt,yv=y[ti],y[vi]
    pw=(yt==0).sum()/max((yt==1).sum(),1)
    print(f'\nFold {fi+1}: train={len(ti)}, val={len(vi)}, pw={pw:.2f}')
    
    m1=lgb.LGBMClassifier(
        n_estimators=2000, max_depth=7, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.05,
        min_child_samples=20, scale_pos_weight=pw, random_state=42, verbose=-1
    )
    m1.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
    oof_lgb[vi]=m1.predict_proba(Xv)[:,1]
    p=(oof_lgb[vi]>0.5).astype(int)
    print(f'  LGB: F1={f1_score(yv,p):.4f} AUC={roc_auc_score(yv,oof_lgb[vi]):.4f} actual_trees={m1.n_estimators_}')

    m2=xgb.XGBClassifier(
        n_estimators=2000, max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pw, tree_method='hist', random_state=42, verbosity=0
    )
    m2.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False)
    oof_xgb[vi]=m2.predict_proba(Xv)[:,1]

    m3=CatBoostClassifier(iterations=2000,depth=7,learning_rate=0.03,l2_leaf_reg=3.0,
                          class_weights={0:1.0,1:pw},random_seed=42,verbose=0)
    m3.fit(Xt,yt,eval_set=(Xv,yv),early_stopping_rounds=100)
    oof_cb[vi]=m3.predict_proba(Xv)[:,1]

print('\n' + '='*50)
print('Individual Models:')
for name, oof in [('LightGBM',oof_lgb),('XGBoost',oof_xgb),('CatBoost',oof_cb)]:
    bf,bt=0,0.5
    for t in np.arange(0.05,0.95,0.005):
        p=(oof>t).astype(int)
        if p.sum()==0: continue
        f=f1_score(y,p,zero_division=0)
        if f>bf: bf,bt=f,t
    p=(oof>bt).astype(int)
    print(f'  {name:>10s}: F1={bf:.4f} Rec={recall_score(y,p):.4f} Prec={precision_score(y,p):.4f} AUC={roc_auc_score(y,oof):.4f} th={bt:.3f}')

print('\nEnsemble search...')
for w_lgb in [0.3,0.4,0.5,0.6,0.7]:
    for w_xgb in [0.1,0.2,0.25,0.3]:
        w_cb=1.0-w_lgb-w_xgb
        if w_cb<0.1: continue
        prob=w_lgb*oof_lgb+w_xgb*oof_xgb+w_cb*oof_cb
        bf,bt=0,0.5
        for t in np.arange(0.05,0.95,0.005):
            p=(prob>t).astype(int)
            if p.sum()==0: continue
            f=f1_score(y,p,zero_division=0)
            if f>bf: bf,bt=f,t
        p=(prob>bt).astype(int)
        if bf>0.84:
            print(f'  w={w_lgb:.1f}/{w_xgb:.1f}/{w_cb:.1f}: F1={bf:.4f} Rec={recall_score(y,p):.4f} AUC={roc_auc_score(y,prob):.4f}')

np.savez_compressed(os.path.join(OUTPUT_DIR,'sgcc_gbdt_new.npz'),
                    oof_lgb=oof_lgb,oof_xgb=oof_xgb,oof_cb=oof_cb,y=y)
print('\nSaved sgcc_gbdt_new.npz')
