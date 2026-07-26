import os,sys,numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
sys.path.insert(0,r'D:\Project\ThiefElectricity')
from data_engine import ElectricityTheftDataEngine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score
import lightgbm as lgb,pickle

OUT=os.path.join(os.path.dirname(__file__),'output')
os.makedirs(OUT,exist_ok=True)

print("Running data_engine pipeline...")
engine=ElectricityTheftDataEngine()
engine.load_data('data/raw_data.csv')
print(f'  Loaded: {engine.raw_data.shape}')

engine.handle_missing_values()
engine.handle_outliers()
engine.extract_deep_features()

X=engine.features_df.values.astype(np.float64)
X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
y=engine.flag
names=engine.feature_names

print(f'  Features: {X.shape}, names: {len(names)}')
print(f'  Range: [{X.min():.1f},{X.max():.1f}]')

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
oof=np.zeros(len(y))
for fi,(ti,vi) in enumerate(skf.split(X,y)):
    pw=(y[ti]==0).sum()//max((y[ti]==1).sum(),1)
    m=lgb.LGBMClassifier(n_estimators=1000,max_depth=8,learning_rate=0.03,num_leaves=63,
        subsample=0.8,colsample_bytree=0.8,reg_alpha=0.05,reg_lambda=0.05,
        min_child_samples=20,scale_pos_weight=pw,random_state=42,verbose=-1)
    m.fit(X[ti],y[ti],eval_set=[(X[vi],y[vi])],
          callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
    oof[vi]=m.predict_proba(X[vi])[:,1]
    bf,bt=0,0.5
    for t in np.arange(.05,.95,.005):
        p=(oof[vi]>t).astype(int)
        if p.sum()==0:continue
        f=f1_score(y[vi],p,zero_division=0)
        if f>bf:bf,bt=f,t
    p=(oof[vi]>bt).astype(int)
    print(f'Fold {fi+1}: F1={bf:.4f} AUC={roc_auc_score(y[vi],oof[vi]):.4f}')

bf,bt=0,0.5
for t in np.arange(.05,.95,.005):
    p=(oof>t).astype(int)
    if p.sum()==0:continue
    f=f1_score(y,p,zero_division=0)
    if f>bf:bf,bt=f,t
p=(oof>bt).astype(int)
print(f'\nOverall: F1={bf:.4f} Rec={recall_score(y,p):.4f} Prec={precision_score(y,p):.4f} AUC={roc_auc_score(y,oof):.4f}')

np.savez_compressed(os.path.join(OUT,'dengine_features.npz'),
    X=X, y=y, oof=oof, names=np.array(names,dtype=object))
pickle.dump({'names':names},open(os.path.join(OUT,'dengine_meta.pkl'),'wb'))
print(f'\nSaved dengine_features.npz')
