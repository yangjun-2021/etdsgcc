import os,sys
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
sys.path.insert(0, r'D:\Project\ThiefElectricity')
import numpy as np
from config import OUTPUT_DIR, SEED, N_FOLDS
from dl_data import load_raw_data, prepare_sequences, find_oof, best_f1, evaluate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; print(f'Device: {DEVICE}')

X_raw,y=load_raw_data(); X_seq=prepare_sequences(X_raw)
print(f'X_seq: {X_seq.shape}, y: {len(y)}')

oof_v225=find_oof('v225_results_20')
bf1,bt1=best_f1(y,oof_v225);print(f'V225: F1={bf1:.4f} AUC={roc_auc_score(y,oof_v225):.4f}')

d=np.load(os.path.join(OUTPUT_DIR,'sgcc_preprocessed.npz'))
sf=np.nan_to_num(d['stat_features'],nan=0.0,posinf=0.0,neginf=0.0)
mr=d['impute_mask'].mean(axis=1).reshape(-1,1)
X_new=np.hstack([sf,mr]); X_new=np.clip(X_new,-1e6,1e6)
print(f'New features: {X_new.shape}')

skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)

# Expert C: GBDT on new features only
print('\n--- Expert C: GBDT(new 54 features) ---')
oof_new=np.zeros(len(y))
for fi,(ti,vi) in enumerate(skf.split(X_new,y)):
    pw=(y[ti]==0).sum()/max((y[ti]==1).sum(),1)
    m=lgb.LGBMClassifier(n_estimators=2000,max_depth=7,learning_rate=0.03,num_leaves=63,
        subsample=0.8,colsample_bytree=0.8,scale_pos_weight=pw,random_state=42,verbose=-1)
    m.fit(X_new[ti],y[ti],eval_set=[(X_new[vi],y[vi])],
          callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(0)])
    oof_new[vi]=m.predict_proba(X_new[vi])[:,1]
bf_n=best_f1(y,oof_new)[0]; print(f'New GBDT: F1={bf_n:.4f} AUC={roc_auc_score(y,oof_new):.4f}')

# TCN with V225 prior
class TCN_Res(nn.Module):
    def __init__(self,in_ch=3,h=40):
        super().__init__()
        self.tcn=nn.Sequential(
            nn.Conv1d(in_ch,h,7,padding=3),nn.BatchNorm1d(h),nn.GELU(),
            nn.Conv1d(h,h,7,padding=6,dilation=2),nn.BatchNorm1d(h),nn.GELU(),
            nn.Conv1d(h,h,7,padding=12,dilation=4),nn.BatchNorm1d(h),nn.GELU(),
            nn.Conv1d(h,h,7,padding=24,dilation=8),nn.BatchNorm1d(h),nn.GELU(),
            nn.Conv1d(h,h,7,padding=48,dilation=16),nn.BatchNorm1d(h),nn.GELU(),
        )
        self.head=nn.Sequential(nn.Linear(h+1,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(self,x,prior=None):
        x=self.tcn(x);x=torch.mean(x,dim=2)
        if prior is not None: x=torch.cat([x,prior.reshape(-1,1)],dim=1)
        return self.head(x).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self,a=0.75,g=2.0):super().__init__();self.a,self.g=a,g
    def forward(self,lo,t):
        p=torch.sigmoid(lo).clamp(1e-6,1-1e-6)
        return (-self.a*(1-p)**self.g*torch.log(p)*t-(1-self.a)*p**self.g*torch.log(1-p)*(1-t)).mean()

def pred_b(model,Xn,pn,bs=256):
    model.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(Xn),bs):
            xb=torch.FloatTensor(Xn[i:i+bs]).to(DEVICE)
            pb=torch.FloatTensor(pn[i:i+bs]).to(DEVICE) if pn is not None else None
            out.append(torch.sigmoid(model(xb,pb)).cpu().numpy())
    return np.concatenate(out)

print('\n--- Expert D: TCN(V225 prior) ---')
oof_tcn=np.zeros(len(y))
for fi,(ti,vi) in enumerate(skf.split(np.zeros(len(y)),y)):
    np.random.seed(SEED+fi);torch.manual_seed(SEED+fi)
    model=TCN_Res(in_ch=3,h=40).to(DEVICE);crit=FocalLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25,eta_min=1e-6)
    ds=TensorDataset(torch.FloatTensor(X_seq[ti]),torch.FloatTensor(oof_v225[ti]).reshape(-1,1),torch.FloatTensor(y[ti]))
    dl=DataLoader(ds,batch_size=64,shuffle=True,drop_last=True)
    bf,bs_=0,None;pat=0
    for ep in range(25):
        model.train()
        for bx,bp,by in dl:
            bx,bp,by=bx.to(DEVICE),bp.to(DEVICE),by.to(DEVICE)
            opt.zero_grad();loss=crit(model(bx,bp),by)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        sch.step();vp=pred_b(model,X_seq[vi],oof_v225[vi].reshape(-1,1))
        bf_val=max((f1_score(y[vi],(vp>t).astype(int),zero_division=0) for t in np.arange(.2,.8,.01)),default=0)
        if bf_val>bf: bf=bf_val;bs_=model.state_dict().copy();pat=0
        else: pat+=1
        if pat>=5: break
    model.load_state_dict(bs_);oof_tcn[vi]=pred_b(model,X_seq[vi],oof_v225[vi].reshape(-1,1))
    print(f'Fold {fi+1}: F1={bf:.4f}')
bf_t=best_f1(y,oof_tcn)[0]; print(f'TCN: F1={bf_t:.4f} AUC={roc_auc_score(y,oof_tcn):.4f}')

# MEGA stack: V225 + new GBDT + TCN
print('\n--- META LEARNER: XGBoost Stacking ---')
X_meta=np.column_stack([
    oof_v225.reshape(-1,1), oof_new.reshape(-1,1), oof_tcn.reshape(-1,1),
    np.abs(oof_v225-oof_new).reshape(-1,1),
    np.abs(oof_v225-oof_tcn).reshape(-1,1),
    np.abs(oof_new-oof_tcn).reshape(-1,1),
])

oof_meta=np.zeros(len(y))
for fi,(ti,vi) in enumerate(skf.split(np.zeros(len(y)),y)):
    pw=(y[ti]==0).sum()/max((y[ti]==1).sum(),1)
    m=xgb.XGBClassifier(n_estimators=500,max_depth=4,learning_rate=0.03,
        scale_pos_weight=pw,subsample=0.8,tree_method='hist',random_state=SEED,verbosity=0)
    m.fit(X_meta[ti],y[ti]); oof_meta[vi]=m.predict_proba(X_meta[vi])[:,1]

bf_m,bt_m=best_f1(y,oof_meta)
p_m=(oof_meta>bt_m).astype(int)
print(f'META: F1={bf_m:.4f} Rec={recall_score(y,p_m):.4f} Prec={precision_score(y,p_m):.4f} AUC={roc_auc_score(y,oof_meta):.4f}')

# Also add raw features to meta
X_meta2=np.column_stack([X_meta,X_new])
oof_meta2=np.zeros(len(y))
for fi,(ti,vi) in enumerate(skf.split(np.zeros(len(y)),y)):
    pw=(y[ti]==0).sum()/max((y[ti]==1).sum(),1)
    m=xgb.XGBClassifier(n_estimators=500,max_depth=4,learning_rate=0.03,
        scale_pos_weight=pw,subsample=0.8,tree_method='hist',random_state=SEED,verbosity=0)
    m.fit(X_meta2[ti],y[ti]); oof_meta2[vi]=m.predict_proba(X_meta2[vi])[:,1]

bf_m2,bt_m2=best_f1(y,oof_meta2)
p_m2=(oof_meta2>bt_m2).astype(int)
print(f'META+Features: F1={bf_m2:.4f} Rec={recall_score(y,p_m2):.4f} Prec={precision_score(y,p_m2):.4f} AUC={roc_auc_score(y,oof_meta2):.4f}')

# Hill-climb over all 4 sources
sources=[oof_v225,oof_new,oof_tcn,oof_meta2]
names=['V225','NewGBDT','TCN','Meta2']
n_s=len(sources); w=np.ones(n_s)/n_s
def score(wt):
    wt=np.maximum(wt,0);wt=wt/wt.sum()
    prob=np.zeros(len(y))
    for i,s in enumerate(sources): prob+=wt[i]*s
    bf,_=best_f1(y,prob); return bf

best_w,best_s=w.copy(),score(w)
for it in range(2000):
    improved=False
    for i in np.random.permutation(n_s):
        for d in [.005,-.005,.015,-.015,.03,-.03]:
            tw=best_w.copy();tw[i]+=d;tw=np.maximum(tw,0);tw=tw/tw.sum()
            s=score(tw)
            if s>best_s: best_s=s;best_w=tw.copy();improved=True
    if it%200==0: print(f'HC iter {it}: {best_s:.4f}')
    if not improved: break
prob_final=np.zeros(len(y))
for i,s in enumerate(sources): prob_final+=best_w[i]*s
bf_f,bt_f=best_f1(y,prob_final)
p_f=(prob_final>bt_f).astype(int)
print(f'\nFINAL HILL-CLIMB: F1={bf_f:.4f} Rec={recall_score(y,p_f):.4f} Prec={precision_score(y,p_f):.4f} AUC={roc_auc_score(y,prob_final):.4f}')
tp=((p_f==1)&(y==1)).sum();fp=((p_f==1)&(y==0)).sum();fn=((p_f==0)&(y==1)).sum();tn=((p_f==0)&(y==0)).sum()
print(f'TP/FP/FN/TN: {tp}/{fp}/{fn}/{tn}')
print(f'Weights: ',{n:f'{w:.4f}' for n,w in zip(names,best_w) if w>.01})

# Recall-constrained
print(f'\nRecall>=0.90:')
rc_f1,rc_t=0,.5
for t in np.arange(.02,.95,.002):
    p=(prob_final>t).astype(int); r=recall_score(y,p,zero_division=0)
    if r<0.90: continue
    f=f1_score(y,p,zero_division=0)
    if f>rc_f1: rc_f1,rc_t=f,t
if rc_f1>0:
    p=(prob_final>rc_t).astype(int)
    print(f'  F1={rc_f1:.4f} Rec={recall_score(y,p):.4f} Prec={precision_score(y,p):.4f} th={rc_t:.3f}')
else:
    mr=max(recall_score(y,(prob_final>t).astype(int)) for t in np.arange(.01,.99,.002))
    print(f'  IMPOSSIBLE. Max recall={mr:.4f}')

np.savez_compressed(os.path.join(OUTPUT_DIR,'final_fusion.npz'),
    prob_final=prob_final,oof_v225=oof_v225,oof_new=oof_new,oof_tcn=oof_tcn,oof_meta=oof_meta2,y=y)
print('\nSaved final_fusion.npz')