import os,numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from config import OUTPUT_DIR, SEED, N_FOLDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import xgboost as xgb
import sys; sys.path.insert(0, r'D:\Project\ThiefElectricity')
from dl_data import find_oof

d=np.load(os.path.join(OUTPUT_DIR,'sgcc_preprocessed.npz'))
sf=np.nan_to_num(d['stat_features'],nan=0.0,posinf=0.0,neginf=0.0)
y=d['flags']; im=d['impute_mask']; mr=im.mean(axis=1).reshape(-1,1)
X=np.hstack([sf,mr]); X=np.clip(X,-1e6,1e6)

v225=find_oof('v225_results_20')
THR_V225=0.74

# Define: error = 1 if V225 misclassifies
y_error=((v225>THR_V225).astype(int)!=y).astype(int)
print(f'V225 errors: {y_error.sum()}/{len(y)} ({y_error.mean()*100:.1f}%)')
print(f'  FP: {((v225>THR_V225)&(y==0)).sum()}, FN: {((v225<=THR_V225)&(y==1)).sum()}')
print(f'  Error AUC achievable: {roc_auc_score(y_error, abs(v225-0.5)):.4f} (using distance to 0.5)')

# Find V225 uncertainty region
uncertain=(v225>0.3)&(v225<0.7)
print(f'\nV225 uncertain (0.3<p<0.7): {uncertain.sum()} samples')
print(f'  Among these: error={y_error[uncertain].sum()}, correct={((~y_error)&uncertain).sum()}')
print(f'  Error rate in uncertain: {y_error[uncertain].mean()*100:.1f}%')

skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)

# Strategy: predict error, then correct V225 on high-error-probability samples
oof_error=np.zeros(len(y))
for fi,(ti,vi) in enumerate(skf.split(X,y)):
    Xt,Xv=X[ti],X[vi]; yt,yv=y_error[ti],y_error[vi]
    pw=(yt==0).sum()/max((yt==1).sum(),1)
    qt=QuantileTransformer(output_distribution='normal',random_state=SEED+fi)
    Xt_qt=qt.fit_transform(Xt); Xv_qt=qt.transform(Xv)
    m=xgb.XGBClassifier(n_estimators=1000,max_depth=5,learning_rate=0.03,
        scale_pos_weight=pw,subsample=0.8,tree_method='hist',random_state=SEED,verbosity=0)
    m.fit(Xt_qt,yt); oof_error[vi]=m.predict_proba(Xv_qt)[:,1]
    print(f'Fold {fi+1}: ErrorPred AUC={roc_auc_score(yv,oof_error[vi]):.4f}')

print(f'\nError prediction AUC: {roc_auc_score(y_error, oof_error):.4f}')

# Correction strategy: for samples where error_prob > threshold, flip V225
best_f1_corrected=0; best_ep_th=0.5; best_corr_th=THR_V225
for ep_th in np.arange(0.1,0.9,0.02):
    need_correction=oof_error>ep_th
    corrected=np.where(need_correction, 1-v225, v225)
    bf,_=0,0.5
    for t in np.arange(.05,.95,.005):
        p=(corrected>t).astype(int)
        if p.sum()==0: continue
        f=f1_score(y,p,zero_division=0)
        if f>bf: bf,_=f,t
    if bf>best_f1_corrected: best_f1_corrected=bf; best_ep_th=ep_th

print(f'\nV225 baseline: F1=0.8457')
print(f'After error correction:')
print(f'  Best error prob threshold: {best_ep_th:.2f} (correct {need_correction.sum()} samples)')

need_correction=oof_error>best_ep_th
corrected_v225=np.where(need_correction, 1.0-v225, v225)

bf_final,bt_final=0,0.5
for t in np.arange(.05,.95,.005):
    p=(corrected_v225>t).astype(int)
    if p.sum()==0: continue
    f=f1_score(y,p,zero_division=0)
    if f>bf_final: bf_final,bt_final=f,t
p_final=(corrected_v225>bt_final).astype(int)
tp=((p_final==1)&(y==1)).sum(); fp=((p_final==1)&(y==0)).sum()
fn=((p_final==0)&(y==1)).sum()

n_corrected=need_correction.sum()
n_correct_corrections=((need_correction)&(y_error==1)&((corrected_v225>bt_final)==y)).sum()
n_wrong_corrections=((need_correction)&(y_error==0)&((corrected_v225>bt_final)!=y)).sum()

print(f'  F1={bf_final:.4f} Rec={recall_score(y,p_final):.4f} Prec={precision_score(y,p_final):.4f}')
print(f'  AUC={roc_auc_score(y,corrected_v225):.4f}')
print(f'  Δ vs V225: {bf_final-0.8457:+.4f}')
print(f'  Corrected {n_corrected} samples: {n_correct_corrections} good, {n_wrong_corrections} bad')

# Weighted blending: V225_weighted = α*V225 + (1-α)*new_prediction*error_prob
# This is softer than flipping
best_f1_blend=0; best_alpha=0.5; best_bt=0.5
for alpha in np.arange(0.7,1.0,0.02):
    blended=v225*alpha + oof_error*(1-alpha)*np.where(v225>0.5, 0.2, 0.8)
    bf,bt=0,0.5
    for t in np.arange(.05,.95,.005):
        p=(blended>t).astype(int)
        if p.sum()==0: continue
        f=f1_score(y,p,zero_division=0)
        if f>bf: bf,bt=f,t
    if bf>best_f1_blend: best_f1_blend=bf; best_alpha=alpha; best_bt=bt

print(f'\nSoft blending (α*V225 + (1-α)*error_correction):')
print(f'  Best α={best_alpha:.2f}: F1={best_f1_blend:.4f}')
print(f'  Δ vs V225: {best_f1_blend-0.8457:+.4f}')

# Conclusion
print(f'\n{"="*50}')
if best_f1_blend>0.846:
    print('SUCCESS: Error correction improves V225!')
elif best_f1_corrected>0.846:
    print('SUCCESS: Hard correction improves V225!')
else:
    print('CONFIRMED: 63 features cannot improve V225 (Δ={:.4f})'.format(
        max(best_f1_blend,best_f1_corrected)-0.8457))
    print('V225 AUC=0.98 uses 100+ sophisticated features from data_engine.py (1900 lines)')
    print('Our 63 features AUC=0.84 - gap is in feature engineering depth')
    print('RECOMMEND: Copy data_engine.py feature pipeline from D:\\Project\\ThiefElectricity')
