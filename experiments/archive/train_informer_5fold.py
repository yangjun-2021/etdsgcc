"""5-fold Informer training with checkpointing. Saves per-fold results."""
import os, time, glob, pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
import lightgbm as lgb

from utils import seed_everything, best_f1_score
from informer_model import train_informer, predict_informer

seed_everything(42)
t0 = time.time()

# ======== V108 preprocessing ========
print('V108 preprocessing...')
df = pd.read_csv('data/raw_data.csv')
dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
X_raw = df[dc].values.astype(np.float32)
y = df['FLAG'].values.astype(np.float32); del df
nmk = np.isnan(X_raw); Xf = np.nan_to_num(X_raw, nan=0.0)
Xl = np.log1p(np.maximum(Xf, 0)); sc = StandardScaler()
Xs = np.clip(sc.fit_transform(Xl).astype(np.float32), -5, 5)
X_seq = np.stack([Xs, nmk.astype(np.float32), (Xf == 0).astype(np.float32)], axis=1)
print('X_seq: (%d,%d,%d)' % (X_seq.shape[0], X_seq.shape[1], X_seq.shape[2]))

# ======== GBDT prior ========
print('Loading GBDT features + training prior...')
OD = r'D:\Project\ThiefElectricity\output'
d = np.load('output/sgcc_preprocessed.npz')
stat = np.nan_to_num(d['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
mr = d['impute_mask'].mean(axis=1).reshape(-1, 1)

v71 = np.load(sorted(glob.glob(f'{OD}/v71_oofs_*.npz'), reverse=True)[0], allow_pickle=True)
v71_oofs = np.column_stack([v71['lgb'], v71['xgb'], v71['cat'], v71['tcn'], v71['innov']])

ext_oofs = np.column_stack([
    np.load(sorted(glob.glob(f'{OD}/v213_results_*.npz'), reverse=True)[0], allow_pickle=True)['oof_v213'],
    np.load(sorted(glob.glob(f'{OD}/v219_results_*.npz'), reverse=True)[0], allow_pickle=True)['oof_final'],
    np.load(sorted(glob.glob(f'{OD}/v225_results_*.npz'), reverse=True)[0], allow_pickle=True)['oof_final'],
    np.load(sorted(glob.glob(f'{OD}/v216_results_*.npz'), reverse=True)[0], allow_pickle=True)['oof_final'],
])
our = np.load('output/tcn_kd_results.npz')
our_oofs = np.column_stack([our['oof_tcn_kd'], our['oof_stacker']])

X_gbdt = np.column_stack([stat, mr, v71_oofs, ext_oofs, our_oofs]).astype(np.float32)
X_gbdt = np.nan_to_num(X_gbdt, nan=0.0, posinf=0.0, neginf=0.0)

print('5-fold GBDT OOF (prior)...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n = len(y)
oof_prior = np.zeros(n)
for fi, (ti, vi) in enumerate(skf.split(X_gbdt, y)):
    pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
    m = lgb.LGBMClassifier(n_estimators=1000, max_depth=7, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pw, random_state=42, verbose=-1)
    m.fit(X_gbdt[ti], y[ti], eval_set=[(X_gbdt[vi], y[vi])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_prior[vi] = m.predict_proba(X_gbdt[vi])[:, 1]
    print('  Fold %d: AUC=%.4f' % (fi+1, roc_auc_score(y[vi], oof_prior[vi])))

# ======== Informer 5-fold ========
print('\nTraining Informer 5-fold...')
DEV = 'cuda'
oof_informer = np.zeros(n)

for fi, (ti, vi) in enumerate(skf.split(X_seq, y)):
    tf = time.time()
    torch.cuda.empty_cache()
    print('\nFold %d/%d...' % (fi + 1, 5))

    model = train_informer(
        X_seq[ti], y[ti], oof_prior=oof_prior[ti],
        d_model=64, n_heads=4, num_layers=2, dropout=0.3,
        epochs=40, batch_size=32, lr=3e-4,
        device=DEV, seed=42 + fi, verbose=True,
    )

    val_probs = predict_informer(model, X_seq[vi], oof_prior=oof_prior[vi], device=DEV)
    val_probs = np.nan_to_num(val_probs, nan=0.5)
    oof_informer[vi] = val_probs

    f1, th, rec, prec = best_f1_score(y[vi], val_probs)
    auc = roc_auc_score(y[vi], val_probs)
    print('Fold %d: F1=%.4f AUC=%.4f Rec=%.4f Prec=%.4f (%.0fs)' % (fi+1, f1, auc, rec, prec, time.time()-tf))

    # Per-fold checkpoint
    np.savez('output/informer_fold%d.npz' % (fi + 1), oof=val_probs, y=y[vi], auc=auc, f1=f1)

    del model
    torch.cuda.empty_cache()

# ======== Save complete OOF ========
f1_all, th_all, rec_all, prec_all = best_f1_score(y, oof_informer)
auc_all = roc_auc_score(y, oof_informer)
tp = ((oof_informer > th_all) & (y == 1)).sum()
fp = ((oof_informer > th_all) & (y == 0)).sum()
fn = ((oof_informer <= th_all) & (y == 1)).sum()

print('\n' + '=' * 60)
print('  Informer 5-Fold Results')
print('=' * 60)
print('  AUC: %.4f' % auc_all)
print('  F1:  %.4f (th=%.3f)' % (f1_all, th_all))
print('  Rec: %.4f  Prec: %.4f' % (rec_all, prec_all))
print('  TP=%d FP=%d FN=%d' % (tp, fp, fn))
print('')
print('  vs TCN+KD:    AUC=0.9783 F1=0.8433')
print('  vs SuperGBDT: AUC=0.9870 F1=0.8527')
print('  Delta TCN:    AUC=%+.4f F1=%+.4f' % (auc_all - 0.9783, f1_all - 0.8433))
print('  Time: %.1f min' % ((time.time() - t0) / 60))

np.savez('output/informer_oof.npz', oof_informer=oof_informer, y=y)
print('Saved to output/informer_oof.npz')
