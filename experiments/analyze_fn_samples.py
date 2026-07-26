"""Deep analysis of current FN samples."""
import os, sys
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR

meta = np.load(os.path.join(OUTPUT_DIR, 'sgcc_mega_meta.npz'))['oof_final']
y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
X_seq = pre['X_seq']
mask = pre['impute_mask']

best_th = 0.5
best_f1 = 0
for th in np.arange(0.05, 0.95, 0.005):
    pred = (meta > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > best_f1: best_f1, best_th = f, th
pred = (meta > best_th).astype(int)

fn_idx = np.where((pred == 0) & (y == 1))[0]
fp_idx = np.where((pred == 1) & (y == 0))[0]
tp_idx = np.where((pred == 1) & (y == 1))[0]

# Load strong OOFs
oofs = {}
for f,k in [
    ('autoresearch_best.npz','oof_final'),
    ('mega_boost_enhanced.npz','oof_final'),
    ('informer_oof.npz','oof_informer'),
    ('amst_3ch_recall10_oof.npz','oof_amst_3ch_recall10'),
    ('patch_transformer_raw_3ch_oof.npz','oof_patch_transformer_raw_3ch'),
    ('supcon_raw_3ch_oof.npz','oof_supcon_raw_3ch'),
    ('strong_gbdt_prior_oof.npz','oof_strong_gbdt_prior'),
]:
    try:
        oofs[f] = np.load(os.path.join(OUTPUT_DIR,f))[k]
    except Exception as e:
        pass

print('=== FN score statistics across strong OOFs ===')
print('OOF                           mean    median   q90     max')
for name, oof in oofs.items():
    scores = oof[fn_idx]
    print(f'{name:30s} {scores.mean():.3f}   {np.median(scores):.3f}   {np.percentile(scores,90):.3f}   {scores.max():.3f}')

print('\n=== How many FNs are scored >0.5 by each OOF? ===')
for name, oof in oofs.items():
    n = (oof[fn_idx] > 0.5).sum()
    print(f'{name:30s} {n:4d}/{len(fn_idx)} ({n/len(fn_idx)*100:.1f}%)')

print('\n=== How many FNs are scored >0.5 by ANY strong DL OOF? ===')
dl_oofs = ['informer_oof.npz', 'amst_3ch_recall10_oof.npz', 'patch_transformer_raw_3ch_oof.npz', 'supcon_raw_3ch_oof.npz']
any_dl = np.zeros(len(fn_idx), dtype=bool)
for name in dl_oofs:
    if name in oofs:
        any_dl |= (oofs[name][fn_idx] > 0.5)
print(f'Any DL >0.5: {any_dl.sum()}/{len(fn_idx)} ({any_dl.sum()/len(fn_idx)*100:.1f}%)')

# Most "rescuable" FNs: those where at least one DL model scores high
print('\n=== Rescuable FNs (DL score >0.7) ===')
for name in dl_oofs:
    if name in oofs:
        n = (oofs[name][fn_idx] > 0.7).sum()
        print(f'{name:30s} {n:4d}/{len(fn_idx)} ({n/len(fn_idx)*100:.1f}%)')

# Identify samples where all models fail
all_low = np.ones(len(fn_idx), dtype=bool)
for name in dl_oofs:
    if name in oofs:
        all_low &= (oofs[name][fn_idx] <= 0.5)
print(f'\nAll DL <=0.5: {all_low.sum()}/{len(fn_idx)} ({all_low.sum()/len(fn_idx)*100:.1f}%)')

# Feature stats for all-low FNs
val = X_seq[:,0,:].astype(np.float64)
obs_mask = ~mask
mr = mask.mean(axis=1)
clean = np.where(obs_mask, val, np.nan)
mean_cons = np.nan_to_num(np.nanmean(clean, axis=1))
print('\n=== All-low FN features ===')
print(f'missing_ratio: {mr[fn_idx][all_low].mean():.3f}')
print(f'mean_cons: {mean_cons[fn_idx][all_low].mean():.3f}')
print(f'std_cons: {np.nan_to_num(np.nanstd(clean,axis=1))[fn_idx][all_low].mean():.3f}')
