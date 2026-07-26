"""Mega hill-climb over 16+ diverse OOFs — no training needed."""
import numpy as np, glob, time
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

OD = r'D:\Project\ThiefElectricity\output'

def load_npz(prefix, key):
    files = sorted(glob.glob(f'{OD}/{prefix}*.npz'), reverse=True)
    return np.load(files[0], allow_pickle=True)[key]

t0 = time.time()
y = np.load(glob.glob(f'{OD}/v225_results_*.npz')[0], allow_pickle=True)['y']

sources = {
    'v213': load_npz('v213_results_', 'oof_v213'),
    'v219': load_npz('v219_results_', 'oof_final'),
    'v225': load_npz('v225_results_', 'oof_final'),
    'v108_s': load_npz('v108_results_', 'oof_v108_stacked'),
    'v71_in': load_npz('v71_oofs_', 'innov'),
    'v216': load_npz('v216_results_', 'oof_final'),
    'v210': load_npz('v210_results_', 'oof_final'),
    'v234_co': load_npz('v234_results_', 'oof_combo'),
    'v229_iso': load_npz('v229_results_', 'oof_iso'),
    'v208_avg': load_npz('v208_results_', 'oof_avg'),
    'v212': load_npz('v212_results_', 'oof_v212'),
    'v72_s': load_npz('v72_oofs_', 'stack'),
    'v231': load_npz('v231_results_', 'oof_v231'),
}

our = np.load('output/tcn_kd_results.npz')
sources['ours_s'] = our['oof_stacker']
sources['ours_t'] = our['oof_tcn_kd']
sources['ours_b'] = our['oof_blend']

print(f'Loaded {len(sources)} OOFs in {(time.time()-t0):.0f}s')

# Per-OOF F1
for nm in sorted(sources.keys()):
    auc = roc_auc_score(y, sources[nm])
    bf = max((f1_score(y, (sources[nm] > th).astype(int), zero_division=0)
              for th in np.arange(0.05, 0.95, 0.005)), default=0)
    print(f'  {nm:12s}: AUC={auc:.4f} F1={bf:.4f}')

# Hill-climb
names = list(sources.keys())
n = len(names)
w = np.ones(n) / n

# Fast scoring function (coarse grid, vectorized)
TH_GRID = np.arange(0.05, 0.95, 0.005)  # coarse: 180 values
def score(wt):
    wt = np.maximum(wt, 0); wt = wt / wt.sum()
    p = np.zeros(len(y), dtype=np.float64)
    for i, nm in enumerate(names):
        if wt[i] > 0.001:
            p += wt[i] * sources[nm]
    bf = 0
    for th in TH_GRID:
        pred = p > th
        if not pred.any(): continue
        tp = ((pred) & (y == 1)).sum()
        precision_denom = pred.sum()
        recall_denom = (y == 1).sum()
        if precision_denom == 0 or recall_denom == 0: continue
        f = 2 * tp / (precision_denom + recall_denom)
        if f > bf: bf = f
    return bf

best_s = score(w)
best_w = w.copy()
print(f'Start (equal): F1={best_s:.4f}')

# Fewer iterations, larger steps
for it in range(500):
    improved = False
    for i in np.random.permutation(n):
        for d in [0.01, -0.01, 0.02, -0.02]:
            tw = best_w.copy(); tw[i] += d
            tw = np.maximum(tw, 0); tw = tw / tw.sum()
            s = score(tw)
            if s > best_s + 1e-6:
                best_s, best_w = s, tw.copy(); improved = True
    if it % 100 == 0:
        print(f'  HC iter {it}: F1={best_s:.4f}')
    if not improved:
        break

p = np.zeros(len(y))
for i, nm in enumerate(names):
    p += best_w[i] * sources[nm]

# Fine search at optimal threshold
bf, bt = 0, 0.5
for th in np.arange(max(0.01, bt - 0.05), min(0.99, bt + 0.05), 0.001):
    pred = p > th
    if not pred.any(): continue
    tp = (pred & (y == 1)).sum()
    denom_p = pred.sum()
    denom_r = (y == 1).sum()
    if denom_p == 0 or denom_r == 0: continue
    f = 2 * tp / (denom_p + denom_r)
    if f > bf: bf, bt = f, th

pred = p > bt
tp = ((pred == 1) & (y == 1)).sum()
fp = ((pred == 1) & (y == 0)).sum()
fn = ((pred == 0) & (y == 1)).sum()
rec = recall_score(y, pred)
prec = precision_score(y, pred)
auc = roc_auc_score(y, p)

print(f'\nMEGA HILL-CLIMB ({n} OOFs):')
print(f'  F1={bf:.4f}  AUC={auc:.4f}  Rec={rec:.4f}  Prec={prec:.4f}  th={bt:.3f}')
print(f'  TP={tp}  FP={fp}  FN={fn}')
print(f'  V225: F1=0.8457 AUC=0.9804 TP=2952 FP=414 FN=663')
print(f'  Delta vs V225: F1={bf-0.8457:+.4f}')
print('  Top weights:')
for i, nm in enumerate(names):
    if best_w[i] > 0.02:
        print(f'    {nm:12s}: {best_w[i]:.4f}')
print(f'  Time: {(time.time()-t0)/60:.1f} min')

np.savez('output/mega_hillclimb.npz', oof_final=p, weights=best_w, names=names, y=y)
print('Saved to output/mega_hillclimb.npz')
