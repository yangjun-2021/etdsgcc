"""Optimize a weighted ensemble of top OOF signals for F1 on original labels."""
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR

# Load top OOF signals identified earlier
oof_files = [
    ('hillclimb_best_oof.npz', 'oof_hillclimb'),
    ('sgcc_mega_meta.npz', 'oof_final'),
    ('sgcc_mega_meta_xgb_d3_1000.npz', 'oof_final'),
    ('autoresearch_best.npz', 'oof_final'),
    ('mega_boost_enhanced.npz', 'oof_final'),
    ('mega_boost_final.npz', 'oof_final'),
    ('feature_rich_meta_oof.npz', 'oof_feature_rich_meta'),
    ('final_blend.npz', 'oof_final'),
    ('smart_blend.npz', 'oof_final'),
    ('mega_boost_informer.npz', 'oof_final'),
    ('ultimate.npz', 'oof_final'),
    ('nn_meta_oof.npz', 'oof_nn_meta'),
    ('informer_oof.npz', 'oof_informer'),
    ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10'),
    ('strong_gbdt_prior_oof.npz', 'oof_strong_gbdt_prior'),
    ('informer_3ch_strong_prior_oof.npz', 'oof_informer_3ch_strong_prior'),
    ('amst_3ch_strong_prior_oof.npz', 'oof_amst_3ch_strong_prior'),
    ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch'),
    ('patch_transformer_raw_3ch_oof.npz', 'oof_patch_transformer_raw_3ch'),
]

y = None
oofs = {}
for fname, key in oof_files:
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        continue
    try:
        d = np.load(path)
        if y is None:
            for k in d.keys():
                if k in ('y', 'flags', 'labels'):
                    y = d[k]
                    break
        if key in d.files and d[key].ndim == 1:
            oofs[fname.replace('.npz', '')] = d[key]
    except Exception as e:
        print(f'Failed to load {fname}: {e}')

print(f'Loaded {len(oofs)} OOF signals, n={len(y)}, positives={y.sum()}')

# Compute individual F1
names = list(oofs.keys())
P = np.column_stack([oofs[n] for n in names])
individual = []
for i, n in enumerate(names):
    oof = P[:, i]
    best_f1 = 0
    for th in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y, (oof > th).astype(int), zero_division=0)
        if f > best_f1: best_f1 = f
    individual.append((best_f1, n))
individual.sort(reverse=True)
print('\nTop 10 individual OOFs:')
for f, n in individual[:10]:
    print(f'  {n}: F1={f:.4f}')

# Use top-K for weight optimization
top_k = 8
top_names = [n for _, n in individual[:top_k]]
P_top = np.column_stack([oofs[n] for n in top_names])
print(f'\nOptimizing weights for: {top_names}')

# Random search over weight simplex
rng = np.random.RandomState(42)
best_f1, best_w = 0, None
ths = np.arange(0.05, 0.95, 0.01)
for trial in range(50000):
    # sample from Dirichlet(1,...,1) to stay on simplex
    w = rng.dirichlet(np.ones(top_k))
    oof = P_top @ w
    # best threshold (vectorized)
    preds = (oof[:, None] > ths[None, :]).astype(int)  # [N, n_th]
    tps = ((preds == 1) & (y[:, None] == 1)).sum(axis=0)
    fps = ((preds == 1) & (y[:, None] == 0)).sum(axis=0)
    fns = ((preds == 0) & (y[:, None] == 1)).sum(axis=0)
    prec = np.where(tps + fps > 0, tps / (tps + fps), 0)
    rec = np.where(tps + fns > 0, tps / (tps + fns), 0)
    f = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
    if f.max() > best_f1:
        best_f1, best_w = f.max(), w.copy()

print(f'\nBest weighted ensemble (random search): F1={best_f1:.4f}')
for name, w in zip(top_names, best_w):
    print(f'  {name}: {w:.4f}')

oof_best = P_top @ best_w
best_th = 0.5
best_f1_th = 0
for th in np.arange(0.05, 0.95, 0.001):
    pred = (oof_best > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > best_f1_th:
        best_f1_th, best_th = f, th
pred = (oof_best > best_th).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f'\nFinal: F1={best_f1_th:.4f}, th={best_th:.3f}, Rec={recall_score(y,pred):.4f}, '
      f'Prec={precision_score(y,pred):.4f}, AUC={roc_auc_score(y,oof_best):.4f}, TP={tp}, FP={fp}, FN={fn}')

# Save best ensemble
np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'optimized_weighted_ensemble.npz'),
    oof_optimized=oof_best, flags=y, weights=best_w, names=np.array(top_names), threshold=best_th,
)
print(f'Saved to output/optimized_weighted_ensemble.npz')
