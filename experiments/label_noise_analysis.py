"""Analyze potential label noise using strong OOF consensus."""
import os, sys
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR

y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']

# Load strong OOFs
oofs = {}
for f, k in [
    ('sgcc_mega_meta.npz', 'oof_final'),
    ('autoresearch_best.npz', 'oof_final'),
    ('mega_boost_enhanced.npz', 'oof_final'),
    ('informer_oof.npz', 'oof_informer'),
    ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10'),
    ('patch_transformer_raw_3ch_oof.npz', 'oof_patch_transformer_raw_3ch'),
    ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch'),
    ('strong_gbdt_prior_oof.npz', 'oof_strong_gbdt_prior'),
]:
    try:
        oofs[f] = np.load(os.path.join(OUTPUT_DIR, f))[k]
    except Exception:
        pass

P = np.column_stack(list(oofs.values()))
mean_prob = P.mean(axis=1)
median_prob = np.median(P, axis=1)
min_prob = P.min(axis=1)
max_prob = P.max(axis=1)

# Consensus: fraction of OOFs predicting positive at threshold 0.5
consensus_pos = (P > 0.5).mean(axis=1)

print('=== Label noise analysis ===')
print(f'Total samples: {len(y)}, positives: {y.sum()}, negatives: {(y==0).sum()}')

# Potential false positives: label=1 but consensus low
fp_candidates = (y == 1) & (consensus_pos < 0.3)
print(f'\nPotential false positives (label=1, consensus<0.3): {fp_candidates.sum()}')
print(f'  Their mean prob={mean_prob[fp_candidates].mean():.3f}, median={np.median(median_prob[fp_candidates]):.3f}')

# Potential missed positives: label=0 but consensus high
fn_candidates = (y == 0) & (consensus_pos > 0.7)
print(f'\nPotential missed positives (label=0, consensus>0.7): {fn_candidates.sum()}')
print(f'  Their mean prob={mean_prob[fn_candidates].mean():.3f}, median={np.median(median_prob[fn_candidates]):.3f}')

# More detailed breakdown
print('\n=== Breakdown by label and consensus ===')
for label in [0, 1]:
    idx = np.where(y == label)[0]
    print(f'\nTrue label {label}:')
    for low, high in [(0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.0)]:
        n = ((consensus_pos[idx] >= low) & (consensus_pos[idx] < high)).sum()
        print(f'  consensus [{low:.1f},{high:.1f}): {n:5d}')

# Simulate cleaning: flip labels of strong candidates and recompute meta F1
meta = np.load(os.path.join(OUTPUT_DIR, 'sgcc_mega_meta.npz'))['oof_final']
print('\n=== Simulate label cleaning ===')
for fp_th in [0.1, 0.2, 0.3]:
    for fn_th in [0.7, 0.8, 0.9]:
        y_clean = y.copy()
        flip_fp = (y == 1) & (consensus_pos < fp_th)
        flip_fn = (y == 0) & (consensus_pos > fn_th)
        y_clean[flip_fp] = 0
        y_clean[flip_fn] = 1
        n_flip = flip_fp.sum() + flip_fn.sum()
        # Recompute best F1 on cleaned labels using current meta OOF
        best = max(f1_score(y_clean, (meta > th).astype(int), zero_division=0) for th in np.arange(0.05, 0.95, 0.005))
        print(f'  fp_th={fp_th:.1f} fn_th={fn_th:.1f} flips={n_flip:4d} meta F1={best:.4f}')
