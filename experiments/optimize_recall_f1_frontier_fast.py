"""Fast vectorized search for a weighted OOF blend with recall >= 0.90 and F1 > 0.90."""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything

seed_everything(SEED)

flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags'].astype(np.int64)
n_pos = flags.sum()

TOP_SIGNALS = [
    ('hillclimb_best_oof.npz', 'oof_hillclimb'),
    ('sgcc_mega_meta.npz', 'oof_final'),
    ('sgcc_mega_meta_xgb_d3_1000.npz', 'oof_final'),
    ('autoresearch_best.npz', 'oof_final'),
    ('meta_raw_only_oof.npz', 'oof_meta_raw_only'),
    ('mega_boost_enhanced.npz', 'oof_final'),
    ('mega_boost_final.npz', 'oof_final'),
    ('heterogeneous_ensemble.npz', 'oof_final'),
    ('feature_rich_meta_oof.npz', 'oof_feature_rich_meta'),
    ('stronger_gbdt_prior_v3.npz', 'prior'),
]

signals = []
names = []
for fname, key in TOP_SIGNALS:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(fpath):
        continue
    arr = np.load(fpath)[key].astype(np.float64)
    if len(arr) != len(flags):
        continue
    signals.append(arr)
    names.append(f'{fname}:{key}')
    print(f'Loaded {fname}:{key}')

X = np.column_stack(signals)
n = X.shape[1]
print(f'\nBlending {n} signals, {len(flags)} samples')

def pr_curve(prob):
    order = np.argsort(prob)[::-1]
    y_sorted = flags[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    rec = tp / n_pos
    prec = tp / (tp + fp)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return rec, prec, f1

# Equal weight baseline
w_eq = np.ones(n) / n
prob_eq = X.dot(w_eq)
rec_eq, prec_eq, f1_eq = pr_curve(prob_eq)
idx = np.argmax(f1_eq)
print(f'Equal-weight blend: F1={f1_eq[idx]:.4f}, Rec={rec_eq[idx]:.4f}, Prec={prec_eq[idx]:.4f}')

n_samples = 2000
best_overall = {'f1': 0.0}
best_recall90 = {'f1': 0.0}
frontier = []
for i in range(n_samples):
    w = np.random.dirichlet(np.ones(n))
    prob = X.dot(w)
    rec, prec, f1 = pr_curve(prob)
    idx_best = np.argmax(f1)
    frontier.append({'f1': f1[idx_best], 'recall': rec[idx_best], 'precision': prec[idx_best]})
    if f1[idx_best] > best_overall['f1']:
        best_overall = {'f1': f1[idx_best], 'recall': rec[idx_best], 'precision': prec[idx_best], 'w': w.copy()}
    # best with recall >= 0.90
    mask90 = rec >= 0.90
    if mask90.any():
        idx90 = np.argmax(f1 * mask90)  # argmax over masked; zeros elsewhere
        if f1[idx90] > best_recall90['f1']:
            best_recall90 = {'f1': f1[idx90], 'recall': rec[idx90], 'precision': prec[idx90], 'w': w.copy()}

frontier_df = pd.DataFrame(frontier)
frontier_df.to_csv(os.path.join(OUTPUT_DIR, 'recall_f1_frontier_fast.csv'), index=False)

print('\nBest unconstrained blend:', best_overall)
print('\nBest blend with recall >= 0.90:', best_recall90)

# Pareto summary: best F1 in recall bins
frontier_df['recall_bin'] = (frontier_df['recall'] * 20).astype(int) / 20.0
pareto = frontier_df.loc[frontier_df.groupby('recall_bin')['f1'].idxmax()].sort_values('recall_bin')
print('\nPareto frontier (recall bin -> best F1):')
print(pareto[['recall_bin','f1','recall','precision']].to_string(index=False))

# Save best weights for possible reuse
if best_recall90['f1'] > 0:
    np.savez(os.path.join(OUTPUT_DIR, 'best_recall90_blend_weights.npz'),
             names=names, weights=best_recall90['w'])
