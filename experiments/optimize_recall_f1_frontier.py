"""Search for a weighted OOF blend that maximizes F1 while pushing recall >= 0.90.

Uses the top-K individual OOF signals.  For each random weight vector we do a
threshold search and record the (recall, precision, F1) frontier.  The goal is
to see whether any convex combination of existing strong signals can satisfy
both F1 > 0.90 and recall > 0.90 on the original SGCC labels.
"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything

seed_everything(SEED)

flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']

# Top OOF signals discovered by quick_oof_eval.csv (continuous probabilities, not labels)
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

# Baseline: equal weight
eq_w = np.ones(n) / n
eq_prob = X.dot(eq_w)
best_f1, best_th = 0, 0.5
for th in np.arange(0.05, 0.95, 0.005):
    f = f1_score(flags, (eq_prob > th).astype(int), zero_division=0)
    if f > best_f1:
        best_f1, best_th = f, th
pred = (eq_prob > best_th).astype(int)
print(f'Equal-weight blend: F1={best_f1:.4f}, Rec={recall_score(flags,pred):.4f}, Prec={precision_score(flags,pred,zero_division=0):.4f}, th={best_th:.3f}')

# Random weight search
frontier = []
n_samples = 5000
best_overall = {'f1': 0}
best_recall_constrained = {'f1': 0}
for i in range(n_samples):
    # Dirichlet random weights (convex combination)
    w = np.random.dirichlet(np.ones(n))
    prob = X.dot(w)
    # threshold search
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (prob > th).astype(int)
        if pred.sum() == 0:
            continue
        rec = recall_score(flags, pred, zero_division=0)
        prec = precision_score(flags, pred, zero_division=0)
        f = f1_score(flags, pred, zero_division=0)
        frontier.append({'f1': f, 'recall': rec, 'precision': prec, 'th': th})
        if f > best_overall['f1']:
            best_overall = {'f1': f, 'recall': rec, 'precision': prec, 'th': th, 'w': w.copy()}
        if rec >= 0.90 and f > best_recall_constrained['f1']:
            best_recall_constrained = {'f1': f, 'recall': rec, 'precision': prec, 'th': th, 'w': w.copy()}

frontier_df = pd.DataFrame(frontier)
frontier_df.to_csv(os.path.join(OUTPUT_DIR, 'recall_f1_frontier_random.csv'), index=False)

print('\nBest unconstrained:', best_overall)
print('\nBest with recall >= 0.90:', best_recall_constrained)

# Print Pareto frontier (max F1 for each recall bin)
frontier_df['recall_bin'] = (frontier_df['recall'] * 20).astype(int) / 20.0
pareto = frontier_df.loc[frontier_df.groupby('recall_bin')['f1'].idxmax()].sort_values('recall_bin')
print('\nPareto frontier (recall bin -> best F1):')
print(pareto[['recall_bin','f1','recall','precision','th']].to_string(index=False))
