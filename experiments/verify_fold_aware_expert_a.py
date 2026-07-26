"""
Phase 4: Verify the actual impact of fold-aware (leakage-safe) preprocessing.

Controlled experiment: same base feature set (compute_statistical_features +
novel missing features), two variants:
  - global: statistics fit on ALL users (legacy preprocess_sgcc behaviour)
  - fold-aware: statistics fit on each fold's TRAINING users only

Any Δ is attributable purely to leakage through global statistics
(winsorize bounds + global P95/P99/P999 + volatility/entropy clip maxima).

Usage:
    C:/Users/yangj/.conda/envs/ml/python.exe experiments/verify_fold_aware_expert_a.py
"""
import os
import sys
import time
import warnings

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings('ignore')

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.data.preprocess_sgcc import (
    load_sgcc, three_layer_imputation, compute_statistical_features,
    compute_novel_missing_features,
)
from src.data.fold_aware_preprocessor import FoldAwarePreprocessor
from src.training.expert_a import ExpertATrainer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score


def best_f1(y, p):
    best, th_best = 0.0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y, (p > th).astype(int), zero_division=0)
        if f > best:
            best, th_best = f, th
    return best, th_best


def build_features(transformed, impute_mask, raw, residuals, g_percentiles=None):
    """Base feature set: statistical + novel missing features."""
    stat, _ = compute_statistical_features(
        transformed, impute_mask, raw, residuals, g_percentiles=g_percentiles)
    novel = compute_novel_missing_features(raw, transformed)
    return np.column_stack([stat, novel]).astype(np.float32)


def train_gbdt_oof(stat_features, y, fold_assignments):
    """Train Expert A (GBDT ensemble) with pre-defined folds; return OOF."""
    trainer = ExpertATrainer(dataset='sgcc')
    oof, _, _ = trainer.train(stat_features, y, impute_mask=None,
                              fold_assignments=fold_assignments)
    return oof


def main():
    t0 = time.time()
    print('=' * 70)
    print('Phase 4: fold-aware vs global preprocessing (Expert A, base features)')
    print('=' * 70)

    raw, flags, cons_no, date_cols = load_sgcc()
    imputed, impute_mask = three_layer_imputation(raw, flags)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_assignments = np.zeros(len(flags), dtype=int)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(flags)), flags)):
        fold_assignments[val_idx] = k

    n = len(flags)
    oof_global = np.zeros(n)
    oof_foldaware = np.zeros(n)

    for k in range(N_FOLDS):
        train_idx = np.where(fold_assignments != k)[0]
        val_idx = np.where(fold_assignments == k)[0]
        print(f'\n--- Fold {k+1}/{N_FOLDS}: train={len(train_idx)} val={len(val_idx)} ---')

        # ---- global variant (legacy): stats fit on ALL users ----
        prep_g = FoldAwarePreprocessor()
        prep_g.fit(raw, imputed)  # all users
        tg, _, vg, eg, rg = prep_g.transform_imputed(imputed, impute_mask)
        feats_g = build_features(tg, impute_mask, raw, rg,
                                 g_percentiles={'g_p95': prep_g.global_p95_,
                                                'g_p99': prep_g.global_p99_,
                                                'g_p999': prep_g.global_p999_})
        # fold-aware features for the SAME comparison on this fold's data
        prep_f = FoldAwarePreprocessor()
        prep_f.fit(raw[train_idx], imputed[train_idx])
        tf, _, vf, ef, rf = prep_f.transform_imputed(imputed, impute_mask)
        feats_f = build_features(tf, impute_mask, raw, rf,
                                 g_percentiles=prep_f.transform_raw_for_stats(
                                     raw[train_idx], impute_mask[train_idx]))

        # Feature magnitude sanity check
        delta = np.abs(feats_g - feats_f)
        print(f'  feature diff: mean|Δ|={delta.mean():.6f}, max|Δ|={delta.max():.6f}')

        # Train both on this fold's train split, predict val
        for tag, feats, store in (('global', feats_g, oof_global),
                                   ('fold-aware', feats_f, oof_foldaware)):
            trainer = ExpertATrainer(dataset='sgcc')
            X_tr, X_va = feats[train_idx], feats[val_idx]
            y_tr = flags[train_idx]
            # reuse trainer internals per single fold
            stat_aug = trainer._prepare_features(feats, None)
            import lightgbm as lgb
            import xgboost as xgb
            from catboost import CatBoostClassifier
            lgb_m = lgb.LGBMClassifier(**trainer.config['gbdt_params']['lgb'])
            lgb_m.fit(stat_aug[train_idx], y_tr,
                      eval_set=[(stat_aug[val_idx], flags[val_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=0)])
            xgb_m = xgb.XGBClassifier(**trainer.config['gbdt_params']['xgb'])
            xgb_m.fit(stat_aug[train_idx], y_tr, verbose=False)
            cb_m = CatBoostClassifier(**trainer.config['gbdt_params']['catboost'])
            cb_m.fit(stat_aug[train_idx], y_tr, verbose=0)
            p = (lgb_m.predict_proba(stat_aug[val_idx])[:, 1]
                 + xgb_m.predict_proba(stat_aug[val_idx])[:, 1]
                 + cb_m.predict_proba(stat_aug[val_idx])[:, 1]) / 3.0
            store[val_idx] = p
            f1, th = best_f1(flags[val_idx], p)
            print(f'  [{tag}] fold F1={f1:.4f} AUC={roc_auc_score(flags[val_idx], p):.4f}')

    # Overall comparison
    f1g, thg = best_f1(flags, oof_global)
    f1f, thf = best_f1(flags, oof_foldaware)
    aucg = roc_auc_score(flags, oof_global)
    aucf = roc_auc_score(flags, oof_foldaware)
    pred_agree = ((oof_global > thg).astype(int) == (oof_foldaware > thf).astype(int)).mean()

    lines = [
        '# fold-aware vs global preprocessing (Expert A, base feature set)',
        '',
        '| variant | F1 | threshold | AUC |',
        '|---|---|---|---|',
        f'| global (legacy) | {f1g:.4f} | {thg:.3f} | {aucg:.4f} |',
        f'| fold-aware | {f1f:.4f} | {thf:.3f} | {aucf:.4f} |',
        f'| **Δ (fold-aware − global)** | **{f1f-f1g:+.4f}** | — | **{aucf-aucg:+.4f}** |',
        '',
        f'- prediction agreement: {pred_agree*100:.2f}%',
        f'- n={n}, folds={N_FOLDS}, seed={SEED}',
        f'- runtime: {(time.time()-t0)/60:.1f} min',
        '',
        'Interpretation: if |ΔF1| < 0.005, global-statistics leakage has',
        'negligible practical impact on this pipeline (as predicted by the',
        'self-audit); the fix is still correct hygiene and closes the',
        'reviewer\'s data-leakage concern (方法学审核报告 6.1/6.2).',
    ]
    report = '\n'.join(lines)
    print('\n' + report)

    out_md = os.path.join(OUTPUT_DIR, 'fold_aware_vs_global.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'fold_aware_verify.npz'),
        oof_global=oof_global, oof_foldaware=oof_foldaware, y=flags)
    print(f'\nsaved -> {out_md}')


if __name__ == '__main__':
    main()
