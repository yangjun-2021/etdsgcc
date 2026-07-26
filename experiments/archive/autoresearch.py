"""
AutoResearch: Autonomous experiment loop for Mega Boost
=========================================================
Target: Maximize pure-auto F1 beyond 0.8602
Strategy: Systematic search over expert configs + feature subsets
"""
import os, time, glob, warnings, traceback
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from utils import seed_everything

SEED = 42
N_FOLDS = 5
OUT = 'output'
OD = r'D:\Project\ThiefElectricity\output'


def load_features_and_labels():
    """Load mega features and labels once."""
    base = np.load(os.path.join(OUT, 'sgcc_preprocessed.npz'))
    y = base['flags'].astype(int)
    stat_feat = np.nan_to_num(base['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)

    novel = np.load(os.path.join(OUT, 'novel_features.npz'))
    novel_feat = np.nan_to_num(novel['features'], nan=0.0, posinf=0.0, neginf=0.0)

    deng = np.load(os.path.join(OUT, 'dengine_features.npz'))
    deng_feat = np.nan_to_num(deng['X'], nan=0.0, posinf=0.0, neginf=0.0)

    import pandas as pd
    raw_df = pd.read_csv('data/raw_data.csv')
    date_cols = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = raw_df[date_cols].values.astype(float)
    del raw_df
    n_days = raw.shape[1]
    n = len(y)

    N_PAA = 50
    seg_size = n_days / N_PAA
    paa_feat = np.zeros((n, N_PAA), dtype=np.float32)
    for i in range(N_PAA):
        start = int(round(i * seg_size))
        end = int(round((i + 1) * seg_size))
        if end > start:
            paa_feat[:, i] = np.nanmean(raw[:, start:end], axis=1)
    paa_feat = np.nan_to_num(paa_feat, nan=0.0)

    residuals = base['residuals']
    impute_mask = base['impute_mask']
    half = n_days // 2

    res_agg_list = [
        np.nanmean(residuals, axis=1).reshape(-1, 1),
        np.nanstd(residuals, axis=1).reshape(-1, 1),
        np.nanmean(np.abs(residuals), axis=1).reshape(-1, 1),
        np.nanmax(np.abs(residuals), axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 25, axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 75, axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 90, axis=1).reshape(-1, 1),
        np.nanpercentile(residuals, 95, axis=1).reshape(-1, 1),
    ]
    res_first = np.nanmean(residuals[:, :half], axis=1).reshape(-1, 1)
    res_second = np.nanmean(residuals[:, half:], axis=1).reshape(-1, 1)
    res_trend = (res_second - res_first) / (np.maximum(np.abs(res_first), 1e-6))
    res_agg_list.append(res_trend.reshape(-1, 1))
    for w in [30, 60, 90, 180]:
        if n_days >= w:
            res_agg_list.append(np.nanmean(np.abs(residuals[:, -w:]), axis=1).reshape(-1, 1))
    res_agg = np.nan_to_num(np.column_stack(res_agg_list), nan=0.0, posinf=0.0, neginf=0.0)

    mask_agg_list = [impute_mask.astype(float).mean(axis=1).reshape(-1, 1)]
    for seg_start, seg_end in [
        (0, n_days // 4),
        (n_days // 4, n_days // 2),
        (n_days // 2, 3 * n_days // 4),
        (3 * n_days // 4, n_days),
        (0, half),
        (half, n_days),
    ]:
        mask_agg_list.append(impute_mask[:, seg_start:seg_end].astype(float).mean(axis=1).reshape(-1, 1))
    missing_runs = np.zeros(n)
    for i in range(n):
        runs = []
        cr = 0
        for m in impute_mask[i]:
            if m: cr += 1
            else:
                if cr > 0: runs.append(cr)
                cr = 0
        if cr > 0: runs.append(cr)
        missing_runs[i] = max(runs) if runs else 0
    mask_agg_list.append(missing_runs.reshape(-1, 1))
    mask_agg_list.append((~impute_mask).sum(axis=1).reshape(-1, 1) / n_days)
    mask_agg = np.nan_to_num(np.column_stack(mask_agg_list), nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'y': y,
        'stat_feat': stat_feat,
        'novel_feat': novel_feat,
        'deng_feat': deng_feat,
        'paa_feat': paa_feat,
        'res_agg': res_agg,
        'mask_agg': mask_agg,
    }


def load_existing_oofs(y):
    existing = {}
    our = np.load(os.path.join(OUT, 'tcn_kd_results.npz'))
    existing['TCN+KD'] = our['oof_tcn_kd']
    existing['Super-GBDT'] = np.load(os.path.join(OUT, 'super_gbdt.npz'))['oof_super']
    smart = np.load(os.path.join(OUT, 'smart_blend.npz'))
    existing['SmartBlend'] = smart['oof_final']
    for tag, prefix, key in [
        ('V213', 'v213_results_', 'oof_v213'),
        ('V219', 'v219_results_', 'oof_final'),
        ('V225', 'v225_results_', 'oof_final'),
        ('V216', 'v216_results_', 'oof_final'),
    ]:
        files = sorted(glob.glob(os.path.join(OD, f'{prefix}*.npz')), reverse=True)
        if files:
            d = np.load(files[0], allow_pickle=True)
            if key in d.files:
                existing[tag] = d[key]
    return existing


def build_X(features, active_groups):
    parts = []
    if 'stat' in active_groups:
        parts.append(features['stat_feat'])
    if 'novel' in active_groups:
        parts.append(features['novel_feat'])
    if 'deng' in active_groups:
        parts.append(features['deng_feat'])
    if 'paa' in active_groups:
        parts.append(features['paa_feat'])
    if 'res' in active_groups:
        parts.append(features['res_agg'])
    if 'mask' in active_groups:
        parts.append(features['mask_agg'])
    if not parts:
        parts.append(features['stat_feat'])
    X = np.column_stack(parts)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e4, 1e4).astype(np.float32)
    return X


def train_one_expert(X, y, name, config, skf):
    n = len(y)
    oof = np.zeros(n)
    tag = config['tag']
    del config['tag']

    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[ti], X[vi]
        y_tr, y_va = y[ti], y[vi]
        pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        if tag == 'lgb':
            m = lgb.LGBMClassifier(**config, scale_pos_weight=pw, random_state=SEED + fi, verbose=-1)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        elif tag == 'xgb':
            m = xgb.XGBClassifier(**config, scale_pos_weight=pw, random_state=SEED + fi)
            m.fit(X_tr, y_tr)
        elif tag == 'cat':
            m = cb.CatBoostClassifier(**config)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=80, verbose=False)
        elif tag == 'et':
            m = ExtraTreesClassifier(**config, random_state=SEED + fi, n_jobs=-1, bootstrap=True)
            m.fit(X_tr, y_tr)
        elif tag == 'rf':
            m = RandomForestClassifier(**config, random_state=SEED + fi, n_jobs=-1, bootstrap=True)
            m.fit(X_tr, y_tr)

        p = m.predict_proba(X_va)[:, 1]
        oof[vi] = p

    oof = np.nan_to_num(oof, nan=0.5)
    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
    auc = roc_auc_score(y, oof)
    return oof, bf, auc, name


def evaluate_ensemble(all_oofs, y, verbose=False):
    names = list(all_oofs.keys())
    P = np.column_stack([all_oofs[nm] for nm in names])
    n = len(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_stack = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        meta = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED)
        meta.fit(P[ti], y[ti])
        oof_stack[vi] = meta.predict_proba(P[vi])[:, 1]

    bf = 0
    for th in np.arange(0.05, 0.95, 0.001):
        pred = (oof_stack > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > bf: bf = f
    auc = roc_auc_score(y, oof_stack)

    if verbose:
        corrs = np.corrcoef(P.T)
        avg_corr = (corrs.sum() - len(names)) / (len(names) * (len(names) - 1))
        print(f"  n_experts={len(names):2d}  avg_corr={avg_corr:.3f}  F1={bf:.4f}  AUC={auc:.4f}")

    return bf, auc, oof_stack


# ═══════════════════════════════════════════════════════════════════
# AUTORESEARCH LOOP
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = time.time()
    seed_everything(SEED)

    print("Loading features...")
    features = load_features_and_labels()
    y = features['y']
    existing = load_existing_oofs(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    best_f1 = 0
    best_run = None
    results_log = []

    expert_configs = [
        {'tag': 'lgb', 'name': 'LGB', 'n_estimators': 1200, 'max_depth': 7, 'learning_rate': 0.05,
         'num_leaves': 63, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1},
        {'tag': 'xgb', 'name': 'XGB', 'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.04,
         'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
         'min_child_weight': 5, 'tree_method': 'hist', 'verbosity': 0},
        {'tag': 'cat', 'name': 'Cat', 'iterations': 800, 'depth': 8, 'learning_rate': 0.06,
         'l2_leaf_reg': 3.0, 'subsample': 0.8, 'verbose': 0, 'random_seed': SEED},
        {'tag': 'et', 'name': 'ET', 'n_estimators': 200, 'max_depth': 12,
         'min_samples_leaf': 20, 'max_features': 0.3, 'class_weight': 'balanced'},
        {'tag': 'rf', 'name': 'RF', 'n_estimators': 200, 'max_depth': 12,
         'min_samples_leaf': 20, 'max_features': 0.3, 'class_weight': 'balanced'},
    ]

    feature_groups_pool = [
        ['stat', 'novel', 'deng', 'paa', 'res', 'mask'],  # ALL (537 dims)
        ['stat', 'novel', 'paa', 'res', 'mask'],            # no deng
        ['stat', 'deng', 'paa', 'res', 'mask'],             # no novel
        ['stat', 'novel', 'deng', 'res', 'mask'],           # no paa
        ['stat', 'novel', 'deng', 'paa', 'mask'],           # no res
        ['stat', 'novel', 'deng', 'paa'],                   # no mask
        ['stat', 'novel', 'deng'],                          # minimal
    ]

    print(f"\n{'=' * 70}")
    print(f"  AUTORESEARCH: {len(feature_groups_pool)} feature × {len(expert_configs)} expert variants")
    print(f"{'=' * 70}")

    run_idx = 0
    for fg_id, active_groups in enumerate(feature_groups_pool):
        X = build_X(features, active_groups)

        for num_experts in range(3, 6):
            for start_idx in range(len(expert_configs) - num_experts + 1):
                selected_configs = expert_configs[start_idx:start_idx + num_experts]

                run_idx += 1
                print(f"\n--- Run {run_idx}: {len(active_groups)} groups, {num_experts} experts ---")

                try:
                    t_run = time.time()
                    all_oofs = {}

                    for cfg in selected_configs:
                        name = cfg.pop('name')
                        oof, bf, auc, nm = train_one_expert(X, y, name, cfg.copy(), skf)
                        all_oofs[name] = oof
                        cfg['name'] = name

                    for name, oof in existing.items():
                        all_oofs[name] = oof

                    bf, auc, oof_stack = evaluate_ensemble(all_oofs, y, verbose=False)

                    n_total = len(all_oofs)
                    run_time = time.time() - t_run
                    entry = {
                        'run': run_idx, 'groups': '+'.join(active_groups),
                        'n_dims': X.shape[1], 'n_new_experts': num_experts,
                        'n_total_models': n_total, 'f1': bf, 'auc': auc,
                        'time_min': run_time / 60,
                    }
                    results_log.append(entry)

                    status = '*** NEW BEST ***' if bf > best_f1 else ''
                    if bf > best_f1:
                        best_f1 = bf
                        best_run = entry
                        np.savez_compressed(
                            os.path.join(OUT, 'autoresearch_best.npz'),
                            oof_final=oof_stack, y=y, f1=bf, auc=auc,
                            groups=active_groups, n_experts=num_experts,
                        )

                    print(f"  {n_total} models  |  F1={bf:.4f}  AUC={auc:.4f}"
                          f"  |  {X.shape[1]} dims  |  {run_time:.0f}s  {status}")

                except Exception as e:
                    print(f"  FAILED: {e}")
                    traceback.print_exc()

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  AUTORESEARCH COMPLETE")
    print(f"{sep}")
    print(f"\n  Best Run: {best_run}")

    sorted_results = sorted(results_log, key=lambda x: -x['f1'])
    print(f"\n  {'Run':<5s} {'F1':>8s} {'AUC':>8s} {'Dims':>5s} {'Models':>6s} {'Groups'}")
    print(f"  {'-' * 60}")
    for r in sorted_results[:10]:
        print(f"  {r['run']:>4d}  {r['f1']:>8.4f} {r['auc']:>8.4f} {r['n_dims']:>5d} "
              f"{r['n_total_models']:>6d}  {r['groups']}")

    total_min = (time.time() - t_start) / 60
    print(f"\n  Total time: {total_min:.1f} min  |  Best F1: {best_f1:.4f}")
    print(f"  vs V225 (0.8457): {best_f1 - 0.8457:+.4f}")
    print(f"  vs Super-GBDT (0.8527): {best_f1 - 0.8527:+.4f}")

    np.savez(os.path.join(OUT, 'autoresearch_log.npz'),
             results=np.array([(r['f1'], r['auc'], r['n_dims'], r['n_total_models']) for r in results_log]),
             groups=[r['groups'] for r in results_log])
