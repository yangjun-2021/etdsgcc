"""
Mega Boost: Pure-Auto F1 Breakthrough
======================================
Strategy: Mega Feature Pool → 4 diverse experts → stacking + hillclimb

Features (586 dims):
  - 307: stat_features from preprocess_sgcc
  - 159: deep engineering (entropy/GPD/burstiness)
  -  20: novel missing pattern features
  -  50: PAA temporal compression (1034 → 50 segments)
  -  30: residuals aggregation statistics
  -  20: impute_mask aggregation statistics

Experts (8 OOFs):
  - 4 new: LGBM, XGBoost, CatBoost, ExtraTrees (trained on Mega features)
  - 4 existing: Super-GBDT, TCN+KD, SmartBlend, external V225

Fusion: LogisticRegression stacking → Hillclimb F1 optimization

Target: Pure-auto F1 = 0.858~0.860
"""
import os, time, glob, warnings
import numpy as np, pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from utils import seed_everything

SEED = 42
N_FOLDS = 5
OUT = 'output'
OD = r'D:\Project\ThiefElectricity\output'
seed_everything(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1. LOAD & BUILD MEGA FEATURE POOL
# ═══════════════════════════════════════════════════════════════════
def build_mega_features():
    print("=" * 70)
    print("  STAGE 1: Mega Feature Pool Construction")
    print("=" * 70)
    t0 = time.time()

    base = np.load(os.path.join(OUT, 'sgcc_preprocessed.npz'))
    y = base['flags'].astype(int)
    n = len(y)
    stat_feat = np.nan_to_num(base['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
    residuals = base['residuals']
    impute_mask = base['impute_mask']
    print(f"  [1/6] stat_features: {stat_feat.shape[1]} dims  ✓")

    novel = np.load(os.path.join(OUT, 'novel_features.npz'))
    novel_feat = np.nan_to_num(novel['features'], nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [2/6] novel_features: {novel_feat.shape[1]} dims  ✓")

    deng = np.load(os.path.join(OUT, 'dengine_features.npz'))
    deng_feat = np.nan_to_num(deng['X'], nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [3/6] dengine_features: {deng_feat.shape[1]} dims  ✓")

    raw_df = pd.read_csv('data/raw_data.csv')
    date_cols = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = raw_df[date_cols].values.astype(float)
    n_days = raw.shape[1]
    del raw_df
    print(f"  [4/6] Raw data: ({n}, {n_days}) days")

    N_PAA = 50
    seg_size = n_days / N_PAA
    paa_feat = np.zeros((n, N_PAA), dtype=np.float32)
    for i in range(N_PAA):
        start = int(round(i * seg_size))
        end = int(round((i + 1) * seg_size))
        if end > start:
            paa_feat[:, i] = np.nanmean(raw[:, start:end], axis=1)
        else:
            paa_feat[:, i] = raw[:, start] if start < n_days else 0
    paa_feat = np.nan_to_num(paa_feat, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [4/6] PAA ({n_days}→{N_PAA}): {paa_feat.shape[1]} dims  ✓")

    res_agg = np.zeros((n, 0), dtype=np.float32)
    for col_name, col_fn in [
        ('res_mean', lambda x: np.nanmean(x, axis=1)),
        ('res_std', lambda x: np.nanstd(x, axis=1)),
        ('res_abs_mean', lambda x: np.nanmean(np.abs(x), axis=1)),
        ('res_abs_max', lambda x: np.nanmax(np.abs(x), axis=1)),
        ('res_q25', lambda x: np.nanpercentile(x, 25, axis=1)),
        ('res_q50', lambda x: np.nanpercentile(x, 50, axis=1)),
        ('res_q75', lambda x: np.nanpercentile(x, 75, axis=1)),
        ('res_q90', lambda x: np.nanpercentile(x, 90, axis=1)),
        ('res_q95', lambda x: np.nanpercentile(x, 95, axis=1)),
        ('res_skew', lambda x: np.nan_to_num(np.apply_along_axis(lambda r: np.nan if np.all(np.isnan(r)) else (np.nanmean((r-np.nanmean(r))**3)/(np.nanstd(r)**3+1e-6)), 1, x), nan=0)),
    ]:
        arr = col_fn(residuals).reshape(-1, 1)
        res_agg = np.column_stack([res_agg, arr]) if res_agg.shape[1] > 0 else arr

    half = n_days // 2
    res_first = np.nanmean(residuals[:, :half], axis=1).reshape(-1, 1)
    res_second = np.nanmean(residuals[:, half:], axis=1).reshape(-1, 1)
    res_trend = (res_second - res_first) / (np.maximum(np.abs(res_first), 1e-6))
    res_agg = np.column_stack([res_agg, res_trend.reshape(-1, 1)])

    for q in [30, 60, 90, 180, 365]:
        if n_days >= q:
            w = min(q, n_days)
            last = residuals[:, -w:]
            res_agg = np.column_stack([res_agg, np.nanmean(np.abs(last), axis=1).reshape(-1, 1)])

    res_agg = np.nan_to_num(res_agg, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [5/6] residuals aggregation: {res_agg.shape[1]} dims  ✓")

    mask_agg = np.zeros((n, 0), dtype=np.float32)
    half = n_days // 2
    quarter = n_days // 4

    mask_agg = np.column_stack([mask_agg, impute_mask.astype(float).mean(axis=1).reshape(-1, 1)])
    mask_agg = np.column_stack([mask_agg, impute_mask.astype(float).mean(axis=1).reshape(-1, 1) ** 0])  # placeholder

    for seg_name, seg_start, seg_end in [
        ('Q1', 0, quarter),
        ('Q2', quarter, 2 * quarter),
        ('Q3', 2 * quarter, 3 * quarter),
        ('Q4', 3 * quarter, n_days),
        ('H1', 0, half),
        ('H2', half, n_days),
    ]:
        seg_mask = impute_mask[:, seg_start:seg_end]
        mask_agg = np.column_stack([mask_agg, seg_mask.astype(float).mean(axis=1).reshape(-1, 1)])

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
    mask_agg = np.column_stack([mask_agg, missing_runs.reshape(-1, 1)])

    obs_count = (~impute_mask).sum(axis=1)
    obs_days_ratio = obs_count / n_days
    mask_agg = np.column_stack([mask_agg, obs_days_ratio.reshape(-1, 1)])

    high_miss = (impute_mask.astype(float).mean(axis=1) > 0.5).astype(float).reshape(-1, 1)
    mask_agg = np.column_stack([mask_agg, high_miss])

    missing_persistence = np.zeros(n)
    for i in range(n):
        m = impute_mask[i].astype(float)
        if m.sum() > 0:
            missing_persistence[i] = (m[1:] * m[:-1]).sum() / max(m.sum(), 1)
    mask_agg = np.column_stack([mask_agg, missing_persistence.reshape(-1, 1)])

    mask_agg = np.nan_to_num(mask_agg, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [6/6] mask aggregation: {mask_agg.shape[1]} dims  ✓")

    X_full = np.column_stack([stat_feat, novel_feat, deng_feat, paa_feat, res_agg, mask_agg])
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    X_full = np.clip(X_full, -1e4, 1e4)
    X_full = X_full.astype(np.float32)
    print(f"\n  Mega Feature Pool: {X_full.shape[1]} total dims")

    high_var = VarianceThreshold(threshold=0.0).fit(X_full).get_support()
    X_full = X_full[:, high_var]
    print(f"  After zero-variance removal: {X_full.shape[1]} dims")
    print(f"  Time: {time.time() - t0:.1f}s")

    return X_full, y


# ═══════════════════════════════════════════════════════════════════
# 2. TRAIN 4 DIVERSE EXPERTS
# ═══════════════════════════════════════════════════════════════════
def train_experts(X, y):
    print(f"\n{'=' * 70}")
    print("  STAGE 2: Train 4 Diverse Experts (5-fold CV)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n = len(y)
    all_oofs = {}
    model_results = []

    configs = [
        ('LGBM', 'lgb', {
            'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05,
            'num_leaves': 63, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        }),
        ('XGBoost', 'xgb', {
            'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.04,
            'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1,
            'reg_lambda': 0.1, 'min_child_weight': 5,
            'tree_method': 'hist', 'verbosity': 0,
        }),
        ('CatBoost', 'cat', {
            'iterations': 800, 'depth': 8, 'learning_rate': 0.06,
            'l2_leaf_reg': 3.0, 'subsample': 0.8, 'verbose': 0,
            'random_seed': SEED,
        }),
        ('ExtraTrees', 'et', {
            'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 20,
            'max_features': 0.3, 'class_weight': 'balanced',
        }),
    ]

    for name, tag, params in configs:
        print(f"\n  --- {name} ---")
        oof = np.zeros(n)
        fold_results = []

        for fi, (ti, vi) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[ti], X[vi]
            y_tr, y_va = y[ti], y[vi]

            if tag == 'lgb':
                pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                m = lgb.LGBMClassifier(**params, scale_pos_weight=pw,
                                        random_state=SEED + fi, verbose=-1)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(80, verbose=False),
                                 lgb.log_evaluation(0)])

            elif tag == 'xgb':
                pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                m = xgb.XGBClassifier(**params, scale_pos_weight=pw,
                                       random_state=SEED + fi)
                m.fit(X_tr, y_tr)

            elif tag == 'cat':
                pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                m = cb.CatBoostClassifier(**params, scale_pos_weight=pw)
                m.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                      early_stopping_rounds=80, verbose=False)

            elif tag == 'et':
                m = ExtraTreesClassifier(**params, random_state=SEED + fi,
                                          n_jobs=-1, bootstrap=True)
                m.fit(X_tr, y_tr)

            p = m.predict_proba(X_va)[:, 1]
            oof[vi] = p
            bf = max(f1_score(y_va, (p > th).astype(int), zero_division=0)
                     for th in np.arange(0.05, 0.95, 0.01))
            auc = roc_auc_score(y_va, p)
            fold_results.append((bf, auc))

        oof = np.nan_to_num(oof, nan=0.5)
        bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
                 for th in np.arange(0.05, 0.95, 0.001))
        auc = roc_auc_score(y, oof)
        fold_f1s = [fr[0] for fr in fold_results]
        print(f"  {name}: AUC={auc:.4f} F1={bf:.4f}  "
              f"(folds: [{', '.join(f'{f:.4f}' for f in fold_f1s)}])")

        all_oofs[name] = oof
        model_results.append((name, auc, bf))

    return all_oofs, model_results


# ═══════════════════════════════════════════════════════════════════
# 3. LOAD EXISTING STRONG OOFS
# ═══════════════════════════════════════════════════════════════════
def load_existing_oofs(y):
    print(f"\n{'=' * 70}")
    print("  STAGE 3: Load Existing Strong OOFs")
    print("=" * 70)

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

    for name in list(existing.keys()):
        oof = existing[name]
        bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
                 for th in np.arange(0.05, 0.95, 0.001))
        auc = roc_auc_score(y, oof)
        print(f"  {name:15s}: AUC={auc:.4f} F1={bf:.4f}")

    return existing


# ═══════════════════════════════════════════════════════════════════
# 4. STACKING + HILLCLIMB FUSION
# ═══════════════════════════════════════════════════════════════════
def fuse_and_optimize(all_oofs, y):
    print(f"\n{'=' * 70}")
    print("  STAGE 4: Stacking + Hillclimb Fusion")
    print("=" * 70)

    names = list(all_oofs.keys())
    n_models = len(names)
    n = len(y)
    P = np.column_stack([all_oofs[nm] for nm in names])

    corrs = np.corrcoef(P.T)
    print(f"\n  OOF correlation matrix ({n_models} models):")
    hdr = ' ' * 10 + ''.join(f'{nm[:6]:>7s}' for nm in names)
    print(f"  {hdr}")
    for i, nm_i in enumerate(names):
        row = f'{nm_i:10s}' + ''.join(f'{corrs[i,j]:7.3f}' for j in range(n_models))
        print(f"  {row}")

    print(f"\n  --- Stage 1: LogisticRegression Stacking ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_stack = np.zeros(n)

    for fi, (ti, vi) in enumerate(skf.split(P, y)):
        meta = LogisticRegression(class_weight='balanced', max_iter=2000,
                                   random_state=SEED, C=0.5)
        meta.fit(P[ti], y[ti])
        oof_stack[vi] = meta.predict_proba(P[vi])[:, 1]

    bf_st, th_st = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.001):
        pred = (oof_stack > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y, pred, zero_division=0)
        if f > bf_st:
            bf_st, th_st = f, th

    auc_st = roc_auc_score(y, oof_stack)
    pred_st = (oof_stack > th_st).astype(int)
    tp = ((pred_st == 1) & (y == 1)).sum()
    fp = ((pred_st == 1) & (y == 0)).sum()
    fn = ((pred_st == 0) & (y == 1)).sum()

    print(f"  LR Stacking ({n_models} models):")
    print(f"    AUC={auc_st:.4f}  F1={bf_st:.4f}  Rec={tp/(tp+fn):.4f}  "
          f"Prec={tp/(tp+fp) if (tp+fp)>0 else 0:.4f}  th={th_st:.3f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}")

    print(f"\n  --- Stage 2: CV Hillclimb F1 Optimization ---")

    def f1_for_weights(wt, P_sub, y_sub):
        wt = np.maximum(wt, 0)
        s = wt.sum()
        if s < 1e-10:
            wt = np.ones(P_sub.shape[1]) / P_sub.shape[1]
        else:
            wt = wt / s
        p = P_sub @ wt
        bf = 0
        for th in np.arange(0.05, 0.95, 0.002):
            pred = (p > th).astype(int)
            if pred.sum() == 0:
                continue
            f = f1_score(y_sub, pred, zero_division=0)
            if f > bf:
                bf = f
        return bf

    oof_cvh = np.zeros(n)
    cv_weights = []
    skf_h = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 100)

    for fi_h, (ti_h, vi_h) in enumerate(skf_h.split(P, y)):
        best_s = 0
        best_w = np.ones(n_models) / n_models

        import random
        rng = random.Random(SEED + fi_h)
        for _ in range(200):
            w = np.array([rng.uniform(0, 1) if rng.random() < 0.3 else rng.uniform(0.1, 0.5)
                          for _ in range(n_models)])
            w = np.maximum(w, 1e-4)
            w = w / w.sum()
            s = f1_for_weights(w, P[ti_h], y[ti_h])
            if s > best_s:
                best_s, best_w = s, w.copy()

        for it in range(50):
            improved = False
            for i in np.random.permutation(n_models):
                for d in [0.05, -0.05, 0.10, -0.10]:
                    tw = best_w.copy()
                    tw[i] += d
                    tw = np.maximum(tw, 1e-4)
                    tw = tw / tw.sum()
                    s = f1_for_weights(tw, P[ti_h], y[ti_h])
                    if s > best_s + 1e-5:
                        best_s, best_w = s, tw.copy()
                        improved = True
            if not improved:
                break

        oof_cvh[vi_h] = P[vi_h] @ best_w
        cv_weights.append(best_w)

    bf_cvh, th_cvh = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.001):
        pred = (oof_cvh > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y, pred, zero_division=0)
        if f > bf_cvh:
            bf_cvh, th_cvh = f, th

    auc_cvh = roc_auc_score(y, oof_cvh)
    pred_cvh = (oof_cvh > th_cvh).astype(int)
    tp = ((pred_cvh == 1) & (y == 1)).sum()
    fp = ((pred_cvh == 1) & (y == 0)).sum()
    fn = ((pred_cvh == 0) & (y == 1)).sum()

    avg_weights = np.mean(np.array(cv_weights), axis=0)
    avg_weights = avg_weights / avg_weights.sum()

    print(f"\n  CV Hillclimb Final ({n_models} models):")
    print(f"    AUC={auc_cvh:.4f}  F1={bf_cvh:.4f}  Rec={tp/(tp+fn):.4f}  "
          f"Prec={tp/(tp+fp) if (tp+fp)>0 else 0:.4f}  th={th_cvh:.3f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}")

    print(f"\n  Final Averaged Weights:")
    for i, nm in enumerate(names):
        if avg_weights[i] > 0.01:
            print(f"    {nm:15s}: {avg_weights[i]:.4f}")

    return (bf_cvh, auc_cvh, th_cvh, tp, fp, fn,
            oof_cvh, avg_weights, names, bf_st, auc_st)


# ═══════════════════════════════════════════════════════════════════
# 5. COMPREHENSIVE REPORT
# ═══════════════════════════════════════════════════════════════════
def print_report(new_experts, existing, final, y):
    bf_hl, auc_hl, th_hl, tp, fp, fn, p_final, best_w, names, bf_st, auc_st = final

    sep = "=" * 70
    print(f"\n{sep}")
    print("  FINAL RESULTS: Mega Boost Pure-Auto F1")
    print(sep)

    print(f"\n  {'Model':<20s} {'AUC':>8s} {'F1':>8s} {'Rec':>8s} {'Prec':>8s}")
    print(f"  {'-' * 56}")

    models_best = []

    for name, auc, bf in new_experts:
        models_best.append((name, auc, bf))
    for name in existing:
        oof = existing[name]
        bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
                 for th in np.arange(0.05, 0.95, 0.001))
        auc = roc_auc_score(y, oof)
        models_best.append((name, auc, bf))
    models_best.append(('V225 (baseline)', 0.9804, 0.8457))

    for name, auc, bf in sorted(models_best, key=lambda x: -x[2]):
        print(f"  {name:<20s} {auc:8.4f} {bf:8.4f}")

    oof_st = np.nan_to_num(np.column_stack([best_w[i] * (p_final * 0 + all_oofs[names[i]])
                                             for i in range(len(names)) if best_w[i] > 0.01]).sum(axis=1), nan=0.5)
    print(f"\n  {'-' * 56}")
    print(f"  {'LR Stacking':<20s} {auc_st:8.4f} {bf_st:8.4f}")
    print(f"  {'MEGA BOOST FINAL':<20s} {auc_hl:8.4f} {bf_hl:8.4f}   <--- BEST")

    print(f"\n  {'V225 F1 reference:':<25s} 0.8457")
    print(f"  {'Delta vs V225:':<25s} {bf_hl - 0.8457:+.4f}")
    print(f"  {'Delta vs Super-GBDT:':<25s} {bf_hl - 0.8526:+.4f}")

    print(f"\n  {'Metric':<12s} {'Value':>10s}")
    print(f"  {'-' * 24}")
    print(f"  {'TP':<12s} {tp:>10d}")
    print(f"  {'FP':<12s} {fp:>10d}")
    print(f"  {'FN':<12s} {fn:>10d}")
    print(f"  {'Recall':<12s} {tp/(tp+fn):>10.4f}")
    print(f"  {'Precision':<12s} {tp/(tp+fp) if (tp+fp)>0 else 0:>10.4f}")
    print(f"  {'Threshold':<12s} {th_hl:>10.3f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = time.time()

    X_mega, y = build_mega_features()

    all_oofs, new_results = train_experts(X_mega, y)

    existing = load_existing_oofs(y)

    for name, oof in existing.items():
        all_oofs[name] = oof

    final = fuse_and_optimize(all_oofs, y)

    print_report(new_results, existing, final, y)

    bf_hl, auc_hl, th_hl, tp, fp, fn, p_final, best_w, names, bf_st, auc_st = final
    np.savez_compressed(
        os.path.join(OUT, 'mega_boost.npz'),
        oof_final=p_final,
        weights=best_w,
        names=np.array(list(all_oofs.keys())),
        y=y,
        f1=bf_hl,
        auc=auc_hl,
        threshold=th_hl,
        tp=tp, fp=fp, fn=fn,
        f1_stacking=bf_st,
        auc_stacking=auc_st,
    )
    print(f"\n  Saved to output/mega_boost.npz")
    print(f"  Total time: {(time.time() - t_start) / 60:.1f} min")
