"""
Mega Boost Enhanced: Extra OOFs + Extra Features
==================================================
Adds:
  - 7 new OOFs (V71 lgb/xgb/cat/tcn/innov + V229 iso/platt)
  - ~60 new temporal features (monthly, weekly, multi-res PAA, seasonal)
  - 3 new expert models for extra diversity

Target: Pure-auto F1 > 0.8609
"""
import os, time, glob, warnings
import numpy as np, pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from utils import seed_everything

SEED, N_FOLDS, OUT = 42, 5, 'output'
OD = r'D:\Project\ThiefElectricity\output'
seed_everything(SEED)


def load_base_data():
    """Load base data and compute all features."""
    print("=" * 70)
    print("  Loading base data & building enhanced features")
    print("=" * 70)
    t0 = time.time()

    base = np.load(os.path.join(OUT, 'sgcc_preprocessed.npz'))
    y = base['flags'].astype(int)
    n = len(y)
    stat_feat = np.nan_to_num(base['stat_features'], nan=0, posinf=0, neginf=0)
    residuals = base['residuals']
    impute_mask = base['impute_mask']
    n_days = residuals.shape[1]
    half = n_days // 2

    novel = np.load(os.path.join(OUT, 'novel_features.npz'))
    novel_feat = np.nan_to_num(novel['features'], nan=0)

    deng = np.load(os.path.join(OUT, 'dengine_features.npz'))
    deng_feat = np.nan_to_num(deng['X'], nan=0)

    raw_df = pd.read_csv('data/raw_data.csv')
    date_cols = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = raw_df[date_cols].values.astype(float)
    del raw_df
    filled = np.nan_to_num(raw, nan=0)

    print(f"  Base loaded: {stat_feat.shape[1]} stat + {novel_feat.shape[1]} novel "
          f"+ {deng_feat.shape[1]} deng  |  {n_days} days")

    # ─── PAA multi-resolution ───
    all_paa = []
    for n_seg in [25, 50, 100]:
        seg_size = n_days / n_seg
        paa = np.zeros((n, n_seg), dtype=np.float32)
        for i in range(n_seg):
            s = int(round(i * seg_size))
            e = int(round((i + 1) * seg_size))
            if e > s:
                paa[:, i] = np.nanmean(raw[:, s:e], axis=1)
            elif s < n_days:
                paa[:, i] = raw[:, s]
        all_paa.append(np.nan_to_num(paa, nan=0))
    paa_feat = np.column_stack(all_paa)
    print(f"  PAA (25+50+100={paa_feat.shape[1]}): ✓")

    # ─── Residuals aggregation ───
    res_feats = [
        np.nanmean(residuals, axis=1),
        np.nanstd(residuals, axis=1),
        np.nanmean(np.abs(residuals), axis=1),
        np.nanmax(np.abs(residuals), axis=1),
        np.nanpercentile(residuals, 25, axis=1),
        np.nanpercentile(residuals, 75, axis=1),
        np.nanpercentile(residuals, 90, axis=1),
        np.nanpercentile(residuals, 95, axis=1),
    ]
    r1 = np.nanmean(residuals[:, :half], axis=1)
    r2 = np.nanmean(residuals[:, half:], axis=1)
    res_feats.append((r2 - r1) / (np.maximum(np.abs(r1), 1e-6)))
    for w in [30, 60, 90, 180, 365]:
        if n_days >= w:
            res_feats.append(np.nanmean(np.abs(residuals[:, -w:]), axis=1))
    res_agg = np.nan_to_num(np.column_stack([r.reshape(-1, 1) for r in res_feats]), nan=0)
    print(f"  Residuals agg ({res_agg.shape[1]}): ✓")

    # ─── Mask aggregation ───
    mask_feats = [impute_mask.astype(float).mean(axis=1)]
    for ss, se in [(0, n_days//4), (n_days//4, n_days//2),
                    (n_days//2, 3*n_days//4), (3*n_days//4, n_days),
                    (0, half), (half, n_days)]:
        mask_feats.append(impute_mask[:, ss:se].astype(float).mean(axis=1))
    mr = np.zeros(n)
    for i in range(n):
        runs, cr = [], 0
        for m in impute_mask[i]:
            if m: cr += 1
            else:
                if cr > 0: runs.append(cr); cr = 0
        if cr > 0: runs.append(cr)
        mr[i] = max(runs) if runs else 0
    mask_feats.append(mr)
    mask_feats.append((~impute_mask).sum(axis=1) / n_days)
    miss_persist = np.zeros(n)
    for i in range(n):
        mi = impute_mask[i].astype(float)
        if mi.sum() > 0:
            miss_persist[i] = (mi[1:] * mi[:-1]).sum() / max(mi.sum(), 1)
    mask_feats.append(miss_persist)
    mask_agg = np.nan_to_num(np.column_stack([m.reshape(-1, 1) for m in mask_feats]), nan=0)
    print(f"  Mask agg ({mask_agg.shape[1]}): ✓")

    # ─── NEW: Monthly consumption profiles ───
    month_of_day = np.array([j % 365 for j in range(n_days)])
    month_idx = np.digitize(month_of_day, np.linspace(0, 365, 13)) - 1
    month_feats = np.zeros((n, 12), dtype=np.float32)
    for m in range(12):
        mask_m = (month_idx == m)
        if mask_m.sum() > 0:
            month_feats[:, m] = np.nanmean(filled[:, mask_m], axis=1)
    month_feats = np.nan_to_num(month_feats, nan=0)
    print(f"  Monthly profiles ({month_feats.shape[1]}): ✓")

    # ─── NEW: Day-of-week profiles ───
    dow_of_day = np.array([j % 7 for j in range(n_days)])
    dow_feats = np.zeros((n, 7), dtype=np.float32)
    for d in range(7):
        mask_d = (dow_of_day == d)
        if mask_d.sum() > 0:
            dow_feats[:, d] = np.nanmean(filled[:, mask_d], axis=1)
    dow_feats = np.nan_to_num(dow_feats, nan=0)
    print(f"  Day-of-week profiles ({dow_feats.shape[1]}): ✓")

    # ─── NEW: Short-term trend features ───
    trend_feats = []
    for w in [7, 14, 30, 60, 90]:
        if n_days >= w:
            first = filled[:, :w].mean(axis=1)
            last = filled[:, -w:].mean(axis=1)
            trend_feats.append((last - first) / (np.maximum(np.abs(first), 1e-6)))
            trend_feats.append(np.nanmean(np.diff(filled[:, -w:], axis=1), axis=1))
    trend_agg = np.nan_to_num(np.column_stack([t.reshape(-1, 1) for t in trend_feats]), nan=0)
    print(f"  Short-term trends ({trend_agg.shape[1]}): ✓")

    # ─── NEW: Seasonal decomposition residual stats ───
    seasonal_feats = []
    for period, name in [(7, 'weekly'), (30, 'monthly'), (90, 'quarterly')]:
        if n_days >= period * 3:
            for offset in range(period):
                idx = np.arange(offset, n_days, period)
                if len(idx) > 0:
                    val = np.nanmean(filled[:, idx], axis=1) - np.nanmean(filled, axis=1)
                    seasonal_feats.append(val)
    seas_agg = np.nan_to_num(np.column_stack([s.reshape(-1, 1) for s in seasonal_feats]), nan=0)
    print(f"  Seasonal decomposition ({seas_agg.shape[1]}): ✓")

    # ─── NEW: Quantile features across time ───
    quant_feats = []
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        quant_feats.append(np.nanpercentile(filled, q, axis=1))
    quant_agg = np.nan_to_num(np.column_stack([q.reshape(-1, 1) for q in quant_feats]), nan=0)
    print(f"  Global quantiles ({quant_agg.shape[1]}): ✓")

    # ─── Assemble ───
    X_full = np.column_stack([
        stat_feat, novel_feat, deng_feat, paa_feat, res_agg, mask_agg,
        month_feats, dow_feats, trend_agg, seas_agg, quant_agg,
    ])
    X_full = np.nan_to_num(X_full, nan=0, posinf=0, neginf=0)
    X_full = np.clip(X_full, -1e4, 1e4).astype(np.float32)
    print(f"\n  TOTAL features: {X_full.shape[1]} dims")
    print(f"  Time: {time.time() - t0:.1f}s")

    return X_full, y


def load_all_oofs(y):
    """Load ALL available OOFs (existing 11 + V71 5 + V229 2 = 18)."""
    print(f"\n{'=' * 70}")
    print("  Loading all OOFs")
    print("=" * 70)
    all_oofs = {}

    our = np.load(os.path.join(OUT, 'tcn_kd_results.npz'))
    for k, key in [('TCN+KD', 'oof_tcn_kd'), ('TCN-stacker', 'oof_stacker'),
                    ('TCN-blend', 'oof_blend'), ('TCN-hill', 'oof_hill')]:
        if key in our.files:
            all_oofs[k] = our[key]

    all_oofs['Super-GBDT'] = np.load(os.path.join(OUT, 'super_gbdt.npz'))['oof_super']
    all_oofs['SmartBlend'] = np.load(os.path.join(OUT, 'smart_blend.npz'))['oof_final']

    multi = np.load(os.path.join(OUT, 'multi_oof_results.npz'))
    for k in ['oof_lgb', 'oof_xgb', 'oof_cb', 'oof_tcn', 'oof_stack', 'oof_final']:
        if k in multi.files:
            all_oofs['multi_' + k.replace('oof_', '')] = multi[k]

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
                all_oofs[tag] = d[key]

    for prefix, keys_, fmt in [
        ('v71_oofs_', ['lgb', 'xgb', 'cat', 'tcn', 'innov'], 'V71_{}'),
        ('v229_results_', ['oof_iso', 'oof_platt'], 'V229_{}'),
    ]:
        files = sorted(glob.glob(os.path.join(OD, f'{prefix}*.npz')), reverse=True)
        if files:
            d = np.load(files[0], allow_pickle=True)
            for k in keys_:
                if k in d.files:
                    all_oofs[fmt.format(k)] = d[k]

    for name in list(all_oofs.keys()):
        oof = all_oofs[name]
        if len(oof) != len(y):
            del all_oofs[name]
            continue
        bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
                 for th in np.arange(0.05, 0.95, 0.01))
        auc = roc_auc_score(y, oof)
        print(f"  {name:18s}: F1={bf:.4f} AUC={auc:.4f}")

    print(f"\n  Total OOFs loaded: {len(all_oofs)}")
    return all_oofs


def train_new_experts(X, y):
    """Train 4 new diverse experts on enhanced features."""
    print(f"\n{'=' * 70}")
    print("  Training new experts on enhanced features")
    print("=" * 70)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n = len(y)
    new_oofs = {}

    configs = [
        ('LGB', lambda: lgb.LGBMClassifier(n_estimators=800, max_depth=7,
            learning_rate=0.05, num_leaves=63, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=0.1, verbose=-1)),
        ('XGB', lambda: xgb.XGBClassifier(n_estimators=500, max_depth=6,
            learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
            tree_method='hist', verbosity=0)),
        ('CatB', lambda: cb.CatBoostClassifier(iterations=500, depth=8,
            learning_rate=0.06, l2_leaf_reg=3.0, verbose=0)),
        ('RF', lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
            min_samples_leaf=20, max_features=0.3, class_weight='balanced',
            n_jobs=-1)),
    ]

    for name, factory in configs:
        oof = np.zeros(n)
        for fi, (ti, vi) in enumerate(skf.split(X, y)):
            clf = factory()
            pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
            if name == 'LGB':
                clf.set_params(scale_pos_weight=pw, random_state=SEED + fi)
                clf.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                        callbacks=[lgb.early_stopping(150, verbose=False),
                                   lgb.log_evaluation(0)])
            elif name == 'XGB':
                clf.set_params(scale_pos_weight=pw, random_state=SEED + fi)
                clf.fit(X[ti], y[ti])
            elif name == 'CatB':
                clf.set_params(random_seed=SEED + fi)
                clf.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]),
                        early_stopping_rounds=80, verbose=False)
            elif name in ('RF', 'HistGB'):
                clf.set_params(random_state=SEED + fi)
                clf.fit(X[ti], y[ti])
            oof[vi] = clf.predict_proba(X[vi])[:, 1]
        oof = np.nan_to_num(oof, nan=0.5)
        new_oofs[name] = oof
        bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
                 for th in np.arange(0.05, 0.95, 0.01))
        auc = roc_auc_score(y, oof)
        print(f"  {name:6s}: F1={bf:.4f} AUC={auc:.4f}")

    return new_oofs


def meta_learn_and_evaluate(all_oofs, y):
    """Stack all OOFs with multiple meta-learners and find best."""
    print(f"\n{'=' * 70}")
    print(f"  Meta-Learner Stacking ({len(all_oofs)} OOFs)")
    print("=" * 70)

    names = list(all_oofs.keys())
    P = np.column_stack([all_oofs[nm] for nm in names])
    n = len(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {}

    for meta_name, factory in [
        ('XGB_meta', lambda pw: xgb.XGBClassifier(n_estimators=300, max_depth=3,
            learning_rate=0.05, scale_pos_weight=pw, tree_method='hist',
            verbosity=0, random_state=SEED)),
        ('LR_C=1.0', lambda pw: LogisticRegression(C=1.0, class_weight='balanced',
            max_iter=2000, random_state=SEED)),
        ('LR_C=0.5', lambda pw: LogisticRegression(C=0.5, class_weight='balanced',
            max_iter=2000, random_state=SEED)),
        ('LGB_meta', lambda pw: lgb.LGBMClassifier(n_estimators=300, max_depth=3,
            learning_rate=0.05, scale_pos_weight=pw, verbose=-1, random_state=SEED)),
        ('HistGB_meta', lambda pw: HistGradientBoostingClassifier(max_iter=200,
            max_depth=3, learning_rate=0.05, random_state=SEED)),
        ('Avg', None),
    ]:
        if meta_name == 'Avg':
            oof = P.mean(axis=1)
        else:
            oof = np.zeros(n)
            for fi, (ti, vi) in enumerate(skf.split(P, y)):
                pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
                m = factory(pw)
                if meta_name == 'LGB_meta':
                    m.fit(P[ti], y[ti], eval_set=[(P[vi], y[vi])],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(0)])
                else:
                    m.fit(P[ti], y[ti])
                oof[vi] = m.predict_proba(P[vi])[:, 1]

        best_f1, best_th = 0, 0.5
        for th in np.arange(0.05, 0.95, 0.001):
            pred = (oof > th).astype(int)
            if pred.sum() == 0: continue
            f = f1_score(y, pred, zero_division=0)
            if f > best_f1: best_f1, best_th = f, th

        auc = roc_auc_score(y, oof)
        pred_final = (oof > best_th).astype(int)
        tp = ((pred_final == 1) & (y == 1)).sum()
        fp = ((pred_final == 1) & (y == 0)).sum()
        fn = ((pred_final == 0) & (y == 1)).sum()

        results[meta_name] = {
            'f1': best_f1, 'auc': auc, 'th': best_th,
            'tp': tp, 'fp': fp, 'fn': fn,
            'oof': oof,
        }
        print(f"  {meta_name:14s}: F1={best_f1:.4f} AUC={auc:.4f}  "
              f"Rec={tp/(tp+fn):.4f} Prec={tp/(tp+fp) if tp+fp>0 else 0:.4f}")

    return results, names


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = time.time()

    X_enhanced, y = load_base_data()
    new_experts = train_new_experts(X_enhanced, y)
    existing_oofs = load_all_oofs(y)

    all_oofs = {**new_experts, **existing_oofs}
    print(f"\n  Grand total: {len(all_oofs)} OOFs ({len(new_experts)} new + "
          f"{len(existing_oofs)} existing)")

    results, names = meta_learn_and_evaluate(all_oofs, y)

    best = max(results.keys(), key=lambda k: results[k]['f1'])
    r = results[best]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  FINAL: Mega Boost Enhanced")
    print(f"{sep}")
    print(f"  Features: {X_enhanced.shape[1]} dims")
    print(f"  OOFs: {len(all_oofs)}")
    print(f"  Best meta: {best}")
    print(f"  F1={r['f1']:.4f}  AUC={r['auc']:.4f}  th={r['th']:.3f}")
    print(f"  Rec={r['tp']/(r['tp']+r['fn']):.4f}  "
          f"Prec={r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp']>0 else 0:.4f}")
    print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")

    print(f"\n  COMPARISON TABLE:")
    print(f"  {'Model':<22s} {'F1':>8s} {'AUC':>8s}")
    print(f"  {'-' * 40}")
    print(f"  {'V225 (baseline)':<22s} {'0.8457':>8s} {'0.9804':>8s}")
    print(f"  {'Super-GBDT':<22s} {'0.8527':>8s} {'0.9870':>8s}")
    print(f"  {'Mega Boost v1 (LR)':<22s} {'0.8602':>8s} {'0.9846':>8s}")
    print(f"  {'Mega Boost v1 (XGB)':<22s} {'0.8609':>8s} {'0.9871':>8s}")
    print(f"  {'ENHANCED (best)':<22s} {r['f1']:>8.4f} {r['auc']:>8.4f}")
    print(f"\n  vs V225: {r['f1'] - 0.8457:+.4f}")
    print(f"  vs Super-GBDT: {r['f1'] - 0.8527:+.4f}")
    print(f"  vs Mega Boost v1: {r['f1'] - 0.8609:+.4f}")

    np.savez_compressed(
        os.path.join(OUT, 'mega_boost_enhanced.npz'),
        oof_final=r['oof'], y=y, f1=r['f1'], auc=r['auc'],
        threshold=r['th'], tp=r['tp'], fp=r['fp'], fn=r['fn'],
        n_features=X_enhanced.shape[1], n_oofs=len(all_oofs),
        names=np.array(list(all_oofs.keys())),
    )
    print(f"\n  Saved to output/mega_boost_enhanced.npz")
    print(f"  Total time: {(time.time() - t_start) / 60:.1f} min")
