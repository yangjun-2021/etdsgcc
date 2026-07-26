"""Rebuild the §2.5.2 error-risk model per the 4-group 22-dim spec + group ablation.

The original implementation is lost (only manuscript prose survived). This script
rebuilds it faithfully:
- labels: whether the final fusion model errs at theta=0.5643 (final_blend_best OOF)
- 22 features in 4 groups (see GROUPS below, per scripts/fix_expert_issues.py itemization)
- 5-fold OOF LightGBM classifier (AUC equivalent to ranking quality)
- ablation: drop-one-group and single-group AUC

Usage:
    conda run -n ml python experiments/risk_model_rebuild.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything

seed_everything(SEED)
TH = 0.5643


def build_features(X, p):
    """X: [N, T] raw daily consumption; p: [N] final-model probability."""
    N, T = X.shape
    eps = 1e-12
    feats = {}

    # G1 预测概率 (3)
    feats['g1_maxprob'] = np.maximum(p, 1 - p)
    feats['g1_margin'] = np.abs(p - 0.5)
    feats['g1_entropy'] = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))

    # G2 时序统计 (7)：近期（后90天）
    r = X[:, -90:]
    feats['g2_mean90'] = r.mean(axis=1)
    feats['g2_var90'] = r.var(axis=1)
    tgrid = np.arange(90)
    tm = tgrid - tgrid.mean()
    feats['g2_trend90'] = (r * tm).sum(axis=1) / (tm * tm).sum()
    m, s = r.mean(axis=1), r.std(axis=1) + eps
    z = (r - m[:, None]) / s[:, None]
    feats['g2_kurt90'] = (z ** 4).mean(axis=1) - 3
    feats['g2_skew90'] = (z ** 3).mean(axis=1)
    feats['g2_cv90'] = s / (np.abs(m) + eps)
    feats['g2_dayweek'] = X[:, -7:].mean(axis=1) / (X[:, -28:].mean(axis=1) + eps)

    # G3 深度行为 (6)
    xmax = np.abs(X).max(axis=1) + eps
    d = np.diff(X, axis=1)
    feats['g3_flat'] = (np.abs(d) < 0.01 * xmax[:, None]).mean(axis=1)
    sd = d.std(axis=1) + eps
    feats['g3_jump'] = (np.abs(d) > 3 * sd[:, None]).mean(axis=1)
    zero = (X <= eps).astype(int)
    runlen = np.zeros(N)
    for i in range(N):
        c = np.concatenate(([0], zero[i], [0]))
        starts = np.where(np.diff(c) == 1)[0]
        ends = np.where(np.diff(c) == -1)[0]
        runlen[i] = (ends - starts).max() if len(starts) else 0
    feats['g3_zerorun'] = runlen
    xc = X - X.mean(axis=1, keepdims=True)
    denom = (xc ** 2).sum(axis=1) + eps
    feats['g3_autocorr'] = (xc[:, 1:] * xc[:, :-1]).sum(axis=1) / denom
    mu = X.mean(axis=1, keepdims=True)
    sd_all = X.std(axis=1, keepdims=True) + eps
    feats['g3_cusum'] = np.abs(np.cumsum(X - mu, axis=1)).max(axis=1) / (sd_all.squeeze() * np.sqrt(T))
    mo = X.reshape(N, -1, 30) if T % 30 == 0 else None
    if mo is None:
        seg = np.array_split(X, 34, axis=1)
    else:
        seg = [mo[:, i, :] for i in range(mo.shape[1])]
    cvs = [(sg.std() / (np.abs(sg.mean()) + eps)).mean() if sg.size else 0 for sg in seg]
    feats['g3_monthcv'] = np.mean([ (s.std(axis=1) / (np.abs(s.mean(axis=1)) + eps)) for s in seg], axis=0)

    # G4 周相关与频域 (6)
    w = X[:, : T // 7 * 7].reshape(N, -1, 7) if T >= 7 else X[:, :, None]
    wd = w[:, :, :5].mean(axis=(1, 2)) if w.shape[2] >= 5 else w.mean(axis=(1, 2))
    we = w[:, :, 5:7].mean(axis=(1, 2)) if w.shape[2] >= 7 else w.mean(axis=(1, 2))
    feats['g4_wdwe'] = wd / (we + eps)
    dow = w.mean(axis=1)  # [N, 7]
    feats['g4_dowmax'] = dow.max(axis=1) / (dow.mean(axis=1) + eps)
    mo_agg = X.reshape(N, -1, 30).mean(axis=2) if T % 30 == 0 else np.array([s.mean(axis=1) for s in np.array_split(X, 34, axis=1)]).T
    tm2 = np.arange(mo_agg.shape[1]) - (mo_agg.shape[1] - 1) / 2
    feats['g4_motrend'] = (mo_agg * tm2).sum(axis=1) / (tm2 * tm2).sum()
    F = np.abs(np.fft.rfft(X - X.mean(axis=1, keepdims=True), axis=1)) ** 2
    tot = F.sum(axis=1) + eps
    k = F.shape[1]
    feats['g4_fftlow'] = F[:, : max(2, k // 10)].sum(axis=1) / tot
    feats['g4_ffthigh'] = F[:, k // 2 :].sum(axis=1) / tot
    feats['g4_weratio'] = we / (w.mean(axis=(1, 2)) + eps)

    names = list(feats.keys())
    M = np.column_stack([feats[k] for k in names])
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0), names


GROUPS = {
    'G1预测概率': [0, 1, 2],
    'G2时序统计': [3, 4, 5, 6, 7, 8, 9],
    'G3深度行为': [10, 11, 12, 13, 14, 15],
    'G4周频域': [16, 17, 18, 19, 20, 21],
}


def oof_auc(M, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(M, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                               num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=pw, random_state=SEED + fi, verbose=-1)
        m.fit(M[ti], y[ti], eval_set=[(M[vi], y[vi])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[vi] = m.predict_proba(M[vi])[:, 1]
    return roc_auc_score(y, oof), oof


def main():
    X = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))['X_seq'][:, 0, :].astype(np.float64)
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y = cl['y_orig'].astype(int)
    fb = np.load(os.path.join(OUTPUT_DIR, 'final_blend_best_oof.npz'))
    p = fb['oof_final_blend_best'].astype(np.float64)

    pred = (p > TH).astype(int)
    y_err = (pred != y).astype(int)
    print(f'error rate at th={TH}: {y_err.mean() * 100:.2f}% ({y_err.sum()} errors)')

    M, names = build_features(X, p)
    print(f'feature matrix: {M.shape}')

    auc_full, oof_full = oof_auc(M, y_err)
    print(f'\nFULL 22-dim: AUC={auc_full:.4f}')

    print('\nDrop-one-group:')
    for g, idx in GROUPS.items():
        keep = [i for i in range(M.shape[1]) if i not in idx]
        auc, _ = oof_auc(M[:, keep], y_err)
        print(f'  drop {g:10s}: AUC={auc:.4f}  (Δ={auc - auc_full:+.4f})')

    print('\nSingle-group:')
    for g, idx in GROUPS.items():
        auc, _ = oof_auc(M[:, idx], y_err)
        print(f'  only {g:10s}: AUC={auc:.4f}')

    np.savez_compressed(os.path.join(OUTPUT_DIR, 'risk_model_rebuilt_oof.npz'),
                        oof_risk_rebuilt=oof_full, y_err=y_err, feature_names=np.array(names))
    print('\nsaved output/risk_model_rebuilt_oof.npz')


if __name__ == '__main__':
    main()
