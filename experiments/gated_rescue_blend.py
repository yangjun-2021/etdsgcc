"""Gated rescue blend: learn a per-fold gate for hillclimb FN and mix with high-recall signal.

Uses extended features + OOF signals to predict whether hillclimb misses a positive sample.
Then blends hillclimb with a high-recall signal, weighted by the gate probability.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import lightgbm as lgb

seed_everything(SEED)


def load_oof(fname, key_guess):
    fpath = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(fpath):
        return None
    d = np.load(fpath)
    for k in [key_guess] + [kk for kk in d.keys() if kk.startswith('oof_')]:
        if k in d:
            arr = d[k]
            if arr.ndim > 1:
                arr = arr[:, 1] if arr.shape[1] == 2 else arr.ravel()
            return np.nan_to_num(arr.astype(np.float64), nan=0.5)
    return None


def best_f1(y, p):
    best = (0, 0, 0, 0)
    for th in np.linspace(0.01, 0.99, 199):
        pred = (p >= th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y, pred, zero_division=0)
        if f > best[0]:
            best = (f, recall_score(y, pred), precision_score(y, pred), th)
    return best


def main():
    y = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags'].astype(int)
    ext = np.load(os.path.join(OUTPUT_DIR, 'sgcc_extended_features.npz'))['features'].astype(np.float32)
    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage'].astype(np.float32)
    miss = usage['missing_rate'].astype(np.float32)

    signals = {
        'hc': load_oof('hillclimb_best_oof.npz', 'oof_hillclimb'),
        'amst_v3': load_oof('amst_3ch_synthetic_mixed_ls_v3_oof.npz', 'oof_amst_3ch_synthetic_mixed_ls_v3'),
        'supcon_v3': load_oof('supcon_raw_3ch_v3_oof.npz', 'oof_supcon_raw_3ch_v3'),
        'mega_meta': load_oof('mega_meta_all_oofs_oof.npz', 'oof_mega_meta_all_oofs'),
        'informer_large': load_oof('informer_large_strong_prior_oof.npz', 'oof_informer_large_strong_prior'),
        'patch_recall': load_oof('patch_transformer_raw_3ch_recall_oof.npz', 'oof_patch_transformer_raw_3ch_recall'),
    }
    signals = {k: v for k, v in signals.items() if v is not None}
    print('Loaded signals:', list(signals.keys()))

    hc = signals['hc']
    # hillclimb baseline threshold
    _, _, _, th_hc = best_f1(y, hc)
    pred_hc = (hc >= th_hc).astype(int)

    # Build base feature matrix: extended features + usage stats
    base_features = np.column_stack([ext, log_max, miss])
    print('Base features shape:', base_features.shape)

    # Gate target: hillclimb FN
    gate_target = ((y == 1) & (pred_hc == 0)).astype(int)
    print(f'Hillclimb FN gate target positives: {gate_target.sum()} / {len(y)}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gate_oof = np.zeros(len(y))

    for fi, (ti, vi) in enumerate(skf.split(base_features, y)):
        # Features include base + OOF signals from training fold? Using OOF signals directly leaks target.
        # Use only base features for gate; the rescue signal will be applied after.
        X_tr, X_val = base_features[ti], base_features[vi]
        g_tr, g_val = gate_target[ti], gate_target[vi]
        dtrain = lgb.Dataset(X_tr, label=g_tr)
        dval = lgb.Dataset(X_val, label=g_val, reference=dtrain)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'seed': SEED + fi,
        }
        m = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        gate_oof[vi] = m.predict(X_val, num_iteration=m.best_iteration)
        auc = roc_auc_score(g_val, gate_oof[vi])
        print(f'  Fold {fi+1} gate AUC={auc:.4f}')

    print(f'Overall gate AUC={roc_auc_score(gate_target, gate_oof):.4f}')

    # Blend search: for each rescue signal, beta weight, threshold
    print('\nGated blend search:')
    best_overall = (0, None)
    for rescue_name, p_rescue in signals.items():
        if rescue_name == 'hc':
            continue
        for beta in np.arange(0.1, 1.01, 0.1):
            w = beta * gate_oof
            w = np.clip(w, 0, 1)
            p_blend = (1 - w) * hc + w * p_rescue
            f1, rec, prec, th = best_f1(y, p_blend)
            if f1 > best_overall[0]:
                best_overall = (f1, (rescue_name, beta, th, rec, prec, p_blend.copy()))
            # Also top-k hard gating variant
            k = 0.2
            topk = (gate_oof >= np.percentile(gate_oof, (1 - k) * 100))
            p_topk = hc.copy()
            p_topk[topk] = ((1 - beta) * hc + beta * p_rescue)[topk]
            f1k, reck, preck, thk = best_f1(y, p_topk)
            if f1k > best_overall[0]:
                best_overall = (f1k, (rescue_name + '_topk', beta, thk, reck, preck, p_topk.copy()))

    f1, info = best_overall
    name, beta, th, rec, prec, p_best = info
    auc = roc_auc_score(y, p_best)
    print(f'Best gated blend: {name} beta={beta:.1f} th={th:.3f}')
    print(f'  F1={f1:.4f} Rec={rec:.4f} Prec={prec:.4f} AUC={auc:.4f}')

    # Save
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'gated_rescue_blend_oof.npz'),
        flags=y,
        oof_gated_rescue_blend=p_best,
        best_name=name,
        best_beta=float(beta),
        best_th=float(th),
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "gated_rescue_blend_oof.npz")}')


if __name__ == '__main__':
    main()
