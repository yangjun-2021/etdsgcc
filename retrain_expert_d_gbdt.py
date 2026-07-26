"""Train a CPU-efficient GBDT expert on flattened 5-channel time series + stats."""
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import lightgbm as lgb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS


def best_f1(y, p):
    best = (0, 0.5)
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (p > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y, pred)
        if f > best[0]:
            best = (f, th)
    return best


def main():
    print("Loading preprocessed data...")
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
    X_seq = pre['X_seq']  # [N, 5, T]
    stat_features = pre['stat_features']
    flags = pre['flags']
    impute_mask = pre['impute_mask']

    n = len(flags)
    print(f"  X_seq={X_seq.shape}, stat_features={stat_features.shape}")

    # Flatten time series
    X_flat = X_seq.reshape(n, -1)
    miss_ratio = impute_mask.mean(axis=1).reshape(-1, 1)
    X = np.hstack([X_flat, stat_features, miss_ratio])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"  Expert D feature matrix: {X.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n)

    for fi, (ti, vi) in enumerate(skf.split(X, flags)):
        print(f"\n  Fold {fi+1}/{N_FOLDS}")
        pw = (flags[ti] == 0).sum() / max((flags[ti] == 1).sum(), 1)
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            max_depth=8,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.05,
            min_child_samples=50,
            scale_pos_weight=pw,
            random_state=SEED + fi,
            verbose=-1,
        )
        model.fit(
            X[ti], flags[ti],
            eval_set=[(X[vi], flags[vi])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)]
        )
        oof[vi] = model.predict_proba(X[vi])[:, 1]
        f1, th = best_f1(flags[vi], oof[vi])
        print(f"    Val F1={f1:.4f} th={th:.3f}")

    f1, th = best_f1(flags, oof)
    print(f"\nExpert D overall: F1={f1:.4f} th={th:.3f} "
          f"Rec={recall_score(flags, (oof>th).astype(int)):.4f} "
          f"Prec={precision_score(flags, (oof>th).astype(int)):.4f} "
          f"AUC={roc_auc_score(flags, oof):.4f}")

    save_path = os.path.join(OUTPUT_DIR, 'expert_d_gbdt_oof.npz')
    np.savez_compressed(save_path, oof_proba=oof, flags=flags)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
