"""
Expert D: CPU-efficient unsupervised anomaly detection expert for SGCC.

Extracts simple per-user features from the multi-channel sequence X_seq and
computes anomaly scores using Isolation Forest, LOF, and k-NN distance.
Designed to run entirely on CPU and provide a complementary "third view" to
the supervised GBDT and TCN experts.
"""
import os
import time
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

warnings.filterwarnings('ignore')

from config import SEED, N_FOLDS, OUTPUT_DIR


def _extract_sequence_features(X_seq):
    """Extract fast vectorized features from multi-channel sequence [N, C, T]."""
    X_seq = np.asarray(X_seq, dtype=np.float32)
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)

    n, c, t = X_seq.shape
    feats = []

    # Channel-wise basic stats
    for ch in range(c):
        x = X_seq[:, ch, :]
        feats.append(np.mean(x, axis=1))
        feats.append(np.std(x, axis=1))
        feats.append(np.median(x, axis=1))
        feats.append(np.min(x, axis=1))
        feats.append(np.max(x, axis=1))
        feats.append(np.percentile(x, 25, axis=1))
        feats.append(np.percentile(x, 75, axis=1))

        # Trend: second half vs first half
        half = t // 2
        feats.append(np.mean(x[:, half:], axis=1) - np.mean(x[:, :half], axis=1))

        # Zero ratio
        feats.append(np.mean(x == 0, axis=1))

        # Autocorrelation at lag 1, 7, 30
        for lag in [1, 7, 30]:
            if t > lag:
                xm = x - np.mean(x, axis=1, keepdims=True)
                num = np.mean(xm[:, lag:] * xm[:, :-lag], axis=1)
                den = np.mean(xm ** 2, axis=1) + 1e-12
                feats.append(num / den)
            else:
                feats.append(np.zeros(n))

    # Cross-channel interactions
    for i in range(c):
        for j in range(i + 1, c):
            xi = X_seq[:, i, :]
            xj = X_seq[:, j, :]
            feats.append(np.mean(xi * xj, axis=1))
            feats.append(np.mean((xi - xj) ** 2, axis=1))

    feat_matrix = np.column_stack(feats)
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    feat_matrix = np.clip(feat_matrix, -1e4, 1e4)
    return feat_matrix


def _isf_score(model, X):
    """Isolation Forest decision_function: lower = more anomalous."""
    return -model.decision_function(X)


def _lof_score(model, X):
    """LOF decision_function: lower = more anomalous."""
    return -model.decision_function(X)


def _knn_score(model, X):
    """Distance to k-th nearest neighbor: higher = more anomalous."""
    dists, _ = model.kneighbors(X)
    return dists[:, -1]


class ExpertDUnsupervisedTrainer:
    """5-fold unsupervised anomaly expert for SGCC."""

    def __init__(self, dataset='sgcc', n_neighbors=20, n_estimators=200,
                 contamination=0.1):
        if dataset != 'sgcc':
            raise ValueError("ExpertDUnsupervisedTrainer currently supports only 'sgcc'")
        self.dataset = dataset
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.contamination = contamination

    def train(self, X_seq, labels):
        print("=" * 60)
        print("Expert D: Unsupervised Anomaly Detection (SGCC)")
        print("=" * 60)

        t0 = time.time()
        print("\n  Extracting sequence features...")
        X_feat = _extract_sequence_features(X_seq)
        print(f"  Feature matrix: {X_feat.shape}")

        n = len(labels)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        oof_isf = np.zeros(n)
        oof_lof = np.zeros(n)
        oof_knn = np.zeros(n)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_feat, labels)):
            print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
            X_train = X_feat[train_idx]
            X_val = X_feat[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Isolation Forest
            print("    Fitting Isolation Forest...")
            isf = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=SEED + fold_idx,
                n_jobs=-1,
            )
            isf.fit(X_train_s)
            oof_isf[val_idx] = _isf_score(isf, X_val_s)

            # LOF
            print("    Fitting LOF...")
            lof = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1,
            )
            lof.fit(X_train_s)
            oof_lof[val_idx] = _lof_score(lof, X_val_s)

            # k-NN distance
            print("    Fitting k-NN...")
            knn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1)
            knn.fit(X_train_s)
            oof_knn[val_idx] = _knn_score(knn, X_val_s)

        # Normalize each score to [0, 1] across all folds
        def _norm01(x):
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min)
            return np.zeros_like(x)

        oof_isf_n = _norm01(oof_isf)
        oof_lof_n = _norm01(oof_lof)
        oof_knn_n = _norm01(oof_knn)

        # Combine scores
        oof_combined = (oof_isf_n + oof_lof_n + oof_knn_n) / 3.0

        # Metrics
        def _best_f1(y_true, y_prob):
            best_f1, best_th = 0, 0.5
            for th in np.arange(0.05, 0.95, 0.01):
                pred = (y_prob > th).astype(int)
                if pred.sum() == 0:
                    continue
                f = f1_score(y_true, pred, zero_division=0)
                if f > best_f1:
                    best_f1, best_th = f, th
            return best_f1, best_th

        f1_isf, th_isf = _best_f1(labels, oof_isf_n)
        f1_lof, th_lof = _best_f1(labels, oof_lof_n)
        f1_knn, th_knn = _best_f1(labels, oof_knn_n)
        f1_comb, th_comb = _best_f1(labels, oof_combined)

        print(f"\n[Expert D SGCC] Results:")
        print(f"  ISF  F1={f1_isf:.4f} th={th_isf:.3f} AUC={roc_auc_score(labels, oof_isf_n):.4f}")
        print(f"  LOF  F1={f1_lof:.4f} th={th_lof:.3f} AUC={roc_auc_score(labels, oof_lof_n):.4f}")
        print(f"  kNN  F1={f1_knn:.4f} th={th_knn:.3f} AUC={roc_auc_score(labels, oof_knn_n):.4f}")
        print(f"  Comb F1={f1_comb:.4f} th={th_comb:.3f} AUC={roc_auc_score(labels, oof_combined):.4f}")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed/60:.1f} min")

        # Save all scores
        out_path = os.path.join(OUTPUT_DIR, 'sgcc_expert_d_unsupervised.npz')
        np.savez_compressed(
            out_path,
            oof_isf=oof_isf_n,
            oof_lof=oof_lof_n,
            oof_knn=oof_knn_n,
            oof_combined=oof_combined,
            flags=labels,
        )
        print(f"  Saved to {out_path}")

        return oof_combined


def train_unsupervised_sgcc(X_seq, flags):
    """Backward-compatible wrapper."""
    trainer = ExpertDUnsupervisedTrainer(dataset='sgcc')
    return trainer.train(X_seq, flags)


if __name__ == '__main__':
    print("Run expert_d_unsupervised.py through run_expert_d_unsupervised.py")
