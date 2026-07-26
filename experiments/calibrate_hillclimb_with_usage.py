"""Calibrate hillclimb probabilities with usage/missing features via small MLP."""
import os
import sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED, N_FOLDS


def main():
    d = np.load(os.path.join(OUTPUT_DIR, 'hillclimb_best_oof.npz'))
    proba = d['oof_hillclimb'].astype(np.float64).reshape(-1, 1)
    y = d['flags'].astype(int)

    usage = np.load(os.path.join(OUTPUT_DIR, 'usage_features.npz'))
    log_max = usage['log_max_usage'].reshape(-1, 1)
    miss = usage['missing_rate'].reshape(-1, 1)

    X = np.hstack([proba, log_max, miss])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        # Small MLP to avoid overfitting
        m = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=SEED + fi,
        )
        m.fit(X[ti], y[ti])
        oof[vi] = m.predict_proba(X[vi])[:, 1]

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (oof > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    pred = (oof > best_th).astype(int)
    print(f'MLP calibration (proba+log_max+miss): F1={f1_score(y, pred):.4f}, '
          f'Rec={recall_score(y, pred):.4f}, '
          f'Prec={precision_score(y, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y, oof):.4f}, th={best_th:.3f}')

    # Compare to raw hillclimb
    raw_f1, raw_th, raw_rec, raw_prec = 0, 0.5, 0, 0
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (proba.squeeze() > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > raw_f1:
            raw_f1, raw_th, raw_rec, raw_prec = f1, th, recall_score(y, pred, zero_division=0), precision_score(y, pred, zero_division=0)
    print(f'Raw hillclimb: F1={raw_f1:.4f}, Rec={raw_rec:.4f}, Prec={raw_prec:.4f}, th={raw_th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'hillclimb_usage_mlp_calib.npz'),
        oof_hillclimb_usage_mlp=oof,
        flags=y,
    )


if __name__ == '__main__':
    main()
