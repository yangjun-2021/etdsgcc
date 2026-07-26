import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def best_f1_score(y_true, proba, th_grid=None):
    if th_grid is None:
        th_grid = np.arange(0.05, 0.95, 0.005)
    best_th = 0.5
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0
    for th in th_grid:
        pred = (proba > th).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_recall = recall_score(y_true, pred, zero_division=0)
            best_precision = precision_score(y_true, pred, zero_division=0)
    return best_f1, best_th, best_recall, best_precision


def best_f1_recall_constrained(y_true, proba, min_recall=0.90, th_grid=None):
    if th_grid is None:
        th_grid = np.arange(0.02, 0.95, 0.005)
    best_f1 = 0.0
    best_th = 0.5
    best_recall = 0.0
    best_precision = 0.0
    for th in th_grid:
        pred = (proba > th).astype(int)
        recall = recall_score(y_true, pred, zero_division=0)
        if recall < min_recall:
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_recall = recall
            best_precision = precision_score(y_true, pred, zero_division=0)
    return best_f1, best_th, best_recall, best_precision


def evaluate_binary(y_true, proba, prefix=''):
    auc = roc_auc_score(y_true, proba)
    f1, th, rec, prec = best_f1_score(y_true, proba)
    pred = (proba > th).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    if prefix:
        print(f'{prefix}')
    print(f'  AUC={auc:.4f}  F1={f1:.4f}  Recall={rec:.4f}  Precision={prec:.4f}  th={th:.3f}')
    print(f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}')
    return {'auc': auc, 'f1': f1, 'recall': rec, 'precision': prec, 'threshold': th,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}


def evaluate_binary_cv(fold_metrics, prefix=''):
    keys = ['auc', 'f1', 'recall', 'precision']
    print(f'{prefix}')
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        mean_ = np.mean(vals)
        std_ = np.std(vals, ddof=1)
        print(f'  {k.capitalize():>10s}: {mean_:.4f} ± {std_:.4f}  (range: [{np.min(vals):.4f}, {np.max(vals):.4f}])')
    all_auc = [m['auc'] for m in fold_metrics]
    all_f1 = [m['f1'] for m in fold_metrics]
    return {'auc_mean': np.mean(all_auc), 'auc_std': np.std(all_auc, ddof=1),
            'f1_mean': np.mean(all_f1), 'f1_std': np.std(all_f1, ddof=1)}


def make_meta_features(stat_features, miss_ratio, oof_proba_a, oof_proba_b):
    return np.column_stack([
        stat_features,
        miss_ratio,
        oof_proba_a.reshape(-1, 1),
        oof_proba_b.reshape(-1, 1),
        np.abs(oof_proba_a - oof_proba_b).reshape(-1, 1),
    ])
