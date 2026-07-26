"""
Expert C: Informer-based temporal expert for electricity theft detection.

Trains an Informer with ProbSparse self-attention on the multi-channel
sequence input and produces out-of-fold probabilities to feed into the
meta-learner as a third expert.
"""
import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import SEED, N_FOLDS, OUTPUT_DIR, DEVICE
from src.models.informer_model import train_informer, predict_informer


class ExpertCTrainer:
    """Informer trainer compatible with the SGCC/OEDI pipeline."""

    def __init__(self, dataset='sgcc', d_model=128, n_heads=8, num_layers=3,
                 dropout=0.3, epochs=40, batch_size=64, lr=3e-4):
        self.dataset = dataset
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def _get_fold_splits(self, labels, fold_assignments=None):
        if self.dataset == 'sgcc':
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            return list(skf.split(np.zeros(len(labels)), labels))
        else:
            unique_folds = np.unique(fold_assignments)
            splits = []
            for fold_idx in unique_folds:
                train_idx = np.where(fold_assignments != fold_idx)[0]
                val_idx = np.where(fold_assignments == fold_idx)[0]
                splits.append((train_idx, val_idx))
            return splits

    def _find_best_threshold(self, y_true, proba):
        best_f1, best_th = 0.0, 0.5
        for th in np.arange(0.1, 0.9, 0.005):
            pred = (proba > th).astype(int)
            if pred.sum() == 0:
                continue
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        pred = (proba > best_th).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        return best_f1, best_th, rec, prec

    def train(self, X_seq, labels, oof_proba_a=None, fold_assignments=None):
        """Train Informer across folds and return OOF probabilities."""
        print("=" * 60)
        print(f"Expert C: Informer Training ({self.dataset.upper()})")
        print("=" * 60)

        n = len(labels)
        splits = self._get_fold_splits(labels, fold_assignments)
        oof_proba_c = np.zeros(n, dtype=np.float32)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
            torch.cuda.empty_cache()

            X_tr = X_seq[train_idx]
            y_tr = labels[train_idx]
            prior_tr = oof_proba_a[train_idx].astype(np.float32) if oof_proba_a is not None else None

            model = train_informer(
                X_tr, y_tr,
                oof_prior=prior_tr,
                d_model=self.d_model,
                n_heads=self.n_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=DEVICE,
                seed=SEED + fold_idx,
                verbose=True,
            )

            prior_val = oof_proba_a[val_idx].astype(np.float32) if oof_proba_a is not None else None
            val_probs = predict_informer(model, X_seq[val_idx], prior_val, device=DEVICE)
            val_probs = np.nan_to_num(val_probs, nan=0.5)
            oof_proba_c[val_idx] = val_probs

            f1, th, rec, prec = self._find_best_threshold(labels[val_idx], val_probs)
            print(f"  Fold {fold_idx+1} Best: F1={f1:.4f}, Recall={rec:.4f}, "
                  f"Precision={prec:.4f}, th={th:.3f}")

            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, f'{self.dataset}_informer_fold{fold_idx}.pt'))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Overall metrics
        f1, th, rec, prec = self._find_best_threshold(labels, oof_proba_c)
        auc = roc_auc_score(labels, oof_proba_c)
        print(f"\n[Expert C {self.dataset.upper()}] Overall: F1={f1:.4f}, "
              f"Recall={rec:.4f}, Precision={prec:.4f}, AUC={auc:.4f}, th={th:.3f}")

        label_key = 'flags' if self.dataset == 'sgcc' else 'y'
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f'{self.dataset}_expert_c.npz'),
            oof_proba=oof_proba_c,
            **{label_key: labels},
        )
        return oof_proba_c
