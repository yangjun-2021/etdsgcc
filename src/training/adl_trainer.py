"""
ADLTrainer: training loop for ADL-Net on SGCC/OEDI.

Supports:
- 5-fold stratified CV with OOF predictions
- Dictionary reconstruction (unsupervised on all users)
- Supervised binary classification with Focal loss
- MoCo-style contrastive learning on augmented views
- No dependency on any external model prior
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

from config import SEED, N_FOLDS, OUTPUT_DIR, DEVICE
from src.models.adl_net import ADLNet, focal_loss
from src.utils.utils import seed_everything
from src.data.ts_augment import TSAugment


class ADLTrainer:
    """Unified trainer for ADL-Net."""

    def __init__(self, dataset='sgcc', n_atoms=256, sparsity=0.1,
                 d_model=256, n_layers=4, n_heads=8, dropout=0.2,
                 proj_dim=128, queue_size=4096, temperature=0.07,
                 epochs=100, batch_size=64, lr=1e-4, weight_decay=1e-4,
                 patience=15, device=DEVICE,
                 lambda_recon=1.0, lambda_cls=1.0, lambda_contrast=0.5,
                 focal_alpha=0.75, focal_gamma=2.0, label_smoothing=0.0,
                 pos_weight=None, use_weighted_sampler=False,
                 use_tsa=True, tsa_methods=None, tsa_prob=0.5, tsa_severity=0.1):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device

        self.n_atoms = n_atoms
        self.sparsity = sparsity
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.proj_dim = proj_dim
        self.queue_size = queue_size
        self.temperature = temperature

        self.lambda_recon = lambda_recon
        self.lambda_cls = lambda_cls
        self.lambda_contrast = lambda_contrast
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.use_weighted_sampler = use_weighted_sampler

        self.use_tsa = use_tsa
        self.tsa_methods = tsa_methods or ['jitter', 'scale', 'permutation']
        self.tsa_prob = tsa_prob
        self.tsa_severity = tsa_severity

    def _get_fold_splits(self, labels, fold_assignments=None):
        if fold_assignments is not None:
            unique_folds = np.unique(fold_assignments)
            splits = []
            for fold_idx in unique_folds:
                train_idx = np.where(fold_assignments != fold_idx)[0]
                val_idx = np.where(fold_assignments == fold_idx)[0]
                splits.append((train_idx, val_idx))
            return splits
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return list(skf.split(np.zeros(len(labels)), labels))

    def _find_best_threshold(self, y_true, proba):
        best_f1, best_th = 0, 0.5
        best_rec, best_prec = 0, 0
        for th in np.arange(0.05, 0.95, 0.005):
            pred = (proba > th).astype(int)
            if pred.sum() == 0:
                continue
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_rec = recall_score(y_true, pred, zero_division=0)
                best_prec = precision_score(y_true, pred, zero_division=0)
        return best_f1, best_th, best_rec, best_prec

    def _build_loader(self, X, y, batch_size, shuffle, weighted=False):
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).long()
        ds = TensorDataset(X_t, y_t)
        if weighted and shuffle:
            weights = np.where(y == 1, 1.0 / (y.sum() + 1e-8), 1.0 / (len(y) - y.sum() + 1e-8))
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(ds, batch_size=batch_size, sampler=sampler)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _augment_views(self, X):
        """Generate two augmented views of X for contrastive learning."""
        if not self.use_tsa:
            return X, X
        tsa = TSAugment(methods=self.tsa_methods, prob=self.tsa_prob, severity=self.tsa_severity, seed=SEED)
        X_v1, _ = tsa.fit_transform(X, np.zeros(len(X)))
        tsa2 = TSAugment(methods=self.tsa_methods, prob=self.tsa_prob, severity=self.tsa_severity, seed=SEED + 1)
        X_v2, _ = tsa2.fit_transform(X, np.zeros(len(X)))
        return X_v1, X_v2

    def _train_single_network(self, model, train_loader, val_loader, y_val, fold_idx):
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        pos_w = torch.tensor([self.pos_weight], dtype=torch.float32).to(self.device) if self.pos_weight is not None else None

        best_val_f1 = 0
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device).float()

                # Generate two augmented views
                x_np = x.cpu().numpy()
                x_v1_np, x_v2_np = self._augment_views(x_np)
                x_v1 = torch.from_numpy(x_v1_np).float().to(self.device)
                x_v2 = torch.from_numpy(x_v2_np).float().to(self.device)

                optimizer.zero_grad()
                logit, contrast_loss = model(x, x_v2)
                cls_loss = focal_loss(logit, y, alpha=self.focal_alpha, gamma=self.focal_gamma,
                                      pos_weight=pos_w, label_smoothing=self.label_smoothing)

                # Reconstruction loss: only on normal samples (y==0) to learn normal dictionary
                with torch.no_grad():
                    _, _, x_hat, residual, code, _ = model(x, return_embedding=True)
                recon = (x.view(x.size(0), -1) - x_hat).pow(2).mean(dim=1)
                recon_loss = (recon * (1 - y)).sum() / ((1 - y).sum() + 1e-8)

                loss = self.lambda_cls * cls_loss + self.lambda_recon * recon_loss
                if contrast_loss is not None and self.lambda_contrast > 0:
                    loss = loss + self.lambda_contrast * contrast_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Keep dictionary atoms normalized
                model.dictionary.normalize_atoms()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            val_proba = self._predict_proba(model, val_loader)
            val_f1, _, val_rec, val_prec = self._find_best_threshold(y_val, val_proba)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Fold {fold_idx + 1} Epoch {epoch + 1}: val_F1={val_f1:.4f} "
                      f"(best={best_val_f1:.4f}) rec={val_rec:.4f} prec={val_prec:.4f}")

            if patience_counter >= self.patience:
                print(f"    Fold {fold_idx + 1} early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _predict_proba(self, model, loader):
        model.eval()
        probs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                logit, _ = model(x)
                probs.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def train(self, X_seq, labels, fold_assignments=None):
        print("=" * 70)
        print(f"ADL-Net Training ({self.dataset.upper()})")
        print(f"  atoms={self.n_atoms}, d_model={self.d_model}, layers={self.n_layers}, "
              f"lambda_recon={self.lambda_recon}, lambda_contrast={self.lambda_contrast}")
        print("=" * 70)

        n = len(labels)
        X_seq = np.asarray(X_seq, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        splits = self._get_fold_splits(labels, fold_assignments)
        oof_proba = np.zeros(n, dtype=np.float32)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{len(splits)}")
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]

            train_loader = self._build_loader(X_train, y_train, self.batch_size, shuffle=True, weighted=self.use_weighted_sampler)
            val_loader = self._build_loader(X_val, y_val, self.batch_size, shuffle=False, weighted=False)

            model = ADLNet(
                in_channels=X_seq.shape[1],
                seq_len=X_seq.shape[2],
                n_atoms=self.n_atoms,
                sparsity=self.sparsity,
                d_model=self.d_model,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                dropout=self.dropout,
                proj_dim=self.proj_dim,
                queue_size=self.queue_size,
                temperature=self.temperature,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Model params: {n_params:,}")

            model = self._train_single_network(model, train_loader, val_loader, y_val, fold_idx)
            val_proba = self._predict_proba(model, val_loader)
            oof_proba[val_idx] = val_proba

            f1, th, rec, prec = self._find_best_threshold(y_val, val_proba)
            print(f"  Fold {fold_idx + 1} val: F1={f1:.4f}, th={th:.3f}, rec={rec:.4f}, prec={prec:.4f}")

            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, f'{self.dataset}_adl_fold{fold_idx}.pt'))

        overall_f1, overall_th, overall_rec, overall_prec = self._find_best_threshold(labels, oof_proba)
        overall_auc = roc_auc_score(labels, oof_proba)
        print(f"\n[ADL-Net {self.dataset.upper()}] Overall: "
              f"F1={overall_f1:.4f}, th={overall_th:.3f}, rec={overall_rec:.4f}, "
              f"prec={overall_prec:.4f}, AUC={overall_auc:.4f}")

        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f'{self.dataset}_adl.npz'),
            oof_proba=oof_proba, flags=labels,
        )
        return oof_proba


if __name__ == '__main__':
    print("Run ADLTrainer through the main pipeline or import as module.")
