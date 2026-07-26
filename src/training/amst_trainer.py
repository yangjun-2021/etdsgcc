"""
AMSTTrainer: Training loop for AMST-Net on SGCC/OEDI.

Supports:
- 5-fold stratified CV with OOF predictions
- Diffusion-based data augmentation (DiffAug)
- Hard-Negative Supervised Contrastive Learning (HN-SupCon)
- Co-Teaching (two networks cross-supervised for label noise)
- Integration with existing Expert-A (GBDT) OOF probabilities

Example:
    from src.training.amst_trainer import AMSTTrainer
    trainer = AMSTTrainer(dataset='sgcc', use_diffaug=True, use_supcon=True)
    oof = trainer.train(X_seq, labels, impute_mask=impute_mask, oof_proba_a=oof_proba_a)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

from config import SGCC_CONFIG, OEDI_CONFIG, SEED, N_FOLDS, OUTPUT_DIR, DEVICE
from src.models.amst_net import AMSTNet, HardNegativeSupConLoss
from src.models.models import RecallOrientedFocalLoss, GeneralizedCrossEntropyLoss, RecallOrientedGCELoss
from src.data.ts_augment import TSAugment, mixup_augment
from src.data.synthetic_anomalies import SyntheticAnomalyAugmenter


class AMSTTrainer:
    """Unified trainer for AMST-Net."""

    def __init__(self, dataset='sgcc', use_diffaug=True, use_supcon=True,
                 use_coteaching=False, forget_rate=0.2, num_gradual=10,
                 epochs=100, batch_size=64, lr=1e-4, weight_decay=1e-4,
                 patience=20, device=DEVICE, use_prior=True,
                 d_mamba=128, d_trans=256, d_freq=64, proj_dim=128,
                 n_mamba_layers=2, n_trans_layers=4, n_heads=8, dropout=0.2,
                 use_branch_attention=True, d_fusion=128,
                 branch_attn_heads=4, branch_attn_layers=1,
                 synthetic_ratio=2.0, pos_weight=None, label_smoothing=0.0,
                 focal_alpha=0.75, focal_gamma=2.0, recall_weight=3.0,
                 loss_type='focal', gce_q=0.7,
                 use_weighted_sampler=False,
                 use_tsa=False, tsa_methods=None, tsa_prob=0.5,
                 tsa_severity=0.1, tsa_copies=1, use_mixup=False, mixup_alpha=0.2,
                 use_amp=False,
                 use_synthetic_anomalies=False,
                 synthetic_anomaly_kwargs=None):
        self.dataset = dataset
        self.config = SGCC_CONFIG if dataset == 'sgcc' else OEDI_CONFIG
        self.dataset_name = self.config['name']
        self.use_diffaug = use_diffaug
        self.use_supcon = use_supcon
        self.use_coteaching = use_coteaching
        self.use_prior = use_prior
        self.forget_rate = forget_rate
        self.num_gradual = num_gradual
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        # AMP speeds up training but is disabled with SupCon (fp16 matmul risk)
        self.use_amp = use_amp and (not self.use_supcon) and torch.cuda.is_available()

        self.d_mamba = d_mamba
        self.d_trans = d_trans
        self.d_freq = d_freq
        self.proj_dim = proj_dim
        self.n_mamba_layers = n_mamba_layers
        self.n_trans_layers = n_trans_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_branch_attention = use_branch_attention
        self.d_fusion = d_fusion
        self.branch_attn_heads = branch_attn_heads
        self.branch_attn_layers = branch_attn_layers

        # Data augmentation / class-balance hyperparameters
        self.synthetic_ratio = synthetic_ratio
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.recall_weight = recall_weight
        self.loss_type = loss_type
        self.gce_q = gce_q
        self.use_weighted_sampler = use_weighted_sampler
        # Lightweight time-series augmentation (TSA)
        self.use_tsa = use_tsa
        self.tsa_methods = tsa_methods
        self.tsa_prob = tsa_prob
        self.tsa_severity = tsa_severity
        self.tsa_copies = tsa_copies
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # PeerJ-style synthetic anomaly injection
        self.use_synthetic_anomalies = use_synthetic_anomalies
        self.synthetic_anomaly_kwargs = synthetic_anomaly_kwargs or {}

        self.tcn_params = self.config['tcn_params']
        self.train_params = self.config['train_params']

        # SupCon loss weight (relative to classification loss)
        self.lambda_supcon = 0.2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    @staticmethod
    def _find_best_threshold(y_true, proba):
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

    # ------------------------------------------------------------------
    # Diffusion augmentation (if enabled)
    # ------------------------------------------------------------------
    def _augment(self, X_train, y_train):
        # 1) Lightweight time-series augmentation (TSA) - cheap and usually safer
        if self.use_tsa:
            print(f"[AMSTTrainer] TSAug: methods={self.tsa_methods}, copies={self.tsa_copies}, prob={self.tsa_prob}, severity={self.tsa_severity}")
            tsa = TSAugment(
                methods=self.tsa_methods,
                prob=self.tsa_prob,
                severity=self.tsa_severity,
                seed=SEED,
            )
            X_train, y_train = tsa.fit_transform(X_train, y_train, n_aug_per_sample=self.tsa_copies)
            print(f"[AMSTTrainer] After TSAug: {X_train.shape}, theft rate={y_train.mean()*100:.2f}%")

        # 2) Mixup (same-label interpolation)
        if self.use_mixup:
            rng = np.random.RandomState(SEED)
            X_train, y_train = mixup_augment(X_train, y_train, alpha=self.mixup_alpha, rng=rng)
            print(f"[AMSTTrainer] After Mixup: {X_train.shape}, theft rate={y_train.mean()*100:.2f}%")

        # 3) PeerJ-style synthetic anomaly injection (cheap, class-balancing)
        if self.use_synthetic_anomalies:
            kwargs = dict(seed=SEED)
            kwargs.update(self.synthetic_anomaly_kwargs)
            # Default: generate enough synthetic anomalies to double the positives
            n_pos = int(y_train.sum())
            if 'n_synthetic' not in kwargs:
                kwargs['n_synthetic'] = n_pos
            print(f"[AMSTTrainer] SyntheticAnomalies: generating {kwargs['n_synthetic']} samples, kwargs={kwargs}")
            aug = SyntheticAnomalyAugmenter(**kwargs)
            X_syn, y_syn = aug.fit_transform(X_train, y_train)
            X_train = np.concatenate([X_train, X_syn], axis=0)
            y_train = np.concatenate([y_train, y_syn], axis=0)
            print(f"[AMSTTrainer] After SyntheticAnomalies: {X_train.shape}, theft rate={y_train.mean()*100:.2f}%")

        # 4) Diffusion augmentation (expensive; only if enabled and enough samples)
        if not self.use_diffaug:
            return X_train, y_train
        try:
            from src.data.diffusion_augment import DiffAugmenter
        except ImportError:
            print("[AMSTTrainer] DiffAug not available, skipping augmentation.")
            return X_train, y_train

        n_theft = int(y_train.sum())
        n_synthetic = int(self.synthetic_ratio * n_theft)
        n_synthetic = max(n_synthetic, n_theft)  # at least double the positives
        if n_theft < 50:
            print("[AMSTTrainer] Too few theft samples for DiffAug, skipping.")
            return X_train, y_train

        # Use the full multi-channel input for diffusion so synthetic samples have
        # realistic channels, not just value channel + mean-filled others.
        nC = X_train.shape[1] if X_train.ndim == 3 else 1
        print(f"[AMSTTrainer] DiffAug: generating {n_synthetic} synthetic theft samples from {n_theft} real thefts (multi-channel, C={nC})...")
        aug = DiffAugmenter(
            n_synthetic=n_synthetic,
            T=X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1],
            in_channels=nC,
            epochs=30,  # fast; can increase to 100 for final runs
            batch_size=64,
            device=self.device,
        )
        X_syn, y_syn = aug.fit_transform(X_train, y_train)

        # Combine
        X_aug = np.concatenate([X_train, X_syn], axis=0)
        y_aug = np.concatenate([y_train, y_syn], axis=0)
        print(f"[AMSTTrainer] Augmented train set: {X_aug.shape}, theft rate={y_aug.mean()*100:.2f}%")
        return X_aug, y_aug

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    def _build_loaders(self, X, y, oof_a=None, prior=None, shuffle=True):
        """Build DataLoader with optional prior probabilities from Expert-A."""
        tensors = [torch.FloatTensor(X), torch.LongTensor(y)]
        if prior is not None:
            tensors.append(torch.FloatTensor(prior))
            dataset = TensorDataset(*tensors)
        else:
            dataset = TensorDataset(*tensors)

        sampler = None
        if shuffle and self.use_weighted_sampler:
            class_counts = np.bincount(y.astype(int))
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights[y.astype(int)]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.FloatTensor(sample_weights),
                num_samples=len(y),
                replacement=True,
            )
            shuffle = False  # mutually exclusive with sampler

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                            sampler=sampler, num_workers=0, drop_last=shuffle)
        return loader

    # ------------------------------------------------------------------
    # Training step (single network)
    # ------------------------------------------------------------------
    def _train_single_network(self, model, train_loader, val_loader, y_val,
                              epochs, lr, weight_decay, patience, fold_idx):
        model.to(self.device)
        if self.loss_type == 'gce':
            cls_criterion = RecallOrientedGCELoss(
                q=self.gce_q,
                recall_weight=self.recall_weight,
            ).to(self.device)
        elif self.loss_type == 'gce_plain':
            cls_criterion = GeneralizedCrossEntropyLoss(
                q=self.gce_q,
            ).to(self.device)
        else:
            cls_criterion = RecallOrientedFocalLoss(
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                recall_weight=self.recall_weight,
            ).to(self.device)
        if self.pos_weight is not None:
            bce_pos_weight = torch.tensor([self.pos_weight], dtype=torch.float32).to(self.device)
        else:
            bce_pos_weight = None
        supcon_criterion = HardNegativeSupConLoss(
            temperature=0.07, hard_neg_k=5
        ).to(self.device) if self.use_supcon else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        best_val_f1 = 0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0
            for batch in train_loader:
                if len(batch) == 3:
                    x, y, prior = batch
                    x, y, prior = x.to(self.device), y.to(self.device), prior.to(self.device)
                else:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    prior = None

                optimizer.zero_grad()

                # Model forward with optional Expert-A prior
                model_kwargs = {}
                if self.use_prior and prior is not None:
                    model_kwargs['prior'] = prior

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.use_supcon:
                        logit, z = model(x, return_embedding=True, **model_kwargs)
                        cls_loss = cls_criterion(logit, y.float(), pos_weight=bce_pos_weight, label_smoothing=self.label_smoothing)
                        supcon_loss = supcon_criterion(z, y)
                        loss = cls_loss + self.lambda_supcon * supcon_loss
                    else:
                        logit = model(x, **model_kwargs)
                        cls_loss = cls_criterion(logit, y.float(), pos_weight=bce_pos_weight, label_smoothing=self.label_smoothing)
                        loss = cls_loss

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

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
                print(f"    Fold {fold_idx+1} Epoch {epoch+1}: val_F1={val_f1:.4f} "
                      f"(best={best_val_f1:.4f}) rec={val_rec:.4f} prec={val_prec:.4f}")

            if patience_counter >= patience:
                print(f"    Fold {fold_idx+1} early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def _predict_proba(self, model, loader):
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                model_kwargs = {}
                if self.use_prior and len(batch) == 3:
                    model_kwargs['prior'] = batch[2].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logit = model(x, **model_kwargs)
                probs.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(probs, axis=0)

    # ------------------------------------------------------------------
    # Main training loop with 5-fold OOF
    # ------------------------------------------------------------------
    def train(self, X_seq, labels, impute_mask=None, oof_proba_a=None,
              fold_assignments=None):
        print("=" * 70)
        print(f"AMST-Net Training ({self.dataset_name.upper()})")
        print(f"  DiffAug={self.use_diffaug}, TSA={self.use_tsa}, Mixup={self.use_mixup}, SupCon={self.use_supcon}, CoTeaching={self.use_coteaching}, Prior={self.use_prior and oof_proba_a is not None}")
        print(f"  BranchAttention={self.use_branch_attention}, d_fusion={self.d_fusion}, branch_heads={self.branch_attn_heads}")
        print("=" * 70)

        n = len(labels)
        X_seq = np.asarray(X_seq, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if oof_proba_a is not None:
            oof_proba_a = np.asarray(oof_proba_a, dtype=np.float32)

        splits = self._get_fold_splits(labels, fold_assignments)
        oof_proba = np.zeros(n, dtype=np.float32)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{len(splits)}")
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            prior_train = oof_proba_a[train_idx] if oof_proba_a is not None else None
            prior_val = oof_proba_a[val_idx] if oof_proba_a is not None else None

            # Augment training fold only
            X_train_aug, y_train_aug = self._augment(X_train, y_train)
            if prior_train is not None:
                # Expand prior for synthetic samples (use Expert-A mean for synthetic)
                prior_train_aug = np.concatenate([
                    prior_train,
                    np.full(len(y_train_aug) - len(y_train), prior_train.mean(), dtype=np.float32)
                ])
            else:
                prior_train_aug = None

            train_loader = self._build_loaders(X_train_aug, y_train_aug, prior=prior_train_aug, shuffle=True)
            val_loader = self._build_loaders(X_val, y_val, prior=prior_val, shuffle=False)

            model = AMSTNet(
                in_channels=X_seq.shape[1],
                seq_len=X_seq.shape[2],
                d_mamba=self.d_mamba,
                d_trans=self.d_trans,
                d_freq=self.d_freq,
                proj_dim=self.proj_dim,
                n_mamba_layers=self.n_mamba_layers,
                n_trans_layers=self.n_trans_layers,
                n_heads=self.n_heads,
                dropout=self.dropout,
                use_freq=True,
                use_supcon=self.use_supcon,
                prior_dim=1 if self.use_prior and oof_proba_a is not None else 0,
                use_branch_attention=self.use_branch_attention,
                d_fusion=self.d_fusion,
                branch_attn_heads=self.branch_attn_heads,
                branch_attn_layers=self.branch_attn_layers,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Model params: {n_params:,}")

            model = self._train_single_network(
                model, train_loader, val_loader, y_val,
                epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
                patience=self.patience, fold_idx=fold_idx
            )

            val_proba = self._predict_proba(model, val_loader)
            oof_proba[val_idx] = val_proba

            f1, th, rec, prec = self._find_best_threshold(y_val, val_proba)
            print(f"  Fold {fold_idx+1} val: F1={f1:.4f}, th={th:.3f}, rec={rec:.4f}, prec={prec:.4f}")

            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, f'{self.dataset_name}_amst_fold{fold_idx}.pt'))

        # Overall metrics
        overall_f1, overall_th, overall_rec, overall_prec = self._find_best_threshold(labels, oof_proba)
        overall_auc = roc_auc_score(labels, oof_proba)
        print(f"\n[AMST-Net {self.dataset_name.upper()}] Overall: "
              f"F1={overall_f1:.4f}, th={overall_th:.3f}, rec={overall_rec:.4f}, "
              f"prec={overall_prec:.4f}, AUC={overall_auc:.4f}")

        # Save OOF
        label_key = 'flags' if self.dataset == 'sgcc' else 'y'
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f'{self.dataset_name}_amst.npz'),
            oof_proba=oof_proba,
            **{label_key: labels},
        )

        return oof_proba


if __name__ == '__main__':
    print("Run AMSTTrainer through the main pipeline or import as module.")
