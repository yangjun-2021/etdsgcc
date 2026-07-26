import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import lightgbm as lgb
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import DEVICE, N_FOLDS, OUTPUT_DIR, SEED
from src.models.foundation_encoder import (
    AsymmetricFocalLoss,
    TimeSeriesFoundationModel,
    masked_reconstruction_loss,
)
from src.utils.utils import seed_everything


class FoundationTrainer:
    def __init__(self, dataset='sgcc', pretrain_epochs=20, finetune_epochs=60,
                 batch_size=64, lr=1e-4, weight_decay=1e-4, patience=12,
                 patch_len=30, stride=15, d_model=128, n_layers=4, n_heads=8,
                 dropout=0.2, mask_ratio=0.35, lambda_recon=0.3,
                 lambda_consistency=0.1, focal_alpha=0.75, gamma_pos=1.0,
                 gamma_neg=3.0, use_weighted_sampler=True, device=DEVICE):
        self.dataset = dataset
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.mask_ratio = mask_ratio
        self.lambda_recon = lambda_recon
        self.lambda_consistency = lambda_consistency
        self.focal_alpha = focal_alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.use_weighted_sampler = use_weighted_sampler
        self.device = device

    def _get_fold_splits(self, labels, fold_assignments=None):
        if fold_assignments is not None:
            splits = []
            for fold in np.unique(fold_assignments):
                train_idx = np.where(fold_assignments != fold)[0]
                val_idx = np.where(fold_assignments == fold)[0]
                splits.append((train_idx, val_idx))
            return splits
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return list(skf.split(np.zeros(len(labels)), labels))

    def _loader(self, X, y=None, shuffle=True, weighted=False):
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if y is None:
            y_t = torch.zeros(len(X_t), dtype=torch.long)
        else:
            y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
        ds = TensorDataset(X_t, y_t)
        if weighted and y is not None:
            y_arr = np.asarray(y)
            pos = max((y_arr == 1).sum(), 1)
            neg = max((y_arr == 0).sum(), 1)
            weights = np.where(y_arr == 1, 1.0 / pos, 1.0 / neg)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(ds, batch_size=self.batch_size, sampler=sampler, drop_last=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def _create_model(self, X_seq):
        return TimeSeriesFoundationModel(
            in_channels=X_seq.shape[1],
            seq_len=X_seq.shape[2],
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(self.device)

    @staticmethod
    def _best_threshold(y_true, proba):
        best_f1, best_th, best_rec, best_prec = 0, 0.5, 0, 0
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

    def _pretrain(self, model, loader):
        if self.pretrain_epochs <= 0:
            return model
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.pretrain_epochs, eta_min=self.lr * 0.1)
        model.train()
        for epoch in range(self.pretrain_epochs):
            losses = []
            for x, _ in loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                details = model(x, mask_ratio=self.mask_ratio, return_details=True)
                loss = masked_reconstruction_loss(x, details['reconstruction'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(f"    pretrain epoch {epoch + 1}: recon={np.mean(losses):.5f}")
        return model

    def _predict(self, model, loader):
        model.eval()
        probs, anomaly_scores, channel_scores, temporal_scores, embeddings = [], [], [], [], []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                details = model(x, return_details=True)
                probs.append(torch.sigmoid(details['logit']).cpu().numpy())
                anomaly_scores.append(details['anomaly_score'].cpu().numpy())
                channel_scores.append(details['channel_scores'].cpu().numpy())
                temporal_scores.append(details['temporal_scores'].cpu().numpy())
                embeddings.append(details['embedding'].cpu().numpy())
        return (
            np.concatenate(probs),
            np.concatenate(anomaly_scores),
            np.concatenate(channel_scores),
            np.concatenate(temporal_scores),
            np.concatenate(embeddings),
        )

    def _finetune(self, model, train_loader, val_loader, y_val, pos_weight):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.finetune_epochs, eta_min=self.lr * 0.05)
        criterion = AsymmetricFocalLoss(
            alpha=self.focal_alpha,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            pos_weight=pos_weight,
        )
        best_state, best_f1, patience = None, 0, 0
        for epoch in range(self.finetune_epochs):
            model.train()
            losses = []
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.float().to(self.device)
                optimizer.zero_grad()
                clean = model(x, return_details=True)
                masked = model(x, mask_ratio=self.mask_ratio, return_details=True)
                cls_loss = criterion(clean['logit'], y)
                recon_loss = masked_reconstruction_loss(x, masked['reconstruction'])
                consistency = F.mse_loss(torch.sigmoid(clean['logit']), torch.sigmoid(masked['logit']))
                loss = cls_loss + self.lambda_recon * recon_loss + self.lambda_consistency * consistency
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            val_proba, _, _, _, _ = self._predict(model, val_loader)
            val_f1, _, val_rec, val_prec = self._best_threshold(y_val, val_proba)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(f"    finetune epoch {epoch + 1}: loss={np.mean(losses):.5f} val_f1={val_f1:.4f} rec={val_rec:.4f} prec={val_prec:.4f}")
            if patience >= self.patience:
                print(f"    early stopping at epoch {epoch + 1}")
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _build_hybrid_features(self, stat_features, nn_proba, anomaly, channel, temporal, embedding):
        parts = [
            np.nan_to_num(nn_proba.reshape(-1, 1), nan=0.5),
            np.nan_to_num(anomaly.reshape(-1, 1), nan=0.0),
            np.nan_to_num(channel, nan=0.0),
            np.nan_to_num(temporal, nan=0.0),
            np.nan_to_num(embedding, nan=0.0),
        ]
        if stat_features is not None:
            parts.insert(0, np.nan_to_num(stat_features, nan=0.0, posinf=0.0, neginf=0.0))
        X = np.column_stack(parts).astype(np.float32)
        return np.clip(X, -1e4, 1e4)

    def _train_hybrid_head(self, X_hybrid, labels, splits):
        oof_lgb = np.zeros(len(labels), dtype=np.float32)
        oof_xgb = np.zeros(len(labels), dtype=np.float32)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X_hybrid[train_idx], X_hybrid[val_idx]
            y_train = labels[train_idx]
            pos = max((y_train == 1).sum(), 1)
            neg = max((y_train == 0).sum(), 1)
            scale_pos_weight = neg / pos
            lgb_model = lgb.LGBMClassifier(
                n_estimators=1200,
                max_depth=7,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=SEED + fold_idx,
                verbose=-1,
            )
            lgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, labels[val_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            xgb_model = xgb.XGBClassifier(
                n_estimators=800,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=scale_pos_weight,
                tree_method='hist',
                verbosity=0,
                random_state=SEED + fold_idx,
            )
            xgb_model.fit(X_train, y_train)
            oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
            oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
            fold_blend = 0.6 * oof_lgb[val_idx] + 0.4 * oof_xgb[val_idx]
            f1, th, rec, prec = self._best_threshold(labels[val_idx], fold_blend)
            print(f"  hybrid fold {fold_idx + 1}: F1={f1:.4f} th={th:.3f} rec={rec:.4f} prec={prec:.4f}")
        return 0.6 * oof_lgb + 0.4 * oof_xgb

    def train(self, X_seq, labels, stat_features=None, fold_assignments=None):
        seed_everything(SEED)
        start = time.time()
        X_seq = np.asarray(X_seq, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if stat_features is not None:
            stat_features = np.asarray(stat_features, dtype=np.float32)
        splits = self._get_fold_splits(labels, fold_assignments)
        n = len(labels)
        oof = np.zeros(n, dtype=np.float32)
        anomaly = np.zeros(n, dtype=np.float32)
        channel = np.zeros((n, X_seq.shape[1]), dtype=np.float32)
        embedding = None
        temporal = None
        print("=" * 70)
        print(f"Foundation Encoder Training ({self.dataset.upper()})")
        print(f"  patch={self.patch_len}, stride={self.stride}, d_model={self.d_model}, layers={self.n_layers}, mask={self.mask_ratio}")
        print("=" * 70)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{len(splits)}")
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            model = self._create_model(X_seq)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  params={n_params:,} train={len(train_idx)} val={len(val_idx)}")
            pre_loader = self._loader(X_train, shuffle=True)
            model = self._pretrain(model, pre_loader)
            pos = max((y_train == 1).sum(), 1)
            neg = max((y_train == 0).sum(), 1)
            pos_weight = neg / pos
            train_loader = self._loader(X_train, y_train, shuffle=True, weighted=self.use_weighted_sampler)
            val_loader = self._loader(X_val, y_val, shuffle=False)
            model = self._finetune(model, train_loader, val_loader, y_val, pos_weight)
            val_proba, val_anomaly, val_channel, val_temporal, val_embedding = self._predict(model, val_loader)
            if temporal is None:
                temporal = np.zeros((n, val_temporal.shape[1]), dtype=np.float32)
            if embedding is None:
                embedding = np.zeros((n, val_embedding.shape[1]), dtype=np.float32)
            oof[val_idx] = val_proba
            anomaly[val_idx] = val_anomaly
            channel[val_idx] = val_channel
            temporal[val_idx] = val_temporal
            embedding[val_idx] = val_embedding
            f1, th, rec, prec = self._best_threshold(y_val, val_proba)
            print(f"  fold result: F1={f1:.4f} th={th:.3f} rec={rec:.4f} prec={prec:.4f}")
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{self.dataset}_foundation_fold{fold_idx}.pt'))
        nn_f1, nn_th, nn_rec, nn_prec = self._best_threshold(labels, oof)
        nn_auc = roc_auc_score(labels, oof)
        print(f"\n[Foundation NN] F1={nn_f1:.4f} AUC={nn_auc:.4f} th={nn_th:.3f} rec={nn_rec:.4f} prec={nn_prec:.4f}")
        print("\n[Hybrid Head] Training GBDT on stat features + foundation representations...")
        X_hybrid = self._build_hybrid_features(stat_features, oof, anomaly, channel, temporal, embedding)
        hybrid_oof = self._train_hybrid_head(X_hybrid, labels, splits)
        f1, th, rec, prec = self._best_threshold(labels, hybrid_oof)
        auc = roc_auc_score(labels, hybrid_oof)
        print(f"\n[Foundation Hybrid {self.dataset.upper()}] F1={f1:.4f} AUC={auc:.4f} th={th:.3f} rec={rec:.4f} prec={prec:.4f}")
        print(f"  time={(time.time() - start) / 60:.1f} min")
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f'{self.dataset}_foundation.npz'),
            oof_proba=hybrid_oof,
            nn_oof_proba=oof,
            flags=labels,
            anomaly_score=anomaly,
            channel_scores=channel,
            temporal_scores=temporal,
            embedding=embedding,
            f1=f1,
            nn_f1=nn_f1,
            auc=auc,
            threshold=th,
        )
        return {
            'oof_proba_meta': hybrid_oof,
            'oof_proba_a': oof,
            'oof_proba_b': hybrid_oof,
            'flags': labels,
            'best_f1': f1,
            'best_f1_unconstrained': f1,
            'best_th': th,
            'best_th_unconstrained': th,
            'best_recall': rec,
            'best_precision': prec,
            'primary_model_name': 'Foundation Hybrid',
            'anomaly_score': anomaly,
            'channel_scores': channel,
            'temporal_scores': temporal,
        }
