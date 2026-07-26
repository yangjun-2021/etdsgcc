import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import lightgbm as lgb
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from config import DEVICE, N_FOLDS, OUTPUT_DIR, SEED
from src.models.contrastive_encoder import (
    ContrastiveTemporalEncoder,
    AsymmetricFocalLoss,
    knn_contrastive_loss,
    batch_label_contrastive_loss,
)
from src.evaluation.causal_metrics import policy_value_at_k, print_policy_report
from src.utils.utils import seed_everything


class ContrastiveTrainer:
    def __init__(self, dataset='sgcc', seq_target_len=256,
                 contrastive_epochs=10, finetune_epochs=20,
                 batch_size=64, lr=1e-4, lr_finetune=1e-4,
                 weight_decay=1e-4, patience=8,
                 d_model=96, lstm_hidden=32, lstm_layers=2,
                 proj_dim=128, dropout=0.1, temperature=0.5,
                 knn_K=50,
                 focal_alpha=0.75, gamma_pos=1.0, gamma_neg=3.0,
                 use_weighted_sampler=True, use_amp=True, device=DEVICE):
        self.dataset = dataset
        self.seq_target_len = seq_target_len
        self.contrastive_epochs = contrastive_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_finetune = lr_finetune
        self.weight_decay = weight_decay
        self.patience = patience
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.temperature = temperature
        self.knn_K = knn_K
        self.focal_alpha = focal_alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.use_weighted_sampler = use_weighted_sampler
        self.use_amp = use_amp and torch.device(device).type == 'cuda'
        self.device = torch.device(device)

    def _prep_seq(self, X_seq):
        X_seq = np.asarray(X_seq, dtype=np.float32)
        if self.seq_target_len and X_seq.shape[-1] != self.seq_target_len:
            X_t = torch.from_numpy(X_seq)
            X_t = F.adaptive_avg_pool1d(X_t, self.seq_target_len)
            X_seq = X_t.numpy()
        return X_seq

    def _splits(self, y, fold_assignments):
        if fold_assignments is not None:
            return [
                (np.where(fold_assignments != f)[0], np.where(fold_assignments == f)[0])
                for f in np.unique(fold_assignments)
            ]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        return list(skf.split(np.zeros(len(y)), y))

    def _loader(self, X, y=None, indices=None, shuffle=True, weighted=False):
        X_t = torch.from_numpy(X.astype(np.float32))
        y_arr = np.asarray(y if y is not None else np.zeros(len(X)), dtype=np.int64)
        y_t = torch.from_numpy(y_arr)
        tensors = [X_t, y_t]
        if indices is not None:
            idx_t = torch.from_numpy(np.asarray(indices, dtype=np.int64))
            tensors.append(idx_t)
        ds = TensorDataset(*tensors)
        if weighted and y is not None:
            pos = max((y_arr == 1).sum(), 1)
            neg = max((y_arr == 0).sum(), 1)
            weights = np.where(y_arr == 1, 1.0 / pos, 1.0 / neg)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(ds, batch_size=self.batch_size, sampler=sampler)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def _create_model(self):
        return ContrastiveTemporalEncoder(
            in_channels=5, d_model=self.d_model,
            lstm_hidden=self.lstm_hidden, lstm_layers=self.lstm_layers,
            proj_dim=self.proj_dim, dropout=self.dropout,
        ).to(self.device)

    @staticmethod
    def _best_threshold(y_true, proba):
        idx = np.argsort(proba)
        y_sorted = np.asarray(y_true)[idx]
        n = len(y_true)
        total_pos = max(y_sorted.sum(), 1.0)
        cum_pos = np.cumsum(y_sorted[::-1])
        k = np.arange(1, n + 1, dtype=np.float64)
        prec = cum_pos / k
        rec = cum_pos / total_pos
        denom = prec + rec
        denom[denom == 0] = 1.0
        f1 = 2 * prec * rec / denom
        best = int(np.argmax(f1))
        th = float(proba[idx[-(best + 1)]])
        return float(f1[best]), th, float(rec[best]), float(prec[best])

    def _predict(self, model, loader):
        model.eval()
        probs, embeds = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                logit = model(x, mode='classify')
                probs.append(torch.sigmoid(logit).cpu().numpy())
                embeds.append(model(x, mode='embed').cpu().numpy())
        return {
            'proba': np.concatenate(probs),
            'embedding': np.concatenate(embeds),
        }

    def _build_knn_graph(self, stat_features):
        stat_features = np.asarray(stat_features, dtype=np.float32)
        stat_features = np.nan_to_num(stat_features, nan=0.0, posinf=0.0, neginf=0.0)
        nn_model = NearestNeighbors(n_neighbors=self.knn_K + 1, metric='cosine', n_jobs=-1)
        nn_model.fit(stat_features)
        knn_idx = nn_model.kneighbors(stat_features, return_distance=False)
        return knn_idx[:, 1:]

    @staticmethod
    def _build_batch_mask(batch_idx, knn_idx):
        B = len(batch_idx)
        idx_to_pos = {int(idx): i for i, idx in enumerate(batch_idx)}
        mask = np.zeros((B, B), dtype=bool)
        for i, idx in enumerate(batch_idx):
            for nb in knn_idx[idx]:
                pos = idx_to_pos.get(int(nb))
                if pos is not None:
                    mask[i, pos] = True
        return mask

    def _contrastive_pretrain(self, model, loader, knn_idx):
        if self.contrastive_epochs <= 0:
            return model
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        use_knn = knn_idx is not None
        model.train()
        for epoch in range(self.contrastive_epochs):
            losses = []
            for batch in loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                idx = batch[2].numpy() if len(batch) > 2 else None
                opt.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        z = model(x, mode='contrastive')
                        if use_knn and idx is not None:
                            mask_np = self._build_batch_mask(idx, knn_idx)
                            mask_t = torch.from_numpy(mask_np).to(self.device)
                            loss = knn_contrastive_loss(z, y, mask_t, self.temperature)
                        else:
                            loss = batch_label_contrastive_loss(z, y, self.temperature)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    z = model(x, mode='contrastive')
                    if use_knn and idx is not None:
                        mask_np = self._build_batch_mask(idx, knn_idx)
                        mask_t = torch.from_numpy(mask_np).to(self.device)
                        loss = knn_contrastive_loss(z, y, mask_t, self.temperature)
                    else:
                        loss = batch_label_contrastive_loss(z, y, self.temperature)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                losses.append(loss.item())
            if epoch == 0 or (epoch + 1) % 2 == 0:
                mean_loss = float(np.mean(losses))
                min_loss = float(np.min(losses))
                max_loss = float(np.max(losses))
                print("    contrastive epoch {}: loss={:.5f} (min={:.3f} max={:.3f})".format(epoch + 1, mean_loss, min_loss, max_loss))
        return model

    def _supervised_finetune(self, model, train_loader, val_loader, y_val, pos_weight):
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr_finetune, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.finetune_epochs, eta_min=self.lr_finetune * 0.1)
        focal = AsymmetricFocalLoss(self.focal_alpha, self.gamma_pos, self.gamma_neg, pos_weight=pos_weight)
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        best_state, best_f1, patience_ctr = None, 0.0, 0
        for epoch in range(self.finetune_epochs):
            model.train()
            losses = []
            for batch in train_loader:
                x = batch[0].to(self.device)
                y = batch[1].float().to(self.device)
                opt.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logit = model(x, mode='classify')
                        loss = focal(logit, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logit = model(x, mode='classify')
                    loss = focal(logit, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                losses.append(loss.item())
            sch.step()
            preds = self._predict(model, val_loader)
            val_f1, _, val_rec, val_prec = self._best_threshold(y_val, preds['proba'])
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if epoch == 0 or (epoch + 1) % 2 == 0:
                print("    finetune epoch {}: loss={:.4f} val_f1={:.4f} rec={:.4f} prec={:.4f}".format(
                    epoch + 1, float(np.mean(losses)), val_f1, val_rec, val_prec))
            if patience_ctr >= self.patience:
                print("    early stopping at epoch {}".format(epoch + 1))
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _build_hybrid_features(self, stat_features, preds):
        parts = [
            np.nan_to_num(preds['proba'].reshape(-1, 1), nan=0.5),
            np.nan_to_num(preds['embedding'], nan=0.0),
        ]
        if stat_features is not None:
            parts.insert(0, np.nan_to_num(stat_features, nan=0.0, posinf=0.0, neginf=0.0))
        X = np.column_stack(parts).astype(np.float32)
        return np.clip(X, -1e4, 1e4)

    def _train_hybrid_head(self, X_hybrid, labels, splits):
        oof_lgb = np.zeros(len(labels), dtype=np.float32)
        oof_xgb = np.zeros(len(labels), dtype=np.float32)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr, X_va = X_hybrid[train_idx], X_hybrid[val_idx]
            y_tr = labels[train_idx]
            pos = max((y_tr == 1).sum(), 1)
            neg = max((y_tr == 0).sum(), 1)
            spw = neg / pos
            lgb_m = lgb.LGBMClassifier(
                n_estimators=1200, max_depth=7, learning_rate=0.03,
                num_leaves=63, min_child_samples=80, subsample=0.85,
                colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.1,
                scale_pos_weight=spw, random_state=SEED + fold_idx, verbose=-1,
            )
            lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, labels[val_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            xgb_m = xgb.XGBClassifier(
                n_estimators=800, max_depth=5, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=0.1,
                scale_pos_weight=spw, tree_method='hist', verbosity=0,
                random_state=SEED + fold_idx,
            )
            xgb_m.fit(X_tr, y_tr)
            oof_lgb[val_idx] = lgb_m.predict_proba(X_va)[:, 1]
            oof_xgb[val_idx] = xgb_m.predict_proba(X_va)[:, 1]
            fold_blend = 0.6 * oof_lgb[val_idx] + 0.4 * oof_xgb[val_idx]
            f1, th, rec, prec = self._best_threshold(labels[val_idx], fold_blend)
            print("  hybrid fold {}: F1={:.4f} th={:.3f} rec={:.4f} prec={:.4f}".format(
                fold_idx + 1, f1, th, rec, prec))
        return 0.6 * oof_lgb + 0.4 * oof_xgb

    def train(self, X_seq, labels, stat_features=None, fold_assignments=None):
        seed_everything(SEED)
        start = time.time()
        X_seq = self._prep_seq(X_seq)
        labels = np.asarray(labels, dtype=np.int64)
        if stat_features is not None:
            stat_features = np.asarray(stat_features, dtype=np.float32)
        knn_idx = None
        if stat_features is not None:
            print("[KNN] Building stat-feature nearest-neighbor graph (K={})...".format(self.knn_K))
            knn_idx = self._build_knn_graph(stat_features)
        splits = self._splits(labels, fold_assignments)
        n = len(labels)
        oof = np.zeros(n, dtype=np.float32)
        embedding_all = None
        print("=" * 70)
        print("Contrastive Temporal Encoder ({})".format(self.dataset.upper()))
        print("  d_model={} lstm_hidden={} layers={} proj_dim={} seq_len={} knn_K={}".format(
            self.d_model, self.lstm_hidden, self.lstm_layers, self.proj_dim, X_seq.shape[-1], self.knn_K))
        print("=" * 70)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print("\nFold {}/{}".format(fold_idx + 1, len(splits)))
            X_tr, X_va = X_seq[train_idx], X_seq[val_idx]
            y_tr, y_va = labels[train_idx], labels[val_idx]
            model = self._create_model()
            n_params = sum(p.numel() for p in model.parameters())
            print("  params={:,} train={} val={}".format(n_params, len(train_idx), len(val_idx)))
            pre_loader = self._loader(X_tr, y_tr, indices=train_idx, shuffle=True)
            model = self._contrastive_pretrain(model, pre_loader, knn_idx)
            pos = max(int(y_tr.sum()), 1)
            neg = max(int((1 - y_tr).sum()), 1)
            pos_weight = neg / pos
            train_loader = self._loader(X_tr, y_tr, shuffle=True, weighted=self.use_weighted_sampler)
            val_loader = self._loader(X_va, y_va, shuffle=False)
            model = self._supervised_finetune(model, train_loader, val_loader, y_va, pos_weight)
            preds = self._predict(model, val_loader)
            if embedding_all is None:
                embedding_all = np.zeros((n, preds['embedding'].shape[1]), dtype=np.float32)
            oof[val_idx] = preds['proba']
            embedding_all[val_idx] = preds['embedding']
            f1, th, rec, prec = self._best_threshold(y_va, preds['proba'])
            print("  fold result: F1={:.4f} th={:.3f} rec={:.4f} prec={:.4f}".format(f1, th, rec, prec))
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, '{}_contrastive_fold{}.pt'.format(self.dataset, fold_idx)))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        nn_f1, nn_th, nn_rec, nn_prec = self._best_threshold(labels, oof)
        if len(np.unique(labels)) >= 2:
            nn_auc = roc_auc_score(labels, oof)
        else:
            nn_auc = 0.5
        print("\n[Contrastive NN] F1={:.4f} AUC={:.4f} th={:.3f} rec={:.4f} prec={:.4f}".format(
            nn_f1, nn_auc, nn_th, nn_rec, nn_prec))
        print_policy_report('NN', labels, oof)
        print("\n[Hybrid Head] Training GBDT on stat + contrastive embeddings...")
        preds_all = {'proba': oof, 'embedding': embedding_all}
        X_hybrid = self._build_hybrid_features(stat_features, preds_all)
        hybrid_oof = self._train_hybrid_head(X_hybrid, labels, splits)
        f1, th, rec, prec = self._best_threshold(labels, hybrid_oof)
        auc = roc_auc_score(labels, hybrid_oof)
        print("\n[Contrastive Hybrid {}] F1={:.4f} AUC={:.4f} th={:.3f} rec={:.4f} prec={:.4f}".format(
            self.dataset.upper(), f1, auc, th, rec, prec))
        print_policy_report('Hybrid', labels, hybrid_oof)
        print("  time={:.1f} min".format((time.time() - start) / 60))
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, '{}_contrastive.npz'.format(self.dataset)),
            oof_proba=hybrid_oof, nn_oof_proba=oof, flags=labels,
            embedding=embedding_all, f1=f1, nn_f1=nn_f1, auc=auc, threshold=th,
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
            'primary_model_name': 'Contrastive Hybrid',
            'embedding': embedding_all,
        }
