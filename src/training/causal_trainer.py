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
from src.models.cate_tst import CATETSTModel, AsymmetricFocalLoss, mmd_loss
from src.evaluation.causal_metrics import (
    policy_value_at_k,
    pehe_proxy,
    ate_gap,
    print_policy_report,
)
from src.utils.utils import seed_everything


class CausalTrainer:
    def __init__(self, dataset='sgcc', seq_target_len=256,
                 pretrain_epochs=8, counterfactual_epochs=8, dragonnet_epochs=20,
                 batch_size=64, lr=1e-4, weight_decay=1e-4, patience=6,
                 d_model=96, n_layers=2, n_heads=4, lstm_hidden=32, lstm_pool=8,
                 dropout=0.2, recon_segments=32,
                 lambda_recon=0.5, lambda_prop=0.5, lambda_mmd=0.1,
                 lambda_targeted=0.2, focal_alpha=0.75, gamma_pos=1.0, gamma_neg=3.0,
                 use_weighted_sampler=True, device=DEVICE):
        self.dataset = dataset
        self.seq_target_len = seq_target_len
        self.pretrain_epochs = pretrain_epochs
        self.counterfactual_epochs = counterfactual_epochs
        self.dragonnet_epochs = dragonnet_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.lstm_hidden = lstm_hidden
        self.lstm_pool = lstm_pool
        self.dropout = dropout
        self.recon_segments = recon_segments
        self.lambda_recon = lambda_recon
        self.lambda_prop = lambda_prop
        self.lambda_mmd = lambda_mmd
        self.lambda_targeted = lambda_targeted
        self.focal_alpha = focal_alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.use_weighted_sampler = use_weighted_sampler
        self.device = device

    def _prep(self, X_seq):
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

    def _loader(self, X, y=None, shuffle=True, weighted=False):
        X_t = torch.from_numpy(X.astype(np.float32))
        y_arr = np.asarray(y if y is not None else np.zeros(len(X)), dtype=np.int64)
        y_t = torch.from_numpy(y_arr)
        ds = TensorDataset(X_t, y_t)
        if weighted and y is not None:
            pos = max((y_arr == 1).sum(), 1)
            neg = max((y_arr == 0).sum(), 1)
            weights = np.where(y_arr == 1, 1.0 / pos, 1.0 / neg)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(ds, batch_size=self.batch_size, sampler=sampler)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def _create_model(self, X_seq):
        return CATETSTModel(
            in_channels=X_seq.shape[1],
            seq_len=X_seq.shape[2],
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            lstm_hidden=self.lstm_hidden,
            lstm_pool=self.lstm_pool,
            dropout=self.dropout,
            recon_segments=self.recon_segments,
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

    def _predict(self, model, loader):
        model.eval()
        probs, props, cf_n_mean, cf_t_mean, embeds, residuals = [], [], [], [], [], []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                out = model(x, return_details=True)
                probs.append(torch.sigmoid(out['logit']).cpu().numpy())
                props.append(torch.sigmoid(out['propensity_logit']).cpu().numpy())
                cf_n_mean.append(out['recon_normal'].mean(dim=(1, 2)).cpu().numpy())
                cf_t_mean.append(out['recon_theft'].mean(dim=(1, 2)).cpu().numpy())
                embeds.append(out['embedding'].cpu().numpy())
                residuals.append(out['residual_normal'].cpu().numpy())
        return {
            'proba': np.concatenate(probs),
            'propensity': np.concatenate(props),
            'cf_normal_mean': np.concatenate(cf_n_mean),
            'cf_theft_mean': np.concatenate(cf_t_mean),
            'embedding': np.concatenate(embeds),
            'residual': np.concatenate(residuals),
        }

    def _pretrain(self, model, loader):
        if self.pretrain_epochs <= 0:
            return model
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        model.train()
        for epoch in range(self.pretrain_epochs):
            losses = []
            for x, _ in loader:
                x = x.to(self.device)
                opt.zero_grad()
                pooled, _ = model.encode(x)
                recon = model.decoder_normal(pooled, x.shape[-1])
                loss = F.mse_loss(recon, x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            if epoch == 0 or (epoch + 1) % 2 == 0:
                print("    pretrain epoch {}: recon={:.5f}".format(epoch + 1, float(np.mean(losses))))
        return model

    def _counterfactual_finetune(self, model, x_train, y_train):
        if self.counterfactual_epochs <= 0:
            return model
        normal_idx = np.where(y_train == 0)[0]
        if len(normal_idx) < self.batch_size:
            return model
        loader = self._loader(x_train[normal_idx], shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        model.train()
        for epoch in range(self.counterfactual_epochs):
            losses = []
            for x, _ in loader:
                x = x.to(self.device)
                opt.zero_grad()
                pooled, _ = model.encode(x)
                recon = model.decoder_normal(pooled, x.shape[-1])
                loss = F.mse_loss(recon, x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            if epoch == 0 or (epoch + 1) % 2 == 0:
                print("    cf epoch {}: recon={:.5f}".format(epoch + 1, float(np.mean(losses))))
        return model

    def _dragonnet_train(self, model, train_loader, val_loader, y_val, pos_weight):
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.dragonnet_epochs, 1), eta_min=self.lr * 0.1)
        focal = AsymmetricFocalLoss(self.focal_alpha, self.gamma_pos, self.gamma_neg, pos_weight=pos_weight)
        best_state, best_f1, patience_ctr = None, 0.0, 0
        for epoch in range(self.dragonnet_epochs):
            model.train()
            losses = []
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.float().to(self.device)
                opt.zero_grad()
                out = model(x, return_details=True)
                cls_loss = focal(out['logit'], y)
                normal_mask = (y < 0.5)
                theft_mask = (y > 0.5)
                zero = x.new_zeros(())
                loss_rn = F.mse_loss(out['recon_normal'][normal_mask], x[normal_mask]) if normal_mask.any() else zero
                loss_rt = F.mse_loss(out['recon_theft'][theft_mask], x[theft_mask]) if theft_mask.any() else zero
                loss_recon = loss_rn + loss_rt
                prop_loss = F.binary_cross_entropy_with_logits(out['propensity_logit'], y)
                z_n = out['embedding'][normal_mask]
                z_t = out['embedding'][theft_mask]
                mmd = mmd_loss(z_n, z_t, sigma=1.0)
                targeted = ((torch.sigmoid(out['logit']) - y) * torch.sigmoid(out['propensity_logit'])).pow(2).mean()
                loss = (cls_loss
                        + self.lambda_recon * loss_recon
                        + self.lambda_prop * prop_loss
                        + self.lambda_mmd * mmd
                        + self.lambda_targeted * targeted)
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
                print("    dragonnet epoch {}: loss={:.4f} val_f1={:.4f} rec={:.4f} prec={:.4f}".format(
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
            np.nan_to_num(preds['propensity'].reshape(-1, 1), nan=0.5),
            np.nan_to_num(preds['cf_normal_mean'].reshape(-1, 1), nan=0.0),
            np.nan_to_num(preds['cf_theft_mean'].reshape(-1, 1), nan=0.0),
            np.nan_to_num(preds['residual'], nan=0.0),
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
        X_seq = self._prep(X_seq)
        labels = np.asarray(labels, dtype=np.int64)
        if stat_features is not None:
            stat_features = np.asarray(stat_features, dtype=np.float32)
        splits = self._splits(labels, fold_assignments)
        n = len(labels)
        oof = np.zeros(n, dtype=np.float32)
        prop = np.zeros(n, dtype=np.float32)
        cf_n = np.zeros(n, dtype=np.float32)
        cf_t = np.zeros(n, dtype=np.float32)
        residual_all = None
        embedding_all = None
        print("=" * 70)
        print("CATE-TST Causal Training ({})".format(self.dataset.upper()))
        print("  d_model={} n_layers={} lstm_hidden={} seq_len={}".format(
            self.d_model, self.n_layers, self.lstm_hidden, X_seq.shape[-1]))
        print("=" * 70)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print("\nFold {}/{}".format(fold_idx + 1, len(splits)))
            X_tr, X_va = X_seq[train_idx], X_seq[val_idx]
            y_tr, y_va = labels[train_idx], labels[val_idx]
            model = self._create_model(X_seq)
            n_params = sum(p.numel() for p in model.parameters())
            print("  params={:,} train={} val={}".format(n_params, len(train_idx), len(val_idx)))
            pre_loader = self._loader(X_tr, shuffle=True)
            model = self._pretrain(model, pre_loader)
            model = self._counterfactual_finetune(model, X_tr, y_tr)
            pos = max((y_tr == 1).sum(), 1)
            neg = max((y_tr == 0).sum(), 1)
            pos_weight = neg / pos
            train_loader = self._loader(X_tr, y_tr, shuffle=True, weighted=self.use_weighted_sampler)
            val_loader = self._loader(X_va, y_va, shuffle=False)
            model = self._dragonnet_train(model, train_loader, val_loader, y_va, pos_weight)
            preds = self._predict(model, val_loader)
            if residual_all is None:
                residual_all = np.zeros((n, preds['residual'].shape[1]), dtype=np.float32)
                embedding_all = np.zeros((n, preds['embedding'].shape[1]), dtype=np.float32)
            oof[val_idx] = preds['proba']
            prop[val_idx] = preds['propensity']
            cf_n[val_idx] = preds['cf_normal_mean']
            cf_t[val_idx] = preds['cf_theft_mean']
            residual_all[val_idx] = preds['residual']
            embedding_all[val_idx] = preds['embedding']
            f1, th, rec, prec = self._best_threshold(y_va, preds['proba'])
            print("  fold result: F1={:.4f} th={:.3f} rec={:.4f} prec={:.4f}".format(f1, th, rec, prec))
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, '{}_causal_fold{}.pt'.format(self.dataset, fold_idx)))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        nn_f1, nn_th, nn_rec, nn_prec = self._best_threshold(labels, oof)
        nn_auc = roc_auc_score(labels, oof)
        print("\n[Causal NN] F1={:.4f} AUC={:.4f} th={:.3f} rec={:.4f} prec={:.4f}".format(
            nn_f1, nn_auc, nn_th, nn_rec, nn_prec))
        print_policy_report('NN', labels, oof)
        pehe = pehe_proxy(labels, cf_n, cf_t)
        ate = ate_gap(oof, labels)
        print("  [NN causal] factual_mse={:.4f} ite_mean={:.4f} ite_std={:.4f} ate_gap={:.4f}".format(
            pehe['factual_mse'], pehe['ite_mean'], pehe['ite_std'], ate))
        print("\n[Hybrid Head] Training GBDT on stat + causal representations...")
        preds_all = {
            'proba': oof, 'propensity': prop,
            'cf_normal_mean': cf_n, 'cf_theft_mean': cf_t,
            'residual': residual_all, 'embedding': embedding_all,
        }
        X_hybrid = self._build_hybrid_features(stat_features, preds_all)
        hybrid_oof = self._train_hybrid_head(X_hybrid, labels, splits)
        f1, th, rec, prec = self._best_threshold(labels, hybrid_oof)
        auc = roc_auc_score(labels, hybrid_oof)
        print("\n[Causal Hybrid {}] F1={:.4f} AUC={:.4f} th={:.3f} rec={:.4f} prec={:.4f}".format(
            self.dataset.upper(), f1, auc, th, rec, prec))
        print_policy_report('Hybrid', labels, hybrid_oof)
        print("  time={:.1f} min".format((time.time() - start) / 60))
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, '{}_causal.npz'.format(self.dataset)),
            oof_proba=hybrid_oof, nn_oof_proba=oof, flags=labels,
            propensity=prop, cf_normal_mean=cf_n, cf_theft_mean=cf_t,
            residual=residual_all, embedding=embedding_all,
            f1=f1, nn_f1=nn_f1, auc=auc, threshold=th,
            pehe_factual_mse=pehe['factual_mse'], ite_mean=pehe['ite_mean'],
            ite_std=pehe['ite_std'], ate_gap=ate,
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
            'primary_model_name': 'Causal Hybrid',
            'propensity': prop,
            'cf_normal_mean': cf_n,
            'cf_theft_mean': cf_t,
            'ate_gap': ate,
            'pehe_proxy': pehe,
        }
