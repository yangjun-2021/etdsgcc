"""
Expert B: TCN + Leaf Embedding Training.

Note: "Expert B" is a naming convention for a deep learning component
(TCN with leaf-embedding features). It is NOT a Mixture-of-Experts (MoE)
architecture. "Expert" here denotes the second model in the two-stage
cascade (Expert A GBDT → Expert B TCN → Meta Learner), consuming Expert A's
leaf indices and prior probabilities as auxiliary inputs.

Unified trainer for SGCC and OEDI datasets, parameterized by dataset config.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import warnings

from config import (SGCC_CONFIG, OEDI_CONFIG, SEED, N_FOLDS, OUTPUT_DIR, DEVICE)
from src.models.models import TCNWithLeafEmbedding, RecallOrientedFocalLoss

warnings.filterwarnings('ignore')


def predict_batched(model, X_np, leaf_np, prior_np=None, batch_size=512, device='cuda'):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            end = min(start + batch_size, len(X_np))
            x_batch = torch.FloatTensor(X_np[start:end]).to(device)
            l_batch = torch.LongTensor(leaf_np[start:end]).to(device)
            p_batch = None
            if prior_np is not None:
                p_batch = torch.FloatTensor(prior_np[start:end]).to(device)
            logits = model(x_batch, l_batch, p_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def compute_loss_batched(model, criterion, X_np, leaf_np, y_np, prior_np=None,
                          batch_size=512, device='cuda'):
    total_loss = 0.0
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            end = min(start + batch_size, len(X_np))
            x_batch = torch.FloatTensor(X_np[start:end]).to(device)
            l_batch = torch.LongTensor(leaf_np[start:end]).to(device)
            y_batch = torch.FloatTensor(y_np[start:end]).to(device)
            p_batch = None
            if prior_np is not None:
                p_batch = torch.FloatTensor(prior_np[start:end]).to(device)
            logits = model(x_batch, l_batch, p_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


class ExpertBTrainer:
    """Unified TCN+Leaf trainer for both SGCC and OEDI datasets.

    Parameters
    ----------
    dataset : str
        'sgcc' or 'oedi'
    """

    def __init__(self, dataset='sgcc'):
        self.dataset = dataset
        self.config = SGCC_CONFIG if dataset == 'sgcc' else OEDI_CONFIG
        self.dataset_name = self.config['name']
        self.tcn_params = self.config['tcn_params']
        self.train_params = self.config['train_params']

    def _get_fold_splits(self, X_seq, labels, fold_assignments=None):
        """Return list of (train_idx, val_idx) for each fold."""
        if self.dataset == 'sgcc':
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            return list(skf.split(X_seq, labels))
        else:
            unique_folds = np.unique(fold_assignments)
            splits = []
            for fold_idx in unique_folds:
                train_idx = np.where(fold_assignments != fold_idx)[0]
                val_idx = np.where(fold_assignments == fold_idx)[0]
                splits.append((train_idx, val_idx))
            return splits

    def _create_model(self, in_channels):
        model = TCNWithLeafEmbedding(
            in_channels=in_channels,
            tcn_channels=self.tcn_params['num_channels'],
            kernel_size=self.tcn_params['kernel_size'],
            dropout=self.tcn_params['dropout'],
            n_trees=self.tcn_params['n_trees'] * 2,
            num_leaves=self.tcn_params['num_leaves'],
            leaf_embed_dim=self.tcn_params['leaf_embed_dim'],
            leaf_output_dim=self.tcn_params['leaf_embed_dim'],
            use_prior=True,
        ).to(DEVICE)
        return model

    def _find_best_threshold(self, y_true, proba, th_range=None):
        if th_range is None:
            th_range = np.arange(0.2, 0.8, 0.005)
        best_th = 0.5
        best_f1 = 0
        best_recall = 0
        best_precision = 0
        for th in th_range:
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

    def train(self, X_seq, stat_features, labels, impute_mask=None,
              oof_proba_a=None, leaf_indices=None, fold_assignments=None):
        """Train TCN+Leaf model across folds.

        Parameters
        ----------
        X_seq : np.ndarray
            Multi-channel sequence input [N, C, T]
        stat_features : np.ndarray
            Statistical features (used for miss_ratio in SGCC)
        labels : np.ndarray
            Binary labels
        impute_mask : np.ndarray, optional
            Missing value mask (required for SGCC)
        oof_proba_a : np.ndarray
            Prior probabilities from Expert A
        leaf_indices : np.ndarray
            Combined leaf indices from Expert A
        fold_assignments : np.ndarray, optional
            Pre-defined fold assignments (required for OEDI)

        Returns
        -------
        oof_proba_b : np.ndarray
            Out-of-fold predicted probabilities
        """
        print("=" * 60)
        print(f"Expert B: TCN Training ({self.dataset_name.upper()})")
        print("=" * 60)

        n = len(labels)
        splits = self._get_fold_splits(X_seq, labels, fold_assignments)
        oof_proba_b = np.zeros(n)
        all_best_thresholds = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")

            torch.cuda.empty_cache()

            y_train_np = labels[train_idx]
            y_val_np = labels[val_idx]
            leaf_train_np = leaf_indices[train_idx]
            leaf_val_np = leaf_indices[val_idx]
            prior_train = oof_proba_a[train_idx].astype(np.float32)
            prior_val = oof_proba_a[val_idx].astype(np.float32)

            model = self._create_model(X_seq.shape[1])

            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Model parameters: {n_params:,}")

            criterion = RecallOrientedFocalLoss(
                alpha=self.train_params['focal_alpha'],
                gamma=self.train_params['focal_gamma'],
                recall_weight=self.train_params['recall_weight'],
            )

            optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=self.train_params['lr'],
                                           weight_decay=self.train_params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.train_params['epochs'], eta_min=1e-6
            )

            train_dataset = TensorDataset(
                torch.FloatTensor(X_seq[train_idx]),
                torch.LongTensor(leaf_train_np),
                torch.FloatTensor(y_train_np),
                torch.FloatTensor(prior_train),
            )
            train_loader = DataLoader(train_dataset, batch_size=self.train_params['batch_size'],
                                       shuffle=True, num_workers=0, drop_last=False)

            best_val_auc = 0.0
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(self.train_params['epochs']):
                model.train()
                epoch_loss = 0
                n_batches = 0

                for batch_x, batch_leaf, batch_y, batch_prior in train_loader:
                    batch_x = batch_x.to(DEVICE)
                    batch_leaf = batch_leaf.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    batch_prior = batch_prior.to(DEVICE)
                    optimizer.zero_grad()
                    logits = model(batch_x, batch_leaf, batch_prior)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                val_probs = predict_batched(model, X_seq[val_idx], leaf_indices[val_idx],
                                              prior_val, batch_size=512, device=DEVICE)
                val_loss = compute_loss_batched(model, criterion, X_seq[val_idx], leaf_indices[val_idx],
                                                y_val_np, prior_val, batch_size=512, device=DEVICE)

                best_f1_fold, best_th_fold, _, _ = self._find_best_threshold(
                    labels[val_idx], val_probs, th_range=np.arange(0.2, 0.8, 0.01))

                val_pred = (val_probs > best_th_fold).astype(int)
                val_f1 = f1_score(labels[val_idx], val_pred)
                val_recall = recall_score(labels[val_idx], val_pred)
                val_auc = roc_auc_score(labels[val_idx], val_probs)

                # Select model by validation AUC to avoid optimistic threshold-dependent F1 selection
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                log_interval = 5 if self.dataset == 'sgcc' else 10
                if epoch % log_interval == 0 or epoch == self.train_params['epochs'] - 1:
                    print(f"    Epoch {epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}, "
                          f"val_loss={val_loss:.4f}, F1={val_f1:.4f}, "
                          f"Recall={val_recall:.4f}, AUC={val_auc:.4f}, th={best_th_fold:.2f}")

                if patience_counter >= self.train_params['patience']:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            val_probs = predict_batched(model, X_seq[val_idx], leaf_indices[val_idx],
                                          prior_val, batch_size=512, device=DEVICE)
            oof_proba_b[val_idx] = val_probs

            best_f1_final, best_th, _, _ = self._find_best_threshold(labels[val_idx], val_probs)
            all_best_thresholds.append(best_th)

            val_pred = (val_probs > best_th).astype(int)
            print(f"  Fold {fold_idx+1} Best: F1={best_f1_final:.4f}, "
                  f"Recall={recall_score(labels[val_idx], val_pred):.4f}, "
                  f"Precision={precision_score(labels[val_idx], val_pred):.4f}, th={best_th:.3f}")

            torch.save(model.state_dict(),
                        os.path.join(OUTPUT_DIR, f'{self.dataset_name}_tcn_fold{fold_idx}.pt'))

        # Overall metrics
        best_th_overall = np.median(all_best_thresholds)
        overall_auc = roc_auc_score(labels, oof_proba_b)

        th_search_f1 = 0
        th_search_recall = 0
        for th in np.arange(0.2, 0.8, 0.005):
            pred = (oof_proba_b > th).astype(int)
            f1 = f1_score(labels, pred)
            if f1 > th_search_f1:
                th_search_f1 = f1
                best_th_overall = th
                th_search_recall = recall_score(labels, pred)

        print(f"\n[Expert B {self.dataset_name.upper()}] Overall: F1={th_search_f1:.4f}, "
              f"Recall={th_search_recall:.4f}, AUC={overall_auc:.4f}")
        print(f"  Best overall threshold: {best_th_overall:.3f}")

        label_key = 'flags' if self.dataset == 'sgcc' else 'y'
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f'{self.dataset_name}_expert_b.npz'),
            oof_proba=oof_proba_b,
            **{label_key: labels},
        )

        return oof_proba_b


# Backward-compatible function wrappers
def train_tcn_sgcc(X_seq, stat_features, flags, impute_mask, oof_proba_a, leaf_indices):
    """Backward-compatible wrapper for SGCC TCN training."""
    trainer = ExpertBTrainer(dataset='sgcc')
    return trainer.train(X_seq, stat_features, flags, impute_mask=impute_mask,
                         oof_proba_a=oof_proba_a, leaf_indices=leaf_indices)


def train_tcn_oedi(X_seq, stat_features, y, fold_assignments, oof_proba_a, leaf_indices):
    """Backward-compatible wrapper for OEDI TCN training."""
    trainer = ExpertBTrainer(dataset='oedi')
    return trainer.train(X_seq, stat_features, y, fold_assignments=fold_assignments,
                         oof_proba_a=oof_proba_a, leaf_indices=leaf_indices)


if __name__ == '__main__':
    print("Run expert_b.py through the main pipeline (run_pipeline.py)")
