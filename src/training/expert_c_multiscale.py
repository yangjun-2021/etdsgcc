"""
Expert C (MultiScaleCNN1D): lightweight multi-scale 1D-CNN trainer for SGCC.

This expert consumes the raw multi-channel sequence [N, 5, 1035] directly
without GBDT leaf embeddings. It is designed as a fast, complementary deep
expert to Expert A (GBDT) and Expert B (TCN+Leaf).
"""
import os
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import SGCC_CONFIG, SEED, N_FOLDS, OUTPUT_DIR, DEVICE
from src.models.models import RecallOrientedFocalLoss
from src.models.multiscale_cnn import MultiScaleCNN1D

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_tensor(arr, dtype=torch.float32):
    return torch.from_numpy(arr).to(dtype)


def _clean_input(X):
    """Sanitise sequence input: convert to float32 and replace NaN/Inf."""
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_batched(model, X, batch_size=512, device='cuda'):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            x_batch = _to_tensor(X[start:end]).to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def _compute_loss_batched(model, criterion, X, y, batch_size=512, device='cuda'):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            x_batch = _to_tensor(X[start:end]).to(device)
            y_batch = _to_tensor(y[start:end]).to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def _find_best_threshold(y_true, proba, th_range=None):
    if th_range is None:
        th_range = np.arange(0.2, 0.8, 0.005)
    best_th = 0.5
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ExpertCMultiScaleTrainer:
    """5-fold trainer for the MultiScaleCNN1D SGCC expert."""

    def __init__(self, dataset='sgcc',
                 kernel_sizes=(3, 5, 7, 11),
                 branch_channels=8,
                 hidden_dim=32,
                 dropout=0.25,
                 batch_size=256,
                 epochs=20,
                 lr=5e-4,
                 weight_decay=1e-4,
                 patience=5,
                 focal_alpha=0.85,
                 focal_gamma=2.0,
                 recall_weight=3.0):
        if dataset != 'sgcc':
            raise ValueError("ExpertCMultiScaleTrainer currently supports only 'sgcc'")
        self.dataset = dataset
        self.config = SGCC_CONFIG
        self.dataset_name = self.config['name']

        # Model hyper-parameters
        self.kernel_sizes = kernel_sizes
        self.branch_channels = branch_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Training hyper-parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.recall_weight = recall_weight

    def _create_model(self, in_channels, seq_len):
        model = MultiScaleCNN1D(
            in_channels=in_channels,
            kernel_sizes=self.kernel_sizes,
            branch_channels=self.branch_channels,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(DEVICE)
        return model

    def train(self, X_seq, labels):
        """Train MultiScaleCNN1D across 5 stratified folds.

        Parameters
        ----------
        X_seq : np.ndarray
            Multi-channel sequence input [N, C, T].
        labels : np.ndarray
            Binary labels [N].

        Returns
        -------
        oof_proba_c : np.ndarray
            Out-of-fold predicted probabilities [N].
        """
        print("=" * 60)
        print(f"Expert C: MultiScaleCNN1D Training ({self.dataset_name.upper()})")
        print("=" * 60)

        X_seq = _clean_input(X_seq)
        labels = np.asarray(labels, dtype=np.float32)

        n = len(labels)
        n_channels = X_seq.shape[1]
        seq_len = X_seq.shape[2]

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_proba_c = np.zeros(n)
        all_best_thresholds = []
        fold_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_seq, labels)):
            print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
            t_fold_start = time.time()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            X_train = X_seq[train_idx]
            y_train = labels[train_idx]
            X_val = X_seq[val_idx]
            y_val = labels[val_idx]

            model = self._create_model(n_channels, seq_len)
            print(f"    Model parameters: {model.count_parameters():,}")

            criterion = RecallOrientedFocalLoss(
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                recall_weight=self.recall_weight,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=1e-6
            )

            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )

            best_val_auc = 0.0
            patience_counter = 0
            best_model_state = None

            for epoch in range(self.epochs):
                model.train()
                epoch_loss = 0.0
                n_batches = 0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)

                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                    # Guard against invalid loss values
                    if not torch.isfinite(loss):
                        print(f"    [Warning] non-finite loss at epoch {epoch+1}; skipping batch")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                # Validation
                val_probs = _predict_batched(model, X_val, batch_size=512, device=DEVICE)
                val_loss = _compute_loss_batched(model, criterion, X_val, y_val,
                                                 batch_size=512, device=DEVICE)

                best_f1_fold, best_th_fold, _, _ = _find_best_threshold(
                    y_val, val_probs, th_range=np.arange(0.2, 0.8, 0.01)
                )

                val_pred = (val_probs > best_th_fold).astype(int)
                val_f1 = f1_score(y_val, val_pred, zero_division=0)
                val_recall = recall_score(y_val, val_pred, zero_division=0)
                val_auc = roc_auc_score(y_val, val_probs)

                # Select model by validation AUC (threshold-independent)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 3 == 0 or epoch == self.epochs - 1:
                    print(
                        f"    Epoch {epoch+1:3d}: loss={epoch_loss/max(n_batches,1):.4f}, "
                        f"val_loss={val_loss:.4f}, F1={val_f1:.4f}, "
                        f"Recall={val_recall:.4f}, AUC={val_auc:.4f}, th={best_th_fold:.2f}"
                    )

                if patience_counter >= self.patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            # Restore best weights
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Final validation OOF for this fold
            val_probs = _predict_batched(model, X_val, batch_size=512, device=DEVICE)
            oof_proba_c[val_idx] = val_probs

            best_f1_final, best_th, _, _ = _find_best_threshold(y_val, val_probs)
            all_best_thresholds.append(best_th)

            val_pred = (val_probs > best_th).astype(int)
            print(
                f"  Fold {fold_idx+1} Best: F1={best_f1_final:.4f}, "
                f"Recall={recall_score(y_val, val_pred, zero_division=0):.4f}, "
                f"Precision={precision_score(y_val, val_pred, zero_division=0):.4f}, th={best_th:.3f}"
            )

            # Save fold checkpoint
            ckpt_path = os.path.join(OUTPUT_DIR,
                                     f'{self.dataset_name}_expert_c_multiscale_fold{fold_idx}.pt')
            torch.save(model.state_dict(), ckpt_path)
            fold_models.append(ckpt_path)

            t_fold_elapsed = time.time() - t_fold_start
            print(f"  Fold {fold_idx+1} time: {t_fold_elapsed/60:.1f} min")

        # Overall metrics
        best_th_overall = np.median(all_best_thresholds)
        overall_auc = roc_auc_score(labels, oof_proba_c)

        th_search_f1 = 0.0
        th_search_recall = 0.0
        for th in np.arange(0.2, 0.8, 0.005):
            pred = (oof_proba_c > th).astype(int)
            f1 = f1_score(labels, pred, zero_division=0)
            if f1 > th_search_f1:
                th_search_f1 = f1
                best_th_overall = th
                th_search_recall = recall_score(labels, pred, zero_division=0)

        print(f"\n[Expert C {self.dataset_name.upper()}] Overall: F1={th_search_f1:.4f}, "
              f"Recall={th_search_recall:.4f}, AUC={overall_auc:.4f}")
        print(f"  Best overall threshold: {best_th_overall:.3f}")

        # Save OOF probabilities
        out_path = os.path.join(OUTPUT_DIR, f'{self.dataset_name}_expert_c_multiscale.npz')
        np.savez_compressed(
            out_path,
            oof_proba=oof_proba_c,
            flags=labels,
        )
        print(f"  Saved OOF probabilities to {out_path}")

        return oof_proba_c


# Backward-compatible wrapper
def train_multiscale_cnn_sgcc(X_seq, flags):
    """Backward-compatible wrapper for SGCC MultiScaleCNN training."""
    trainer = ExpertCMultiScaleTrainer(dataset='sgcc')
    return trainer.train(X_seq, flags)


if __name__ == '__main__':
    print("Run expert_c_multiscale.py through run_expert_c_multiscale.py or run_pipeline.py")
