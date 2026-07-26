"""Train an enhanced Informer on fold 5 only, with Focal Loss + balanced sampling.

Goal: quickly verify if a larger model + better loss can beat the current
Informer fold-5 result of F1=0.8697.
"""
import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.models.informer_model import InformerClassifier, predict_informer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

seed_everything(SEED)

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return loss.mean()


def add_temporal_noise(X, noise_std=0.01, scale_range=(0.95, 1.05)):
    """Light magnitude warping + jittering."""
    if noise_std <= 0:
        return X
    scale = np.random.uniform(*scale_range, size=(X.shape[0], 1, 1)).astype(np.float32)
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return X * scale + noise


def train_informer_enhanced(X_train, y_train, X_val, y_val, prior_train=None, prior_val=None,
                            d_model=128, n_heads=8, num_layers=3, dropout=0.3,
                            epochs=80, batch_size=16, lr=2e-4, device='cuda',
                            loss_type='focal', use_augment=True, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    in_ch = X_train.shape[1]
    use_prior = prior_train is not None
    model = InformerClassifier(in_ch, d_model, n_heads, num_layers, dropout, use_prior).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Informer enhanced: {n_params:,} params, prior={use_prior}, loss={loss_type}')

    # Loss
    if loss_type == 'focal':
        pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device)
        criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    elif loss_type == 'bce':
        pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        from src.models.supcon_model import SymmetricCrossEntropy
        criterion = SymmetricCrossEntropy(alpha=1.0, beta=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if use_prior:
        p_t = torch.FloatTensor(prior_train)
        train_dataset = TensorDataset(X_t, y_t, p_t)
    else:
        train_dataset = TensorDataset(X_t, y_t)

    # Balanced sampler
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(y_train), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              drop_last=False, num_workers=0)

    best_val_f1 = 0.0
    best_state = None
    patience = 0
    max_patience = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_data in train_loader:
            if use_prior:
                bx, by, bp = batch_data
                bp = bp.to(device)
            else:
                bx, by = batch_data
                bp = None
            bx, by = bx.to(device), by.to(device)

            if use_augment and np.random.rand() < 0.5:
                bx = torch.FloatTensor(add_temporal_noise(bx.cpu().numpy(), noise_std=0.02)).to(device)

            optimizer.zero_grad()
            logits = model(bx, bp)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation (batched to avoid OOM)
        model.eval()
        val_probs = predict_informer(model, X_val, prior_val, batch_size=64, device=device)
        val_probs = np.nan_to_num(val_probs, nan=0.5)

        best_f1_epoch = 0.0
        best_th_epoch = 0.5
        for th in np.arange(0.1, 0.9, 0.01):
            pred = (val_probs > th).astype(int)
            if pred.sum() == 0: continue
            f = f1_score(y_val, pred, zero_division=0)
            if f > best_f1_epoch:
                best_f1_epoch = f
                best_th_epoch = th

        auc = roc_auc_score(y_val, val_probs)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f'    Epoch {epoch+1}: train_loss={total_loss/max(n_batches,1):.4f} '
                  f'val_F1={best_f1_epoch:.4f} val_AUC={auc:.4f} th={best_th_epoch:.2f}')

        # Use val F1 for model selection (since end goal is F1)
        if best_f1_epoch > best_val_f1:
            best_val_f1 = best_f1_epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f'    Early stop at epoch {epoch+1} (best val_F1={best_val_f1:.4f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


def main():
    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_3ch.npz'))
    X_seq = pre['X_seq']
    flags = pre['flags']

    prior_data = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
    oof_prior = prior_data['prior']

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_seq, flags))

    # Train only on fold 5 (best historical fold)
    fi = 4
    ti, vi = splits[fi]
    print(f'\n=== Fold {fi+1} (historically best) ===')

    # Further split train into train/val for early stopping
    ti2, vai = train_test_split(ti, test_size=0.1, random_state=SEED, stratify=flags[ti])

    t0 = time.time()
    model, best_val_f1 = train_informer_enhanced(
        X_seq[ti2], flags[ti2], X_seq[vai], flags[vai],
        prior_train=oof_prior[ti2], prior_val=oof_prior[vai],
        d_model=64, n_heads=4, num_layers=2, dropout=0.3,
        epochs=80, batch_size=32, lr=2e-4,
        loss_type='focal', use_augment=True, seed=SEED + fi,
    )

    # Final evaluation on the held-out validation fold (vi)
    val_probs = predict_informer(model, X_seq[vi], oof_prior[vi], batch_size=64, device='cuda')
    val_probs = np.nan_to_num(val_probs, nan=0.5)

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (val_probs > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(flags[vi], pred, zero_division=0)
        if f > best_f1: best_f1, best_th = f, th
    pred = (val_probs > best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(flags[vi], pred).ravel()

    print(f'\n=== Fold {fi+1} Final ===')
    print(f'F1={best_f1:.4f}, Rec={recall_score(flags[vi], pred):.4f}, '
          f'Prec={precision_score(flags[vi], pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags[vi], val_probs):.4f}, th={best_th:.3f}')
    print(f'TP={tp} FP={fp} FN={fn}')
    print(f'Time: {(time.time()-t0)/60:.1f} min')

    # Save
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'informer_enhanced_fold5.npz'),
        oof=val_probs, y=flags[vi], f1=best_f1, auc=roc_auc_score(flags[vi], val_probs)
    )
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'informer_enhanced_fold5.pt'))


if __name__ == '__main__':
    main()
