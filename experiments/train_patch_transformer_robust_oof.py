"""Robust Patch Transformer 5-fold OOF for SGCC.

Uses the existing PatchTransformerClassifier but replaces the unstable
SymmetricCrossEntropy loss with BCE/RecallOrientedFocal, gradient clipping,
and AMP.  Goal: produce a diverse OOF that the meta learner can use.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.models.patch_transformer import PatchTransformerClassifier
from src.models.models import RecallOrientedFocalLoss

seed_everything(SEED)

LOG_PATH = os.path.join(OUTPUT_DIR, 'patch_transformer_robust.log')
_log_fh = open(LOG_PATH, 'w', buffering=1, encoding='utf-8')
sys.stdout = _log_fh
sys.stderr = _log_fh
print(f'Logging to {LOG_PATH}')

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
X_seq = pre['X_seq']
flags = pre['flags']

a = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
oof_proba_a = a['oof_proba'].astype(np.float32)

CONFIG = {
    'patch_len': 24,
    'stride': 12,
    'd_model': 128,
    'n_layers': 4,
    'n_heads': 8,
    'dropout': 0.3,
    'epochs': 50,
    'batch_size': 128,
    'lr': 1e-4,
}


def best_f1_th(y_true, proba):
    best = (0, 0.5)
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (proba > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best[0]:
            best = (f, th)
    return best


def train_fold(X_train, y_train, X_val, y_val, prior_train, prior_val, cfg, device, fold_idx):
    in_ch = X_train.shape[1]
    model = PatchTransformerClassifier(
        in_channels=in_ch, seq_len=X_train.shape[2],
        patch_len=cfg['patch_len'], stride=cfg['stride'],
        d_model=cfg['d_model'], n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'], dropout=cfg['dropout'],
        use_prior=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Fold {fold_idx+1} params: {n_params:,}')

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32).to(device)
    criterion = RecallOrientedFocalLoss(alpha=0.75, gamma=2.0, recall_weight=3.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(prior_train),
        torch.FloatTensor(y_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(prior_val),
        torch.FloatTensor(y_val),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, drop_last=False, num_workers=0)

    use_amp = device == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc = 0.0
    best_state = None
    patience = 0
    max_patience = 12

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for bx, bp, by in train_loader:
            bx, bp, by = bx.to(device), bp.to(device), by.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(bx, bp)
                loss = criterion(logits, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for bx, bp, _ in val_loader:
                bx = bx.to(device)
                bp = bp.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(bx, bp)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_probs = np.nan_to_num(val_probs, nan=0.5)
        auc = roc_auc_score(y_val, val_probs)
        f1, _ = best_f1_th(y_val, val_probs)

        if epoch % 5 == 0 or epoch == cfg['epochs'] - 1:
            print(f'    Epoch {epoch+1}: loss={total_loss/max(n_batches,1):.4f} val_F1={f1:.4f} val_AUC={auc:.4f}')

        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= max_patience:
            print(f'    Early stop at epoch {epoch+1} (best val_AUC={best_val_auc:.4f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X, prior, device, batch_size=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.FloatTensor(X[i:i+batch_size]).to(device)
            bp = torch.FloatTensor(prior[i:i+batch_size]).to(device)
            logits = model(bx, bp)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(flags), dtype=np.float32)

    for fi, (ti, vi) in enumerate(skf.split(X_seq, flags)):
        print(f'\n=== Fold {fi+1}/{N_FOLDS} ===')
        model = train_fold(
            X_seq[ti], flags[ti], X_seq[vi], flags[vi],
            oof_proba_a[ti], oof_proba_a[vi],
            CONFIG, device, fi,
        )
        probs = predict(model, X_seq[vi], oof_proba_a[vi], device)
        oof[vi] = np.nan_to_num(probs, nan=0.5)

        f1, th = best_f1_th(flags[vi], oof[vi])
        pred = (oof[vi] > th).astype(int)
        print(f'  Fold {fi+1}: F1={f1_score(flags[vi], pred):.4f}, '
              f'Rec={recall_score(flags[vi], pred):.4f}, '
              f'Prec={precision_score(flags[vi], pred, zero_division=0):.4f}, '
              f'AUC={roc_auc_score(flags[vi], oof[vi]):.4f}, th={th:.3f}')

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'patch_transformer_robust_fold{fi}.pt'))
        del model
        torch.cuda.empty_cache()

    f1, th = best_f1_th(flags, oof)
    pred = (oof > th).astype(int)
    print(f'\n=== Overall Patch Transformer Robust ===')
    print(f'F1={f1_score(flags, pred):.4f}, Rec={recall_score(flags, pred):.4f}, '
          f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags, oof):.4f}, th={th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'patch_transformer_robust_oof.npz'),
        oof_patch_transformer_robust=oof,
        flags=flags,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "patch_transformer_robust_oof.npz")}')
    print(f'Total time: {(time.time()-t0)/60:.1f} minutes')


if __name__ == '__main__':
    main()
