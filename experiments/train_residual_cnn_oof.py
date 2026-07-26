"""Train a lightweight 1D ResNet on STL residuals as a diverse OOF source.

Theft users may have residual patterns that differ from normal users.  This
model uses only the precomputed residual channel, so it is independent of the
existing AMST/Informer models that consume the full multi-channel series.
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
from src.models.models import RecallOrientedFocalLoss

seed_everything(SEED)

LOG_PATH = os.path.join(OUTPUT_DIR, 'residual_cnn.log')
_log_fh = open(LOG_PATH, 'w', buffering=1, encoding='utf-8')
sys.stdout = _log_fh
sys.stderr = _log_fh
print(f'Logging to {LOG_PATH}')

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
residuals = pre['residuals'].astype(np.float32)  # [N, T]
flags = pre['flags'].astype(np.int64)

# normalize per-sample
mean = residuals.mean(axis=1, keepdims=True)
std = residuals.std(axis=1, keepdims=True) + 1e-6
X = ((residuals - mean) / std)[:, None, :]  # [N, 1, T]


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


class ResidualCNN(nn.Module):
    def __init__(self, in_channels=1, seq_len=1034, base_channels=32, n_blocks=4, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        layers = []
        ch = base_channels
        for i in range(n_blocks):
            stride = 2 if i % 2 == 1 else 1
            layers.append(ResBlock1D(ch, ch * 2 if stride == 2 else ch, stride=stride, dropout=dropout))
            if stride == 2:
                ch *= 2
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(ch, ch),
            nn.LayerNorm(ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ch, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)


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


def train_fold(X_train, y_train, X_val, y_val, cfg, device, fold_idx):
    model = ResidualCNN(
        in_channels=X_train.shape[1],
        seq_len=X_train.shape[2],
        base_channels=cfg['base_channels'],
        n_blocks=cfg['n_blocks'],
        dropout=cfg['dropout'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Fold {fold_idx+1} params: {n_params:,}')

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32).to(device)
    criterion = RecallOrientedFocalLoss(alpha=0.75, gamma=2.0, recall_weight=3.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, drop_last=False, num_workers=0)

    use_amp = device == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = 0.0
    best_state = None
    patience = 0
    max_patience = 12

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(bx)
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
            for bx, _ in val_loader:
                bx = bx.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(bx)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_probs = np.nan_to_num(val_probs, nan=0.5)
        auc = roc_auc_score(y_val, val_probs)
        f1, _ = best_f1_th(y_val, val_probs)

        if epoch % 5 == 0 or epoch == cfg['epochs'] - 1:
            print(f'    Epoch {epoch+1}: loss={total_loss/max(n_batches,1):.4f} val_F1={f1:.4f} val_AUC={auc:.4f}')

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= max_patience:
            print(f'    Early stop at epoch {epoch+1} (best val_F1={best_val_f1:.4f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X, device, batch_size=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.FloatTensor(X[i:i+batch_size]).to(device)
            with torch.cuda.amp.autocast(enabled=device == 'cuda' and torch.cuda.is_available()):
                logits = model(bx)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = {
        'base_channels': 32,
        'n_blocks': 4,
        'dropout': 0.25,
        'epochs': 50,
        'batch_size': 128,
        'lr': 3e-4,
    }
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(flags), dtype=np.float32)

    for fi, (ti, vi) in enumerate(skf.split(X, flags)):
        print(f'\n=== Fold {fi+1}/{N_FOLDS} ===')
        model = train_fold(X[ti], flags[ti], X[vi], flags[vi], cfg, device, fi)
        probs = predict(model, X[vi], device)
        oof[vi] = np.nan_to_num(probs, nan=0.5)

        f1, th = best_f1_th(flags[vi], oof[vi])
        pred = (oof[vi] > th).astype(int)
        print(f'  Fold {fi+1}: F1={f1_score(flags[vi], pred):.4f}, '
              f'Rec={recall_score(flags[vi], pred):.4f}, '
              f'Prec={precision_score(flags[vi], pred, zero_division=0):.4f}, '
              f'AUC={roc_auc_score(flags[vi], oof[vi]):.4f}, th={th:.3f}')

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'residual_cnn_fold{fi}.pt'))
        del model
        torch.cuda.empty_cache()

    f1, th = best_f1_th(flags, oof)
    pred = (oof > th).astype(int)
    print(f'\n=== Overall Residual CNN ===')
    print(f'F1={f1_score(flags, pred):.4f}, Rec={recall_score(flags, pred):.4f}, '
          f'Prec={precision_score(flags, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(flags, oof):.4f}, th={th:.3f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'residual_cnn_oof.npz'),
        oof_residual_cnn=oof,
        flags=flags,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "residual_cnn_oof.npz")}')
    print(f'Total time: {(time.time()-t0)/60:.1f} minutes')


if __name__ == '__main__':
    main()
