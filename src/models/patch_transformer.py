"""Simple Patch Transformer classifier for time-series.

Much faster than the custom Informer because it uses PyTorch's optimized
TransformerEncoder on patch tokens instead of a Python-loop attention.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_len, stride, d_model, max_patches=512):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.channel_emb = nn.Parameter(torch.randn(1, in_channels, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, max_patches, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        if T < self.patch_len:
            x = F.pad(x, (0, self.patch_len - T))
        remainder = (T - self.patch_len) % self.stride
        if remainder:
            x = F.pad(x, (0, self.stride - remainder))
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [B, C, N, P]
        tokens = self.proj(patches)  # [B, C, N, D]
        N = tokens.shape[2]
        tokens = tokens + self.channel_emb[:, :, :1, :]
        tokens = tokens + self.pos_emb[:, :, :N, :]
        tokens = self.norm(tokens)
        return tokens.reshape(B, C * N, tokens.shape[-1])  # [B, C*N, D]


class PatchTransformerClassifier(nn.Module):
    def __init__(self, in_channels=5, seq_len=1034, patch_len=30, stride=15,
                 d_model=128, n_layers=4, n_heads=8, dropout=0.2, use_prior=False):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_len, stride, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.use_prior = use_prior
        clf_in = d_model + (1 if use_prior else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(clf_in),
            nn.Linear(clf_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x, prior=None):
        # x: [B, C, T]
        tokens = self.patch_embed(x)  # [B, L, D]
        encoded = self.encoder(tokens)  # [B, L, D]
        pooled = encoded.mean(dim=1)  # [B, D]
        if self.use_prior and prior is not None:
            pooled = torch.cat([pooled, prior.reshape(-1, 1)], dim=1)
        return self.classifier(pooled).squeeze(-1)


def _best_f1(y_true, proba):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        pred = (proba > th).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    return best_f1, best_th


def predict_patch_transformer(model, X_seq, oof_prior=None, batch_size=256, device='cuda'):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            end = min(start + batch_size, len(X_seq))
            xb = torch.FloatTensor(X_seq[start:end]).to(device)
            pb = torch.FloatTensor(oof_prior[start:end]).to(device) if oof_prior is not None else None
            logits = model(xb, pb)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs)


def train_patch_transformer(X_seq, y, oof_prior=None, patch_len=30, stride=15,
                            d_model=128, n_layers=4, n_heads=8, dropout=0.2,
                            epochs=40, batch_size=128, lr=3e-4, device='cuda',
                            seed=42, verbose=True, val_ratio=0.1):
    from sklearn.metrics import f1_score, roc_auc_score
    from src.models.supcon_model import SymmetricCrossEntropy
    from torch.utils.data import DataLoader, TensorDataset

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    N = len(y)
    in_ch = X_seq.shape[1]
    use_prior = oof_prior is not None

    if val_ratio and val_ratio > 0 and N >= 20:
        stratify = y if len(np.unique(y)) > 1 else None
        train_idx, val_idx = train_test_split(
            np.arange(N), test_size=val_ratio, random_state=seed, stratify=stratify)
    else:
        train_idx = np.arange(N)
        val_idx = np.arange(N)

    model = PatchTransformerClassifier(
        in_channels=in_ch, seq_len=X_seq.shape[2], patch_len=patch_len, stride=stride,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
        use_prior=use_prior).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  PatchTransformer: {n_params:,} params (prior={use_prior})")

    criterion = SymmetricCrossEntropy(alpha=1.0, beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = device == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    X_t = torch.FloatTensor(X_seq)
    y_t = torch.FloatTensor(y)
    if use_prior:
        p_t = torch.FloatTensor(oof_prior)
        train_ds = TensorDataset(X_t[train_idx], y_t[train_idx], p_t[train_idx])
        val_ds = TensorDataset(X_t[val_idx], y_t[val_idx], p_t[val_idx])
    else:
        train_ds = TensorDataset(X_t[train_idx], y_t[train_idx])
        val_ds = TensorDataset(X_t[val_idx], y_t[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    best_val_auc = 0.0
    best_state = None
    patience = 0
    max_patience = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_data in train_loader:
            if use_prior:
                bx, by, bp = batch_data; bp = bp.to(device)
            else:
                bx, by = batch_data; bp = None
            bx, by = bx.to(device), by.to(device)
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
        val_probs_all = []
        with torch.no_grad():
            for batch_data in val_loader:
                if use_prior:
                    bx, by, bp = batch_data; bp = bp.to(device)
                else:
                    bx, by = batch_data; bp = None
                bx = bx.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(bx, bp)
                val_probs_all.append(torch.sigmoid(logits).cpu().numpy())
        val_probs = np.concatenate(val_probs_all)
        val_probs = np.nan_to_num(val_probs, nan=0.5)
        y_val = y[val_idx]
        auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.0
        best_f1_epoch = _best_f1(y_val, val_probs)[0]

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"    Epoch {epoch+1}: train_loss={total_loss/max(n_batches,1):.4f} "
                  f"val_F1={best_f1_epoch:.4f} val_AUC={auc:.4f}")

        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= max_patience:
            if verbose:
                print(f"    Early stop at epoch {epoch+1} (best val_AUC={best_val_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
