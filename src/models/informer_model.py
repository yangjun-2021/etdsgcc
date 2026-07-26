"""
Informer-based Electricity Theft Detection Model

Based on Zhou et al. (AAAI 2021) "Informer: Beyond Efficient Transformer
for Long Sequence Time-Series Forecasting"

Key innovations:
  1. ProbSparse Self-Attention: O(L log L) instead of O(L²)
     - Selects top-u dominant queries based on sparsity measurement
     - Drops non-dominant queries (fills with mean V)
  2. Self-Attention Distilling: Progressive sequence length reduction
     - Between encoder layers: Conv1d(stride=2) → ELU → MaxPool1d(stride=2)
     - Creates pyramid representation
  3. Classification adaptation:
     - Uses Informer encoder only (not the forecasting decoder)
     - Global average pooling → MLP classifier
     - GBDT OOF prior input to classification head

Why Informer for SGCC:
  - 1034-day sequences are too long for standard Transformer O(L²)
  - ProbSparse reduces complexity to O(L log L) → feasible for full sequence
  - Distilling captures multi-scale temporal patterns automatically
  - Previous TFT-Transformer was limited by O(L²) → had to use patch embedding
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class ProbSparseAttention(nn.Module):
    """ProbSparse Self-Attention: selects top-u dominant queries.

    Instead of computing full Q·K^T for all queries, we:
      1. Sample a subset of keys for sparsity measurement
      2. Compute M(q_i, K_sample) for each query
      3. Select top-u queries with highest M
      4. Only compute attention for selected queries
      5. For non-selected queries: use mean value (V_mean)

    Complexity: O(L log L) vs O(L²) for standard attention.
    """

    def __init__(self, d_model=128, n_heads=8, dropout=0.3,
                 factor=5, sample_factor=5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor          # c in the paper: top-u = c * ln(L_Q)
        self.sample_factor = sample_factor  # sampling factor for K

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q_sample, K_sample, L_Q):
        """Sparsity measurement: M(q_i, K) for top-u selection."""
        # Q_sample: [B, H, L_Q, d_k]
        # K_sample: [B, H, L_K, d_k]
        # M = max(q·k^T) - mean(q·k^T) over keys
        scores = torch.matmul(Q_sample, K_sample.transpose(-2, -1))  # [B, H, L_Q, L_K]
        scores = scores / (self.d_k ** 0.5)
        M = scores.max(dim=-1)[0] - scores.mean(dim=-1)  # [B, H, L_Q]
        return M

    def forward(self, x):
        """ProbSparse attention forward.

        Args:
            x: [B, L, D] input sequence
        Returns:
            out: [B, L, D] attended sequence
        """
        B, L, D = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L, dk]
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # 1. Select top-u dominant queries using sparsity measurement
        u = max(1, min(int(self.factor * np.log(L)), L // 2))
        U_part = min(self.sample_factor * u, L)

        if U_part < L:
            idx_K = torch.randperm(L, device=x.device)[:U_part]
            K_sample = K[:, :, idx_K, :]
        else:
            K_sample = K

        # M = max(q·k_sample^T) - mean(q·k_sample^T)
        scores_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / (self.d_k ** 0.5)
        M = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)  # [B, H, L]

        # Top-u queries
        M_mean = M.mean(dim=0)  # [H, L] - average over batch
        _, top_idx = torch.topk(M_mean, u, dim=-1)  # [H, u]

        # 2. Gather selected queries only
        Q_reduce = torch.zeros(B, self.n_heads, u, self.d_k, device=x.device)
        for h in range(self.n_heads):
            Q_reduce[:, h] = Q[:, h, top_idx[h], :]

        # 3. Compute attention ONLY for selected queries (O(u * L))
        scores_reduce = torch.matmul(Q_reduce, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_reduce = scores_reduce.softmax(dim=-1)
        attn_reduce = self.dropout(attn_reduce)  # [B, H, u, L]

        # 4. Update: selected queries get full attention, rest get V mean
        V_mean = V.mean(dim=2, keepdim=True)  # [B, H, 1, dk]
        out = V_mean.expand(B, self.n_heads, L, self.d_k).clone()

        for h in range(self.n_heads):
            updated = torch.matmul(attn_reduce[:, h], V[:, h])  # [B, u, dk]
            out[:, h, top_idx[h]] = updated

        # Reshape
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out


class ConvDistilling(nn.Module):
    """Self-attention distilling: reduce sequence length by half.

    Conv1d(stride=2) → ELU → MaxPool1d(stride=2)
    This creates a pyramid representation, halving the sequence length
    at each distilling layer.
    """

    def __init__(self, d_model=128):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                               stride=1, padding=1)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """x: [B, L, D] -> [B, L//2, D]"""
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.conv(x)
        x = self.elu(x)
        x = self.pool(x)  # [B, D, L//2]
        x = x.transpose(1, 2)  # [B, L//2, D]
        return x


class InformerEncoderLayer(nn.Module):
    """Single Informer encoder block: Attention + Distilling."""

    def __init__(self, d_model=128, n_heads=8, dropout=0.3):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class InformerEncoder(nn.Module):
    """Informer Encoder: stack of layers with distilling between them.

    L1: Attention + Distill → L/2
    L2: Attention + Distill → L/4
    L3: Attention (no distill) → L/4

    Output: concatenation of pyramid features → global pooling.
    """

    def __init__(self, d_model=128, n_heads=8, num_layers=3,
                 dropout=0.3, input_dim=4):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(InformerEncoderLayer(d_model, n_heads, dropout))

        self.distill_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.distill_layers.append(ConvDistilling(d_model))

    def forward(self, x):
        """
        Args:
            x: [B, C, T] raw 4-channel input
        Returns:
            features: [B, D'] concatenated pyramid features
        """
        B, C, T = x.shape
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.value_embedding(x)  # [B, T, D]
        x = x + self.pos_embedding[:, :T, :]

        pyramid_features = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.distill_layers):
                pyramid_features.append(x.mean(dim=1))  # [B, D]
                x = self.distill_layers[i](x)  # [B, T//2, D]

        pyramid_features.append(x.mean(dim=1))  # Final layer

        # Concatenate all pyramid levels
        combined = torch.cat(pyramid_features, dim=1)  # [B, num_layers * D]
        return combined


class InformerClassifier(nn.Module):
    """Informer-based classifier for electricity theft detection.

    Architecture:
      1. Informer Encoder (ProbSparse + Distilling)
      2. Pyramid feature concatenation
      3. MLP classifier with optional GBDT OOF prior
    """

    def __init__(self, input_dim=4, d_model=128, n_heads=8, num_layers=3,
                 dropout=0.3, use_prior=False):
        super().__init__()
        self.encoder = InformerEncoder(d_model, n_heads, num_layers,
                                        dropout, input_dim)
        self.use_prior = use_prior

        feat_dim = d_model * num_layers
        if use_prior:
            feat_dim += 1  # GBDT OOF prior

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x, prior=None):
        """
        Args:
            x: [B, C, T] 4-channel raw input
            prior: [B] optional GBDT OOF probabilities
        Returns:
            logit: [B]
        """
        features = self.encoder(x)

        if self.use_prior and prior is not None:
            features = torch.cat([features, prior.reshape(-1, 1)], dim=1)

        logit = self.classifier(features)
        return logit.squeeze(-1)


# ============= TRAINING UTILITIES =============

def predict_informer(model, X_seq, oof_prior=None, batch_size=256, device='cuda'):
    """Batch prediction for Informer model."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            end = min(start + batch_size, len(X_seq))
            xb = torch.FloatTensor(X_seq[start:end]).to(device)
            pb = None
            if oof_prior is not None:
                pb = torch.FloatTensor(oof_prior[start:end]).to(device)
            logits = model(xb, pb)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs)


def train_informer(X_seq, y, oof_prior=None, d_model=128, n_heads=8,
                    num_layers=3, dropout=0.3, epochs=40, batch_size=64,
                    lr=3e-4, device='cuda', seed=42, verbose=True,
                    val_ratio=0.1, use_amp=None):
    """Train Informer classifier with SCE loss.

    Uses a stratified held-out validation split from the training fold for
    early stopping and model selection (fixes the previous bug of validating
    on the training set itself).
    """
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

    # Stratified train/val split inside the fold
    if val_ratio and val_ratio > 0 and N >= 20:
        stratify = y if len(np.unique(y)) > 1 else None
        train_idx, val_idx = train_test_split(
            np.arange(N), test_size=val_ratio, random_state=seed,
            stratify=stratify)
    else:
        train_idx = np.arange(N)
        val_idx = np.arange(N)

    model = InformerClassifier(in_ch, d_model, n_heads, num_layers,
                                dropout, use_prior).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Informer: {n_params:,} params (prior={use_prior}, val_ratio={val_ratio})")

    criterion = SymmetricCrossEntropy(alpha=1.0, beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    # Automatic Mixed Precision for faster training on modern GPUs
    if use_amp is None:
        use_amp = device == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    X_t = torch.FloatTensor(X_seq)
    y_t = torch.FloatTensor(y)

    if use_prior:
        p_t = torch.FloatTensor(oof_prior)
        train_dataset = TensorDataset(X_t[train_idx], y_t[train_idx], p_t[train_idx])
        val_dataset = TensorDataset(X_t[val_idx], y_t[val_idx], p_t[val_idx])
    else:
        train_dataset = TensorDataset(X_t[train_idx], y_t[train_idx])
        val_dataset = TensorDataset(X_t[val_idx], y_t[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=0)

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

        # Validate on held-out split
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_probs_all = []
        with torch.no_grad():
            for batch_data in val_loader:
                if use_prior:
                    bx, by, bp = batch_data; bp = bp.to(device)
                else:
                    bx, by = batch_data; bp = None
                bx, by = bx.to(device), by.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(bx, bp)
                    loss = criterion(logits, by)
                val_loss += loss.item()
                val_batches += 1
                val_probs_all.append(torch.sigmoid(logits).cpu().numpy())

        val_probs = np.concatenate(val_probs_all) if val_probs_all else np.array([])
        val_probs = np.nan_to_num(val_probs, nan=0.5)
        y_val = y[val_idx]

        auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.0
        best_f1_epoch = 0.0
        for th in np.arange(0.1, 0.9, 0.01):
            pred = (val_probs > th).astype(int)
            if pred.sum() == 0:
                continue
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 > best_f1_epoch:
                best_f1_epoch = f1

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"    Epoch {epoch+1}: train_loss={total_loss/max(n_batches,1):.4f} "
                  f"val_loss={val_loss/max(val_batches,1):.4f} "
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


def train_informer_cv(X_seq, y, oof_prior=None, n_folds=5, seed=42,
                       device='cuda', **kwargs):
    """K-fold CV training for Informer."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y))

    for fi, (ti, vi) in enumerate(skf.split(X_seq, y)):
        print(f"\n  Informer Fold {fi+1}/{n_folds}")
        model = train_informer(
            X_seq[ti], y[ti],
            oof_prior[ti] if oof_prior is not None else None,
            device=device, seed=seed + fi, verbose=True, **kwargs
        )
        val_probs = predict_informer(
            model, X_seq[vi],
            oof_prior[vi] if oof_prior is not None else None,
            device=device
        )
        val_probs = np.nan_to_num(val_probs, nan=0.5)
        oof_proba[vi] = val_probs

        bf = 0
        for th in np.arange(0.1, 0.9, 0.005):
            p = (val_probs > th).astype(int)
            if p.sum() == 0: continue
            f = f1_score(y[vi], p, zero_division=0)
            if f > bf: bf = f
        print(f"  Fold {fi+1}: F1={bf:.4f} AUC={roc_auc_score(y[vi], val_probs):.4f}")
        del model; torch.cuda.empty_cache()

    return oof_proba


if __name__ == '__main__':
    print("Testing Informer model...")
    X = np.random.randn(32, 4, 200).astype(np.float32)
    y = (np.random.rand(32) < 0.15).astype(np.float32)
    y[:5] = 1
    prior = np.random.rand(32).astype(np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = train_informer(X, y, prior, d_model=64, n_heads=4, num_layers=2,
                            epochs=5, batch_size=16, device=device)
    probs = predict_informer(model, X, prior, device=device)
    print(f"Informer output: {probs.shape}, range=[{probs.min():.4f}, {probs.max():.4f}]")
    print("Test passed!")
