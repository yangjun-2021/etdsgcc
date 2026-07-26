"""
Multi-Scale Time-Frequency Transformer for Electricity Theft Detection.

Architecture:
  1. Temporal stream: Multi-scale patch embedding (7/14/30 day patches)
     → 4-layer Transformer Encoder with long-range self-attention
  2. Frequency stream: FFT → 8 frequency bands → band attention
  3. Cross-attention fusion: temporal queries frequency
  4. Multi-task heads: classification (SCE) + reconstruction (MSE) + contrast (SupCon)

Key innovations vs prior work:
  - Transformer captures cross-season patterns (TCN receptive field only ~121 steps)
  - Frequency stream finds theft signatures invisible in time domain
  - Multi-task learning regularizes the encoder
  - GBDT OOF as prior input to classification head
  - SCE loss handles ~3% label noise
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

from src.models.supcon_model import SupConLoss, SymmetricCrossEntropy


class PatchEmbedding(nn.Module):
    """Convert time series into patch tokens for Transformer.

    Multi-scale: creates patches at different lengths to capture
    patterns at daily, weekly, and monthly scales simultaneously.
    """

    def __init__(self, in_channels, d_model, patch_lens, stride):
        super().__init__()
        self.patch_lens = patch_lens
        self.stride = stride
        self.projections = nn.ModuleList([
            nn.Conv1d(in_channels, d_model, kernel_size=pl, stride=stride)
            for pl in patch_lens
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            tokens: [B, n_patches, d_model * len(patch_lens)]
        """
        all_tokens = []
        for proj, pl in zip(self.projections, self.patch_lens):
            tokens = proj(x)  # [B, d_model, n_patches]
            tokens = tokens.transpose(1, 2)  # [B, n_patches, d_model]
            all_tokens.append(tokens)

        min_patches = min(t.shape[1] for t in all_tokens)
        all_tokens = [t[:, :min_patches, :] for t in all_tokens]
        combined = torch.cat(all_tokens, dim=-1)

        if combined.shape[-1] != self.projections[0].out_channels:
            combined = combined.view(combined.shape[0], min_patches, -1)

        combined = self.norm(combined)
        return combined, min_patches


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal stream."""

    def __init__(self, d_model=128, nhead=8, num_layers=4, dropout=0.3,
                 max_seq_len=200):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: [B, S, D]
        Returns:
            [B, S, D]
        """
        S = x.shape[1]
        x = x + self.pos_embedding[:, :S, :]
        x = self.encoder(x)
        return x


class FrequencyStream(nn.Module):
    """Frequency domain stream: FFT → band energy → attention.

    Extracts 8 frequency bands and applies self-attention
    to capture inter-band relationships.
    """

    def __init__(self, in_channels, d_model=128, n_bands=8, dropout=0.3):
        super().__init__()
        self.n_bands = n_bands
        self.band_proj = nn.Linear(in_channels * 3, d_model)  # 3 stats per band

        self.pos_embed = nn.Parameter(torch.randn(1, n_bands, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, n_bands, D]
        """
        B, C, T = x.shape

        x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        fft_vals = torch.fft.rfft(x_clean, dim=-1)  # [B, C, T//2+1]
        power = fft_vals.abs() ** 2  # [B, C, T//2+1]
        power = torch.clamp(power, min=1e-10, max=1e6)
        log_power = torch.log(power + 1e-8)

        band_size = max(power.shape[-1] // self.n_bands, 1)
        band_features = []
        for i in range(self.n_bands):
            start = i * band_size
            end = min(start + band_size, power.shape[-1])
            if end <= start:
                end = start + 1
            band_log = log_power[:, :, start:end]  # [B, C, band_size]

            energy = band_log.mean(dim=-1)  # [B, C] - log-scale energy
            energy_std = band_log.std(dim=-1)  # [B, C]

            band_power = power[:, :, start:end]
            total = band_power.sum(dim=-1, keepdim=True) + 1e-8
            norm_power = band_power / total
            entropy = -(norm_power * torch.log(norm_power + 1e-8)).sum(dim=-1)  # [B, C]

            freqs = torch.arange(start, end, device=x.device).float()
            centroid = (freqs * norm_power.squeeze(-1) if False else
                        (freqs * band_power).sum(dim=-1) / (band_power.sum(dim=-1) + 1e-8))  # [B, C]

            band_feat = torch.cat([energy, entropy, centroid], dim=-1)  # [B, C*3]
            band_features.append(band_feat)

        band_features = torch.stack(band_features, dim=1)  # [B, n_bands, C*3]
        band_features = self.band_proj(band_features)  # [B, n_bands, D]
        band_features = band_features + self.pos_embed
        band_features = self.encoder(band_features)  # [B, n_bands, D]

        return band_features


class CrossAttentionFusion(nn.Module):
    """Cross-attention: temporal stream queries frequency stream."""

    def __init__(self, d_model=128, nhead=8, dropout=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, temporal, frequency):
        """
        Args:
            temporal: [B, S, D] - temporal tokens (query)
            frequency: [B, n_bands, D] - frequency tokens (key/value)
        Returns:
            [B, S, D] - fused temporal representation
        """
        attn_out, _ = self.cross_attn(temporal, frequency, frequency)
        x = self.norm1(temporal + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MultiScaleTFTTransformer(nn.Module):
    """Full Multi-Scale Time-Frequency Transformer.

    Multi-task outputs:
      - classification logit
      - reconstructed sequence
      - contrastive projection
    """

    def __init__(self, in_channels=4, d_model=128, nhead=8, num_layers=4,
                 n_bands=8, patch_lens=[7, 14, 30], stride=7,
                 dropout=0.3, proj_dim=64, use_prior=False):
        super().__init__()
        self.d_model = d_model
        self.use_prior = use_prior

        n_scales = len(patch_lens)
        d_patch = d_model

        self.patch_embed = nn.ModuleList([
            nn.Conv1d(in_channels, d_model, kernel_size=pl, stride=stride)
            for pl in patch_lens
        ])
        self.patch_norm = nn.LayerNorm(d_model)

        max_patches = 200
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_patches, d_model) * 0.02
        )

        self.temporal_encoder = TemporalTransformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dropout=dropout, max_seq_len=max_patches,
        )

        self.frequency_stream = FrequencyStream(
            in_channels, d_model, n_bands, dropout,
        )

        self.cross_fusion = CrossAttentionFusion(d_model, nhead, dropout)

        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        cls_input_dim = d_model
        if use_prior:
            cls_input_dim += 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
        )

        self.reconstructor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, in_channels),
        )

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x, return_multi=False, prior=None):
        """
        Args:
            x: [B, C, T]
            return_multi: return all task outputs
            prior: [B] GBDT OOF probabilities

        Returns:
            If return_multi: (logit, recon_target, proj)
            Else: logit
        """
        B, C, T = x.shape

        all_tokens = []
        for embed in self.patch_embed:
            tokens = embed(x)  # [B, d_model, n_patches]
            tokens = tokens.transpose(1, 2)  # [B, n_patches, d_model]
            all_tokens.append(tokens)

        min_patches = min(t.shape[1] for t in all_tokens)
        all_tokens = [t[:, :min_patches, :] for t in all_tokens]
        patch_tokens = torch.mean(torch.stack(all_tokens), dim=0)  # [B, S, D]
        patch_tokens = self.patch_norm(patch_tokens)

        S = patch_tokens.shape[1]
        patch_tokens = patch_tokens + self.pos_embedding[:, :S, :]

        temporal_out = self.temporal_encoder(patch_tokens)  # [B, S, D]

        freq_out = self.frequency_stream(x)  # [B, n_bands, D]

        fused = self.cross_fusion(temporal_out, freq_out)  # [B, S, D]

        pooled = fused.mean(dim=1)  # [B, D]
        pooled = self.global_pool(pooled)

        if self.use_prior and prior is not None:
            cls_input = torch.cat([pooled, prior.reshape(-1, 1)], dim=1)
        else:
            cls_input = pooled
        logit = self.classifier(cls_input)

        if return_multi:
            recon_input = pooled.unsqueeze(1).expand(-1, S, -1)
            recon_target = self.reconstructor(recon_input)  # [B, S, C]
            recon_target = recon_target.mean(dim=1)  # [B, C]
            proj = F.normalize(self.projection(pooled), dim=1)
            return logit.squeeze(-1), recon_target, proj

        return logit.squeeze(-1)


def train_transformer(X_seq, y, oof_prior=None,
                       d_model=128, nhead=8, num_layers=4, n_bands=8,
                       patch_lens=None, stride=7, dropout=0.3, proj_dim=64,
                       epochs=50, batch_size=128, lr=3e-4,
                       supcon_weight=0.3, recon_weight=0.2,
                       sce_alpha=1.0, sce_beta=0.5,
                       device='cuda', seed=42, verbose=True):
    """Train multi-scale time-frequency transformer with multi-task learning.

    L = L_SCE(cls) + recon_weight * L_MSE(recon) + supcon_weight * L_SupCon(proj)

    Args:
        X_seq: [N, C, T] multi-channel time series
        y: [N] binary labels
        oof_prior: [N] GBDT OOF probabilities (optional)
        d_model: transformer hidden dimension
        nhead: number of attention heads
        num_layers: transformer encoder layers
        n_bands: number of frequency bands
        patch_lens: multi-scale patch lengths
        stride: patch stride
        dropout: dropout rate
        proj_dim: contrastive projection dimension
        epochs: max training epochs
        batch_size: training batch size (≥128 for contrastive learning)
        lr: learning rate
        supcon_weight: weight for contrastive loss
        recon_weight: weight for reconstruction loss
        sce_alpha: alpha for SCE loss
        sce_beta: beta for SCE loss
        device: 'cuda' or 'cpu'
        seed: random seed
        verbose: print progress

    Returns:
        trained model
    """
    from sklearn.metrics import f1_score, roc_auc_score

    if patch_lens is None:
        patch_lens = [7, 14, 30]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    N = len(y)
    in_ch = X_seq.shape[1]
    use_prior = oof_prior is not None

    model = MultiScaleTFTTransformer(
        in_channels=in_ch, d_model=d_model, nhead=nhead,
        num_layers=num_layers, n_bands=n_bands,
        patch_lens=patch_lens, stride=stride, dropout=dropout,
        proj_dim=proj_dim, use_prior=use_prior,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Transformer: {n_params:,} params (prior={use_prior})")

    cls_criterion = SymmetricCrossEntropy(alpha=sce_alpha, beta=sce_beta)
    supcon_criterion = SupConLoss(temperature=0.07)
    recon_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    X_t = torch.FloatTensor(X_seq)
    y_t = torch.FloatTensor(y)

    if use_prior:
        p_t = torch.FloatTensor(oof_prior)
        dataset = TensorDataset(X_t, y_t, p_t)
    else:
        dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         drop_last=True, num_workers=0)

    best_f1 = 0
    best_state = None
    patience = 0
    max_patience = 12

    for epoch in range(epochs):
        model.train()
        total_cls = 0
        total_con = 0
        total_rec = 0
        n_batches = 0

        for batch_data in loader:
            if use_prior:
                batch_x, batch_y, batch_p = batch_data
                batch_p = batch_p.to(device)
            else:
                batch_x, batch_y = batch_data
                batch_p = None
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits, recon, proj = model(batch_x, return_multi=True, prior=batch_p)

            cls_loss = cls_criterion(logits, batch_y)

            batch_mean = batch_x.mean(dim=2)  # [B, C]
            recon_loss = recon_criterion(recon, batch_mean)

            n_pos = (batch_y == 1).sum().item()
            n_neg = (batch_y == 0).sum().item()
            if n_pos >= 2 and n_neg >= 2:
                con_loss = supcon_criterion(proj, batch_y)
            else:
                con_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + recon_weight * recon_loss + supcon_weight * con_loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_cls += cls_loss.item()
            total_con += con_loss.item()
            total_rec += recon_loss.item()
            n_batches += 1

        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                all_probs = []
                for start in range(0, N, 512):
                    end = min(start + 512, N)
                    xb = torch.FloatTensor(X_seq[start:end]).to(device)
                    pb = torch.FloatTensor(oof_prior[start:end]).to(device) if use_prior else None
                    logits = model(xb, prior=pb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                    all_probs.append(probs)
                all_probs = np.concatenate(all_probs)
                all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)

            auc = roc_auc_score(y, all_probs)
            best_th = 0.5
            best_f1_val = 0
            for th in np.arange(0.1, 0.9, 0.01):
                pred = (all_probs > th).astype(int)
                if pred.sum() == 0:
                    continue
                f1 = f1_score(y, pred, zero_division=0)
                if f1 > best_f1_val:
                    best_f1_val = f1
                    best_th = th

            print(f"    Epoch {epoch+1}: cls={total_cls/max(n_batches,1):.4f} "
                  f"con={total_con/max(n_batches,1):.4f} "
                  f"rec={total_rec/max(n_batches,1):.4f} "
                  f"F1={best_f1_val:.4f} AUC={auc:.4f} th={best_th:.2f}")

            if best_f1_val > best_f1:
                best_f1 = best_f1_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if patience >= max_patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_transformer(model, X_seq, oof_prior=None, batch_size=512, device='cuda'):
    """Get predictions from transformer model."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            end = min(start + batch_size, len(X_seq))
            xb = torch.FloatTensor(X_seq[start:end]).to(device)
            pb = torch.FloatTensor(oof_prior[start:end]).to(device) if oof_prior is not None else None
            logits = model(xb, prior=pb)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs)


def train_transformer_cv(X_seq, y, oof_prior=None, n_folds=5, seed=42,
                          device='cuda', **kwargs):
    """Train transformer with K-fold CV. Returns OOF predictions."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score
    from src.utils.utils import best_f1_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
        print(f"\n  Transformer Fold {fold_idx+1}/{n_folds}")

        model = train_transformer(
            X_seq[train_idx], y[train_idx],
            oof_prior=oof_prior[train_idx] if oof_prior is not None else None,
            device=device, seed=seed + fold_idx, verbose=True, **kwargs
        )

        val_probs = predict_transformer(
            model, X_seq[val_idx],
            oof_prior=oof_prior[val_idx] if oof_prior is not None else None,
            device=device
        )
        oof_proba[val_idx] = val_probs

        f1, th, rec, prec = best_f1_score(y[val_idx], val_probs)
        auc = roc_auc_score(y[val_idx], val_probs)
        print(f"  Fold {fold_idx+1}: F1={f1:.4f} AUC={auc:.4f} "
              f"Rec={rec:.4f} Prec={prec:.4f} th={th:.3f}")

        del model
        torch.cuda.empty_cache()

    return oof_proba


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("Testing Multi-Scale Time-Frequency Transformer...")
    N, C, T = 64, 4, 100
    np.random.seed(42)
    X = np.random.randn(N, C, T).astype(np.float32)
    y = (np.random.rand(N) < 0.15).astype(np.float32)
    y[:8] = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = train_transformer(X, y, epochs=10, batch_size=16, device=device)
    probs = predict_transformer(model, X, device=device)
    print(f"Output: {probs.shape}, range=[{probs.min():.4f}, {probs.max():.4f}]")
    print("Transformer test passed!")
