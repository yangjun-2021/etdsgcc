"""
AMST-Net: Adaptive Multi-Scale Mamba-Transformer Network for SGCC ETD.

This module implements the proposed MSMT-Encoder (Multi-Scale Mamba-Transformer)
with Hard-Negative Supervised Contrastive Learning (HN-SupCon).

To keep the code runnable on Windows without the hard-to-install mamba-ssm CUDA
extensions, the Mamba branch uses a lightweight Gated 1D-CNN + GRU approximation.
If `mamba_ssm` is available, replace `MambaFallbackBlock` with the official
`MambaBlock` from src.models.mamba_block (to be implemented).

Input:  [B, C, T] where C = 4 (value, fluctuation, entropy, missing mask)
Output: logit [B] + embedding [B, D] for SupCon
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# 1. Positional Encoding & Time-Frequency Utilities
# ------------------------------------------------------------------------------
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe[None, :, :])  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]


def compute_fft_features(x, n_freq=64):
    """Extract magnitude spectrum from each channel.

    x: [B, C, T]
    returns: [B, C*n_freq]
    """
    B, C, T = x.shape
    # FFT along time axis
    fft = torch.fft.rfft(x, dim=2)  # [B, C, T//2+1]
    mag = torch.abs(fft)
    # Take first n_freq bins (low frequencies)
    if mag.shape[2] >= n_freq:
        mag = mag[:, :, :n_freq]
    else:
        pad = n_freq - mag.shape[2]
        mag = F.pad(mag, (0, pad), mode='constant', value=0)
    return mag.reshape(B, -1)


# ------------------------------------------------------------------------------
# 2. Mamba-like Fallback Block (Windows-compatible)
# ------------------------------------------------------------------------------
class MambaFallbackBlock(nn.Module):
    """A lightweight, Mamba-inspired long-range block using gated conv + GRU.

    This is NOT the official Mamba S6 layer, but a drop-in approximation that
    captures local gating and recurrent dynamics with linear complexity.
    """

    def __init__(self, d_model, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        inner = int(d_model * expand)
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)

        self.conv = nn.Conv1d(
            inner, inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=inner, bias=True
        )
        self.act = nn.SiLU()

        self.gru = nn.GRU(inner, inner, batch_first=True, bidirectional=False)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        x_orig = x
        x_in = self.in_proj(x)              # [B, T, 2*inner]
        x_gate, x_main = x_in.chunk(2, dim=-1)  # each [B, T, inner]

        # Local convolution + activation
        x_main = x_main.transpose(1, 2)       # [B, inner, T]
        x_main = self.conv(x_main)[:, :, :T]  # causal conv, keep length
        x_main = x_main.transpose(1, 2)       # [B, T, inner]
        x_main = self.act(x_main) * x_gate

        # Recurrent scan (linear w.r.t. T)
        x_main, _ = self.gru(x_main)          # [B, T, inner]
        x_main = self.out_proj(x_main)        # [B, T, D]
        x_main = self.dropout(x_main)
        return self.norm(x_orig + x_main)


# ------------------------------------------------------------------------------
# 3. CNN-Transformer Branch
# ------------------------------------------------------------------------------
class CNNTransformerBranch(nn.Module):
    """Local CNN feature extraction + Transformer global dependency modeling."""

    def __init__(self, in_channels, d_model=128, n_heads=4, num_layers=2,
                 kernel_size=5, dropout=0.2, max_len=2048):
        super().__init__()
        self.d_model = d_model
        # 1D CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pe = PositionalEncoding1D(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, C, T]
        x = self.cnn(x)                       # [B, D, T]
        x = x.transpose(1, 2)               # [B, T, D]
        x = self.pe(x)
        x = self.transformer(x)               # [B, T, D]
        x = x.transpose(1, 2)               # [B, D, T]
        x = self.pool(x).squeeze(-1)         # [B, D]
        return x


# ------------------------------------------------------------------------------
# 4. Mamba Branch (with fallback)
# ------------------------------------------------------------------------------
class MambaBranch(nn.Module):
    """Mamba-inspired branch for long-range temporal dependencies."""

    def __init__(self, in_channels, d_model=64, num_layers=2, d_conv=4,
                 expand=2, dropout=0.2, downsample=7):
        super().__init__()
        self.downsample = downsample
        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pool = nn.AvgPool1d(downsample, stride=downsample)  # reduce T
        self.blocks = nn.ModuleList([
            MambaFallbackBlock(d_model, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, C, T]
        x = self.proj(x)                      # [B, D, T]
        x = self.pool(x)                      # [B, D, T/downsample]
        x = x.transpose(1, 2)                 # [B, T', D]
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)                 # [B, D, T']
        x = self.out_pool(x).squeeze(-1)       # [B, D]
        return x


# ------------------------------------------------------------------------------
# 5. Frequency Branch
# ------------------------------------------------------------------------------
class FrequencyBranch(nn.Module):
    """1D-CNN over FFT magnitude spectrum."""

    def __init__(self, in_channels, n_freq=64, d_model=64):
        super().__init__()
        self.n_freq = n_freq
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, C, T]
        fft_mag = compute_fft_features(x, n_freq=self.n_freq)  # [B, C*n_freq]
        fft_mag = fft_mag.view(x.shape[0], x.shape[1], self.n_freq)  # [B, C, n_freq]
        out = self.cnn(fft_mag).squeeze(-1)  # [B, d_model]
        out = self.fc(out)
        return out


# ------------------------------------------------------------------------------
# 6. Hard-Negative Supervised Contrastive Loss
# ------------------------------------------------------------------------------
class HardNegativeSupConLoss(nn.Module):
    """Supervised contrastive loss with hard negative mining.

    For each anchor, hardest negatives are selected from the same batch based on
    cosine similarity in the projection space.
    """

    def __init__(self, temperature=0.07, hard_neg_k=5, base_weight=0.2):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_k = hard_neg_k
        self.base_weight = base_weight

    def forward(self, z, labels):
        """
        z:      [B, proj_dim] normalized embeddings
        labels: [B]
        """
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature  # [B, B]
        mask = labels[:, None] == labels[None, :]       # [B, B]
        # Positive mask: same label, exclude diagonal
        pos_mask = mask.clone().float()
        pos_mask.fill_diagonal_(0.0)
        # Negative mask: different label
        neg_mask = (~mask).float()

        # Hard negative mining: select top-k most similar negatives per anchor
        hard_neg_mask = torch.zeros_like(neg_mask)
        if self.hard_neg_k > 0:
            # sim for negatives only
            neg_sim = sim * neg_mask + (-1e9) * (1 - neg_mask)
            _, topk_idx = torch.topk(neg_sim, k=min(self.hard_neg_k, neg_mask.sum(1).max().long().item()), dim=1)
            hard_neg_mask.scatter_(1, topk_idx, 1.0)
        else:
            hard_neg_mask = neg_mask

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        exp_sim = torch.exp(sim - sim_max)

        pos_sum = (exp_sim * pos_mask).sum(dim=1)  # [B]
        neg_sum = (exp_sim * hard_neg_mask).sum(dim=1)  # [B]
        # All positives + all negatives denominator
        denom = pos_sum + neg_sum + 1e-8
        loss = -torch.log((pos_sum + 1e-8) / denom)

        # Only anchors with positives contribute
        has_pos = pos_mask.sum(dim=1) > 0
        return (loss * has_pos).sum() / (has_pos.sum() + 1e-8)


# ------------------------------------------------------------------------------
# 7. Branch Cross-Attention Fusion
# ------------------------------------------------------------------------------
class BranchCrossAttention(nn.Module):
    """Cross-attention fusion across multi-branch representations.

    Each branch (Mamba, Transformer, Frequency) is projected to a common
    dimension and then attends to the others. This lets the model suppress
    noisy branches and amplify discriminative ones per sample, rather than
    relying on a fixed concatenation + FC fusion.
    """

    def __init__(self, branch_dims, d_fusion=128, n_heads=4, dropout=0.2,
                 n_layers=1):
        super().__init__()
        self.branch_dims = branch_dims
        self.d_fusion = d_fusion
        self.n_branches = len(branch_dims)

        # Project each branch to common dimension
        self.branch_projs = nn.ModuleList([
            nn.Linear(d, d_fusion) for d in branch_dims
        ])

        # Self-attention across branch tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_fusion, nhead=n_heads, dim_feedforward=d_fusion * 2,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_fusion)

    def forward(self, branch_feats):
        """
        Args:
            branch_feats: list of [B, D_i] tensors
        Returns:
            [B, d_fusion] fused representation
        """
        # Project and stack: [B, K, D]
        tokens = torch.stack([
            proj(f) for proj, f in zip(self.branch_projs, branch_feats)
        ], dim=1)
        tokens = self.encoder(tokens)  # [B, K, D]
        tokens = self.norm(tokens)
        fused = tokens.mean(dim=1)  # [B, D]
        return fused


# ------------------------------------------------------------------------------
# 8. AMST-Net Main Model
# ------------------------------------------------------------------------------
class AMSTNet(nn.Module):
    """Adaptive Multi-Scale Mamba-Transformer Network.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 5 for SGCC multi-channel).
    seq_len : int
        Input time series length (1035 for SGCC).
    d_mamba : int
        Hidden dim for Mamba branch.
    d_trans : int
        Hidden dim for Transformer branch.
    d_freq : int
        Hidden dim for frequency branch.
    proj_dim : int
        Dimension of SupCon projection head.
    prior_dim : int
        Dimension of optional Expert-A prior to concatenate before classifier.
    use_freq : bool
        Whether to include the FFT branch.
    use_supcon : bool
        Whether to output projection for SupCon loss.
    use_branch_attention : bool
        If True, use cross-attention fusion across branches; otherwise use
        concatenation + FC fusion (legacy behaviour).
    d_fusion : int
        Common dimension for branch cross-attention.
    """

    def __init__(self, in_channels=5, seq_len=1035, d_mamba=128, d_trans=256,
                 d_freq=64, proj_dim=128, n_mamba_layers=2, n_trans_layers=4,
                 n_heads=8, dropout=0.2, use_freq=True, use_supcon=True,
                 prior_dim=1, use_branch_attention=True, d_fusion=128,
                 branch_attn_heads=4, branch_attn_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.use_freq = use_freq
        self.use_supcon = use_supcon
        self.prior_dim = prior_dim
        self.use_branch_attention = use_branch_attention

        self.mamba_branch = MambaBranch(
            in_channels, d_model=d_mamba, num_layers=n_mamba_layers,
            dropout=dropout, downsample=7
        )
        self.trans_branch = CNNTransformerBranch(
            in_channels, d_model=d_trans, n_heads=n_heads,
            num_layers=n_trans_layers, dropout=dropout, max_len=seq_len + 100
        )

        branch_dims = [d_mamba, d_trans]
        if use_freq:
            self.freq_branch = FrequencyBranch(in_channels, d_model=d_freq)
            branch_dims.append(d_freq)
        else:
            self.freq_branch = None

        if use_branch_attention:
            self.branch_fusion = BranchCrossAttention(
                branch_dims=branch_dims,
                d_fusion=d_fusion,
                n_heads=branch_attn_heads,
                dropout=dropout,
                n_layers=branch_attn_layers,
            )
            total_dim = d_fusion
        else:
            total_dim = sum(branch_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.LayerNorm(total_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.branch_fusion = None

        classifier_in = total_dim + prior_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, classifier_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_in // 2, 1),
        )

        if use_supcon:
            self.projector = nn.Sequential(
                nn.Linear(total_dim, total_dim // 2),
                nn.GELU(),
                nn.Linear(total_dim // 2, proj_dim),
            )
        else:
            self.projector = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, prior=None, return_embedding=False):
        """
        x:     [B, C, T]
        prior: [B, prior_dim] optional Expert-A OOF probability
        returns:
            logit: [B]
            z:     [B, proj_dim] if use_supcon and return_embedding
        """
        f_mamba = self.mamba_branch(x)      # [B, d_mamba]
        f_trans = self.trans_branch(x)      # [B, d_trans]

        feats = [f_mamba, f_trans]
        if self.freq_branch is not None:
            f_freq = self.freq_branch(x)    # [B, d_freq]
            feats.append(f_freq)

        if self.use_branch_attention and self.branch_fusion is not None:
            fused = self.branch_fusion(feats)  # [B, d_fusion]
        else:
            fused = torch.cat(feats, dim=1)     # [B, total_dim]
            fused = self.fusion(fused)

        if prior is not None:
            # Broadcast/scalar prior -> [B, prior_dim]
            if prior.dim() == 1:
                prior = prior.unsqueeze(-1)
            if prior.shape[-1] != self.prior_dim:
                prior = F.interpolate(prior.unsqueeze(1), size=self.prior_dim, mode='linear', align_corners=False).squeeze(1)
            fused = torch.cat([fused, prior], dim=1)  # [B, total_dim + prior_dim]
        elif self.prior_dim > 0:
            # No prior provided but model expects one: use zeros (neutral)
            fused = torch.cat([fused, torch.zeros(fused.shape[0], self.prior_dim, device=fused.device, dtype=fused.dtype)], dim=1)

        logit = self.classifier(fused).squeeze(-1)  # [B]

        if return_embedding and self.projector is not None:
            z = self.projector(fused[:, :fused.shape[1] - self.prior_dim] if prior is not None else fused)
            return logit, z
        return logit

    def get_embedding(self, x):
        with torch.no_grad():
            _, z = self.forward(x, return_embedding=True)
            return F.normalize(z, dim=1)


# ------------------------------------------------------------------------------
# 8. Co-Teaching wrapper (two AMST networks with cross-update)
# ------------------------------------------------------------------------------
class CoTeachingAMST:
    """Train two AMST networks simultaneously and exchange clean samples.

    This is a training strategy, not a nn.Module. Use it inside the trainer loop.
    """
    pass  # Implemented in amst_trainer.py


if __name__ == '__main__':
    # Quick sanity check (SGCC: 5 channels)
    B, C, T = 8, 5, 1034
    x = torch.randn(B, C, T)
    prior = torch.rand(B)  # Expert-A OOF probability

    for use_branch_attn in [True, False]:
        print(f"\n--- use_branch_attention={use_branch_attn} ---")
        model = AMSTNet(
            in_channels=C, seq_len=T, use_freq=True, use_supcon=True, prior_dim=1,
            use_branch_attention=use_branch_attn,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"params: {n_params:,}")
        logit, z = model(x, prior=prior, return_embedding=True)
        print("logit:", logit.shape)
        print("embedding:", z.shape)

        # Also test without prior
        logit2 = model(x)
        print("logit (no prior):", logit2.shape)

        # Test SupCon loss
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        criterion = HardNegativeSupConLoss(temperature=0.07, hard_neg_k=2)
        loss = criterion(z, labels)
        print("SupCon loss:", loss.item())
