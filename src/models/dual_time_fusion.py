"""DualTimeFusion: TCN (short-term) + LSTM (long-term) + DCNN (multi-scale).

Based on Huang et al. (2024, Energies) "Dual-Time Fusion" architecture adapted
for our SGCC preprocessing pipeline with identical 5-fold CV splits for fair comparison.

Architecture:
  Short branch: 4-layer causal dilated TCN [64,128,256,128] on raw 1034-length sequence
  Long branch: 2-layer BiLSTM on downsampled 256-length sequence
  DCNN branch: Multi-scale dilated convolutions (rates 1,2,4,8)
  Fusion: Channel attention + concatenation → classifier head
  
Parameters: ~200K (comparable to Expert B's 94K, larger for dual-time modeling)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDilatedBlock(nn.Module):
    """Causal dilated conv block with residual connection."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out[..., :x.shape[-1]]  # causal crop
        return F.relu(out + self.residual(x))


class TCNShortBranch(nn.Module):
    """Short-term TCN captures daily/weekly consumption patterns."""
    def __init__(self, in_channels=5, hidden_dims=(64,128,256,128), kernel_size=5, dropout=0.2):
        super().__init__()
        layers = []
        dims = [in_channels] + list(hidden_dims)
        for i in range(len(hidden_dims)):
            layers.append(CausalDilatedBlock(dims[i], dims[i+1], kernel_size, 2**i, dropout))
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.net(x)
        return self.pool(out).squeeze(-1)


class LSTMLongBranch(nn.Module):
    """Long-term BiLSTM captures monthly/seasonal consumption patterns."""
    def __init__(self, in_channels=5, seq_len=256, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0)
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        # x: (B, C, T) → downsample temporal → (B, T_down, C)
        x = F.adaptive_avg_pool1d(x, self.seq_len).transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.norm(out.mean(dim=1))
        return out


class DCNNMultiScale(nn.Module):
    """Multi-scale dilated CNN for cross-resolution feature extraction."""
    def __init__(self, in_channels=5, base_dim=32, dilations=(1,2,4,8,16), dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, base_dim, 3, dilation=d, padding=d),
                nn.BatchNorm1d(base_dim), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            ))
        self.fusion = nn.Linear(base_dim * len(dilations), base_dim * 2)

    def forward(self, x):
        features = [branch(x).squeeze(-1) for branch in self.branches]
        return self.fusion(torch.cat(features, dim=1))


class DualTimeFusionModel(nn.Module):
    """Dual-time-scale fusion model for electricity theft detection.

    Short branch (TCN): daily/weekly patterns from raw 1034-length seq
    Long branch (LSTM): monthly/seasonal patterns from downsampled 256-length seq
    DCNN branch: multi-scale dilated convolutions
    Fusion: concatenation → MLP classifier
    """
    def __init__(self, in_channels=5, seq_len=1034, d_model=128,
                 stat_dim=353, dropout=0.2):
        super().__init__()
        self.short_branch = TCNShortBranch(in_channels, (32,64,96,64), dropout=dropout)
        self.long_branch = LSTMLongBranch(in_channels, 128, 32, 1, dropout)
        self.dcnn = DCNNMultiScale(in_channels, 16, (1,2,4,8), dropout)
        fusion_dim = 64 + 64 + 32
        self.stat_proj = nn.Linear(stat_dim, 64)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim + 64, d_model),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, stat_features=None):
        short = self.short_branch(x)
        long_f = self.long_branch(x)
        dcnn_f = self.dcnn(x)
        fused = torch.cat([short, long_f, dcnn_f], dim=1)

        if stat_features is not None and stat_features.shape[1] > 0:
            stat_f = self.stat_proj(stat_features)
            fused = torch.cat([fused, stat_f], dim=1)

        return self.fusion(fused).squeeze(-1)


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma_pos=1.0, gamma_neg=3.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha; self.gamma_pos = gamma_pos; self.gamma_neg = gamma_neg
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        pw = torch.as_tensor(float(self.pos_weight or 1), dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pw)
        p = torch.sigmoid(logits)
        loss = self.alpha * targets * (1-p).pow(self.gamma_pos)*bce \
               + (1-self.alpha) * (1-targets) * p.pow(self.gamma_neg)*bce
        return loss.mean()
