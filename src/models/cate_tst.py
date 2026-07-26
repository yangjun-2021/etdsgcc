"""CATE-TST: Causal Transformer + LSTM for electricity theft.

Backbone: per-channel BiLSTM (local) + iTransformer channel attention (global)
    cross-attention fusion.
Heads: outcome_normal (counterfactual reconstruction), outcome_theft, propensity,
    classifier. DragonNet-style shared encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelBiLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size=32, n_pool=8, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.n_pool = n_pool
        self.lstm = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(n_pool)
        self.proj = nn.Linear(hidden_size * 2 * n_pool, hidden_size * 2)

    def forward(self, x):
        b, c, t = x.shape
        h = x.reshape(b * c, t, 1)
        h, _ = self.lstm(h)
        h = self.dropout(self.norm(h))
        h = h.transpose(1, 2)
        h = self.pool(h).reshape(b, c, -1)
        return self.proj(h)


class InvertedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h)
        x = x + self.drop(a)
        h = self.n2(x)
        x = x + self.drop(self.ffn(h))
        return x


class CrossFusion(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, q, kv):
        a, _ = self.attn(self.norm(q), kv, kv)
        g = self.gate(torch.cat([q.mean(1, keepdim=True), a.mean(1, keepdim=True)], dim=-1))
        return a + g * q


class SegmentDecoder(nn.Module):
    def __init__(self, d_model, in_channels, n_segments=32, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.n_segments = n_segments
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, in_channels * n_segments),
        )

    def forward(self, z, target_len):
        b = z.shape[0]
        y = self.head(z).reshape(b, self.in_channels, self.n_segments)
        return F.interpolate(y, size=target_len, mode='linear', align_corners=False)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim=1, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


class CATETSTModel(nn.Module):
    def __init__(self, in_channels=5, seq_len=1035, d_model=96, n_layers=2,
                 n_heads=4, lstm_hidden=32, lstm_pool=8, dropout=0.2, recon_segments=32):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.recon_segments = recon_segments

        self.lstm_branch = ChannelBiLSTM(in_channels, lstm_hidden, lstm_pool, dropout)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, d_model)

        self.channel_embed = nn.Parameter(torch.randn(1, in_channels, d_model) * 0.02)
        self.channel_stats_proj = nn.Linear(6, d_model)
        self.itrans = nn.ModuleList([
            InvertedTransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.fusion = CrossFusion(d_model, n_heads, dropout)
        self.pool_norm = nn.LayerNorm(d_model)

        self.decoder_normal = SegmentDecoder(d_model, in_channels, recon_segments, dropout)
        self.decoder_theft = SegmentDecoder(d_model, in_channels, recon_segments, dropout)

        self.propensity = MLP(d_model, d_model, 1, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + in_channels * 4 + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _stats(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True).clamp_min(1e-6)
        mn = x.amin(dim=2, keepdim=True)
        mx = x.amax(dim=2, keepdim=True)
        q25 = torch.quantile(x, 0.25, dim=2, keepdim=True)
        q75 = torch.quantile(x, 0.75, dim=2, keepdim=True)
        return torch.cat([mean, std, mn, mx, q25, q75], dim=2)

    def _residual(self, x, recon):
        err = x - recon
        return torch.cat([
            err.mean(dim=2),
            err.std(dim=2),
            err.abs().mean(dim=2),
            err.pow(2).mean(dim=2),
        ], dim=1)

    def encode(self, x):
        lstm_tok = self.lstm_proj(self.lstm_branch(x))
        stat_tok = self.channel_stats_proj(self._stats(x)) + self.channel_embed
        for blk in self.itrans:
            stat_tok = blk(stat_tok)
        fused = self.fusion(stat_tok, lstm_tok)
        pooled = self.pool_norm(fused.mean(dim=1))
        return pooled, fused

    def forward(self, x, return_details=False):
        pooled, fused = self.encode(x)
        recon_normal = self.decoder_normal(pooled, x.shape[-1])
        recon_theft = self.decoder_theft(pooled, x.shape[-1])
        prop_logit = self.propensity(pooled)
        residual = self._residual(x, recon_normal)
        cls_in = torch.cat([pooled, residual, prop_logit.unsqueeze(-1)], dim=1)
        logit = self.classifier(cls_in).squeeze(-1)
        if return_details:
            return {
                'logit': logit,
                'propensity_logit': prop_logit,
                'recon_normal': recon_normal,
                'recon_theft': recon_theft,
                'residual_normal': residual,
                'embedding': pooled,
                'fused_channels': fused,
            }
        return logit, recon_normal, prop_logit


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma_pos=1.0, gamma_neg=3.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        pw = None
        if self.pos_weight is not None:
            pw = torch.as_tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pw)
        p = torch.sigmoid(logits)
        loss = self.alpha * targets * (1 - p).pow(self.gamma_pos) * bce \
            + (1 - self.alpha) * (1 - targets) * p.pow(self.gamma_neg) * bce
        return loss.mean()


def mmd_loss(z_a, z_b, sigma=1.0):
    if z_a.numel() == 0 or z_b.numel() == 0:
        return z_a.new_zeros(())

    def _kernel(x, y):
        xx = (x * x).sum(dim=1, keepdim=True)
        yy = (y * y).sum(dim=1, keepdim=True)
        d = xx + yy.t() - 2 * x @ y.t()
        return torch.exp(-d / (2 * sigma * sigma + 1e-8))

    kaa = _kernel(z_a, z_a).mean()
    kbb = _kernel(z_b, z_b).mean()
    kab = _kernel(z_a, z_b).mean()
    return kaa + kbb - 2 * kab
