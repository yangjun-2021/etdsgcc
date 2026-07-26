import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer(nn.Module):
    def __init__(self, in_channels, seq_len, patch_len=30, stride=15, d_model=128, max_patches=512):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
        self.channel_embedding = nn.Parameter(torch.randn(1, in_channels, 1, d_model) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, 1, max_patches, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def _pad(self, x):
        t = x.shape[-1]
        if t < self.patch_len:
            return F.pad(x, (0, self.patch_len - t)), t
        remainder = (t - self.patch_len) % self.stride
        if remainder == 0:
            return x, t
        return F.pad(x, (0, self.stride - remainder)), t

    def forward(self, x, token_mask=None):
        x, original_len = self._pad(x)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        tokens = self.proj(patches)
        n_patches = tokens.shape[2]
        tokens = tokens + self.channel_embedding[:, :tokens.shape[1]]
        tokens = tokens + self.position_embedding[:, :, :n_patches]
        if token_mask is not None:
            mask = token_mask[:, :, :n_patches].unsqueeze(-1)
            tokens = torch.where(mask, self.mask_token.to(tokens.dtype), tokens)
        tokens = self.norm(tokens)
        b, c, n, d = tokens.shape
        return tokens.reshape(b, c * n, d), n, original_len


class ReconstructionHead(nn.Module):
    def __init__(self, d_model, patch_len):
        super().__init__()
        self.head = nn.Linear(d_model, patch_len)

    def forward(self, tokens, n_channels, n_patches, seq_len, patch_len, stride):
        b = tokens.shape[0]
        patches = self.head(tokens).reshape(b, n_channels, n_patches, patch_len)
        total_len = (n_patches - 1) * stride + patch_len
        recon = patches.new_zeros(b, n_channels, total_len)
        weight = patches.new_zeros(b, n_channels, total_len)
        for i in range(n_patches):
            start = i * stride
            end = start + patch_len
            recon[:, :, start:end] = recon[:, :, start:end] + patches[:, :, i]
            weight[:, :, start:end] = weight[:, :, start:end] + 1
        recon = recon / weight.clamp_min(1)
        return recon[:, :, :seq_len]


class TimeSeriesFoundationModel(nn.Module):
    def __init__(self, in_channels=5, seq_len=1035, patch_len=30, stride=15,
                 d_model=128, n_layers=4, n_heads=8, dropout=0.2,
                 n_segments=24, use_revin=True):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_segments = n_segments
        self.use_revin = use_revin

        self.tokenizer = PatchTokenizer(in_channels, seq_len, patch_len, stride, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.reconstruction = ReconstructionHead(d_model, patch_len)
        self.anomaly_proj = nn.Sequential(
            nn.Linear(in_channels * 3 + n_segments, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _normalize(self, x):
        if not self.use_revin:
            return x, None, None
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True).clamp_min(1e-5)
        return (x - mean) / std, mean, std

    def _denormalize(self, x, mean, std):
        if mean is None:
            return x
        return x * std + mean

    def _token_mask(self, x, mask_ratio):
        if mask_ratio <= 0:
            return None
        padded, _ = self.tokenizer._pad(x)
        n_patches = padded.unfold(dimension=2, size=self.patch_len, step=self.stride).shape[2]
        return torch.rand(x.shape[0], x.shape[1], n_patches, device=x.device) < mask_ratio

    def _anomaly_features(self, x, recon):
        err = (x - recon).pow(2)
        channel_mean = err.mean(dim=2)
        channel_std = err.std(dim=2)
        channel_max = err.amax(dim=2)
        temporal = F.adaptive_avg_pool1d(err.mean(dim=1, keepdim=True), self.n_segments).squeeze(1)
        return torch.cat([channel_mean, channel_std, channel_max, temporal], dim=1), channel_mean, temporal

    def forward(self, x, mask_ratio=0.0, return_details=False):
        x_norm, mean, std = self._normalize(x)
        token_mask = self._token_mask(x_norm, mask_ratio)
        tokens, n_patches, original_len = self.tokenizer(x_norm, token_mask)
        encoded = self.encoder(tokens)
        recon_norm = self.reconstruction(
            encoded,
            self.in_channels,
            n_patches,
            original_len,
            self.patch_len,
            self.stride,
        )
        recon = self._denormalize(recon_norm, mean, std)
        pooled = encoded.mean(dim=1)
        anomaly_features, channel_scores, temporal_scores = self._anomaly_features(x, recon)
        anomaly_embedding = self.anomaly_proj(anomaly_features)
        logit = self.classifier(torch.cat([pooled, anomaly_embedding], dim=1)).squeeze(-1)
        if return_details:
            return {
                'logit': logit,
                'reconstruction': recon,
                'token_mask': token_mask,
                'channel_scores': channel_scores,
                'temporal_scores': temporal_scores,
                'anomaly_score': anomaly_features.mean(dim=1),
                'embedding': pooled,
            }
        return logit, recon


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma_pos=1.0, gamma_neg=3.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.as_tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight)
        prob = torch.sigmoid(logits)
        pos_loss = self.alpha * targets * (1 - prob).pow(self.gamma_pos) * bce
        neg_loss = (1 - self.alpha) * (1 - targets) * prob.pow(self.gamma_neg) * bce
        return (pos_loss + neg_loss).mean()


def masked_reconstruction_loss(x, recon, mask=None):
    err = (x - recon).pow(2)
    if mask is None:
        return err.mean()
    if mask.shape != err.shape:
        mask = F.interpolate(mask.float(), size=err.shape[-1], mode='nearest').bool()
    return (err * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
