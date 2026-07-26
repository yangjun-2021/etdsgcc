"""
ADL-Net: Anomaly Dictionary Learning Network for Electricity Theft Detection.

A fully end-to-end model that does NOT rely on any GBDT/Expert-A prior.
Core components:
1. SparseDictionary: learnable dictionary of normal load patterns.
2. ADLEncoder: CNN + Transformer encoder for raw/residual/difference series.
3. MoCoHead: momentum-contrastive learning head.
4. ADLNet: combines all modules into a classifier.
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# 1. Sparse Dictionary (differentiable ISTA soft-thresholding)
# ------------------------------------------------------------------------------
class SparseDictionary(nn.Module):
    """Learn a dictionary D of normal patterns and infer sparse codes."""

    def __init__(self, atom_dim, n_atoms=256, sparsity=0.1, n_iter=5):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_dim = atom_dim
        self.sparsity = sparsity
        self.n_iter = n_iter

        # Initialize dictionary atoms (L2 normalized per atom)
        D = torch.randn(n_atoms, atom_dim)
        D = F.normalize(D, dim=1)
        self.D = nn.Parameter(D)

    def forward(self, x_flat):
        """
        x_flat: [B, atom_dim]
        Returns:
            x_hat:  [B, atom_dim]
            code:   [B, n_atoms]
            residual: [B, atom_dim]
        """
        B = x_flat.shape[0]
        code = torch.zeros(B, self.n_atoms, device=x_flat.device, dtype=x_flat.dtype)

        # Simple ISTA with gradient norm (2*L = 2 * max eigenvalue of D^T D)
        Dt = self.D.t()
        L = torch.linalg.matrix_norm(Dt @ self.D, ord=2).detach()
        step = 0.9 / (L + 1e-8)

        for _ in range(self.n_iter):
            grad = (code @ self.D - x_flat) @ Dt
            code = code - step * grad
            code = F.softshrink(code, self.sparsity)

        x_hat = code @ self.D
        residual = x_flat - x_hat
        return x_hat, code, residual

    def normalize_atoms(self):
        """Keep atoms on unit sphere after each update."""
        with torch.no_grad():
            self.D.data = F.normalize(self.D.data, dim=1)


# ------------------------------------------------------------------------------
# 2. CNN + Transformer encoder
# ------------------------------------------------------------------------------
class ConvDownsample(nn.Module):
    """Lightweight CNN stem: [B, C, T] -> [B, d_model, T'] -> [B, T', d_model]."""

    def __init__(self, in_channels, d_model, kernel_size=5, stride=2, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, d_model // 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, C, T]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x)  # further downsample
        x = x.transpose(1, 2)  # [B, T', d_model]
        return x


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.ffn(x)
        x = self.norm2(x + self.dropout(x2))
        return x


class ADLEncoder(nn.Module):
    """Encode original, residual, and difference series into a compact embedding."""

    def __init__(self, in_channels, seq_len, d_model=256, n_layers=4, n_heads=8, dropout=0.2):
        super().__init__()
        self.cnn = ConvDownsample(in_channels * 3, d_model, dropout=dropout)
        self.pe = FixedPositionalEncoding(d_model, max_len=seq_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_orig, x_res, x_diff):
        x = torch.cat([x_orig, x_res, x_diff], dim=1)  # [B, 3*C, T]
        x = self.cnn(x)  # [B, T', d_model]
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)  # [B, d_model, T']
        x = self.pool(x).squeeze(-1)  # [B, d_model]
        return x


# ------------------------------------------------------------------------------
# 3. MoCo-style contrastive head
# ------------------------------------------------------------------------------
class MoCoHead(nn.Module):
    def __init__(self, d_model, proj_dim=128, K=4096, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.register_buffer("queue", torch.randn(proj_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if self.K % batch_size != 0:
            # Replace in-place circular buffer
            space = self.K - ptr
            if space >= batch_size:
                self.queue[:, ptr:ptr + batch_size] = keys.T
                ptr = (ptr + batch_size) % self.K
            else:
                self.queue[:, ptr:] = keys.T[:, :space]
                self.queue[:, :batch_size - space] = keys.T[:, space:]
                ptr = (batch_size - space) % self.K
            self.queue_ptr[0] = ptr
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
            self.queue_ptr[0] = ptr

    def forward(self, q, k=None, labels=None):
        """
        q: [B, d_model] query embeddings
        k: [B, d_model] key embeddings (momentum view of same samples)
        labels: optional for supervised contrastive (not used here)
        Returns loss if k is provided, else just logits.
        """
        q = F.normalize(self.encoder_q(q), dim=1)
        if k is not None:
            with torch.no_grad():
                self._momentum_update()
                k = F.normalize(self.encoder_k(k), dim=1)
            # positive logits: q · k
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B, 1]
            # negative logits: q · queue
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [B, K]
            logits = torch.cat([l_pos, l_neg], dim=1) / self.T  # [B, 1+K]
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
            self._dequeue_and_enqueue(k)
            return loss
        return q


# ------------------------------------------------------------------------------
# 4. ADL-Net full model
# ------------------------------------------------------------------------------
class ADLNet(nn.Module):
    def __init__(self, in_channels, seq_len, n_atoms=256, sparsity=0.1,
                 d_model=256, n_layers=4, n_heads=8, dropout=0.2,
                 proj_dim=128, queue_size=4096, temperature=0.07):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        atom_dim = in_channels * seq_len
        self.dictionary = SparseDictionary(atom_dim, n_atoms=n_atoms, sparsity=sparsity)
        self.encoder = ADLEncoder(in_channels, seq_len, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.moco = MoCoHead(d_model, proj_dim=proj_dim, K=queue_size, T=temperature)

        # Residual and code statistics projector
        self.residual_stats_dim = in_channels * 3  # mean, std, max per channel
        self.code_stats_dim = 6  # mean, std, max, sparsity, entropy, active_atoms
        self.stats_proj = nn.Sequential(
            nn.Linear(self.residual_stats_dim + self.code_stats_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_difference(self, x):
        # x: [B, C, T]
        diff = torch.diff(x, dim=2, prepend=x[:, :, :1])
        return diff

    def compute_stats(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        mx = x.amax(dim=2)
        return torch.cat([mean, std, mx], dim=1)

    def code_stats(self, code):
        # code: [B, K]
        mean = code.mean(dim=1, keepdim=True)
        std = code.std(dim=1, keepdim=True)
        mx = code.amax(dim=1, keepdim=True)
        sparsity = (code.abs() > 1e-4).float().mean(dim=1, keepdim=True)
        # entropy over normalized absolute coefficients
        p = F.softmax(code.abs(), dim=1)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1, keepdim=True)
        active = (code.abs() > 1e-4).float().sum(dim=1, keepdim=True)
        return torch.cat([mean, std, mx, sparsity, entropy, active], dim=1)

    def forward(self, x_orig, x_view=None, return_embedding=False):
        """
        x_orig: [B, C, T] original series
        x_view: [B, C, T] augmented view (for contrastive learning)
        """
        B, C, T = x_orig.shape
        x_flat = x_orig.view(B, -1)
        x_hat, code, residual = self.dictionary(x_flat)

        x_res = residual.view(B, C, T)
        x_diff = self.compute_difference(x_orig)

        emb = self.encoder(x_orig, x_res, x_diff)  # [B, d_model]

        # stats
        res_stats = self.compute_stats(x_res)
        c_stats = self.code_stats(code)
        stats = self.stats_proj(torch.cat([res_stats, c_stats], dim=1))  # [B, 64]

        logit = self.classifier(torch.cat([emb, stats], dim=1)).squeeze(-1)

        if x_view is not None:
            with torch.no_grad():
                x_view_flat = x_view.view(B, -1)
                with torch.no_grad():
                    x_hat_view, code_view, residual_view = self.dictionary(x_view_flat)
                x_view_res = residual_view.view(B, C, T)
                x_view_diff = self.compute_difference(x_view)
                emb_k = self.encoder(x_view, x_view_res, x_view_diff)
            contrast_loss = self.moco(emb, emb_k)
        else:
            contrast_loss = None

        if return_embedding:
            return logit, emb, x_hat, residual, code, contrast_loss
        return logit, contrast_loss


# ------------------------------------------------------------------------------
# 5. Loss helpers
# ------------------------------------------------------------------------------
def focal_loss(logit, y, alpha=0.75, gamma=2.0, pos_weight=None, label_smoothing=0.0):
    """Focal loss with BCE; supports class weight."""
    y_smooth = y * (1 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(logit, y_smooth, reduction='none', pos_weight=pos_weight)
    p = torch.sigmoid(logit)
    p_t = p * y + (1 - p) * (1 - y)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    loss = (alpha_t * (1 - p_t) ** gamma * bce).mean()
    return loss


if __name__ == '__main__':
    B, C, T = 8, 5, 259
    x = torch.randn(B, C, T)
    x_view = torch.randn(B, C, T)
    model = ADLNet(in_channels=C, seq_len=T, n_atoms=128, d_model=64, n_layers=2, n_heads=4, queue_size=256)
    logit, loss = model(x, x_view)
    print("logit:", logit.shape)
    print("contrast_loss:", loss.item())
    logit2, *_ = model(x)
    print("logit (no view):", logit2.shape)
