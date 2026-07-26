import numpy as np
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 padding=0, dropout=0.3):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.chomp = padding
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.chomp2 = padding
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu_out = nn.GELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if isinstance(self.downsample, nn.Conv1d):
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        res = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp].contiguous()
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.chomp2 > 0:
            out = out[:, :, :-self.chomp2].contiguous()
        out = self.relu2(out)
        out = self.dropout2(out)

        out = out + res
        out = self.relu_out(out)
        return out


class TCNEncoder(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size=7, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = in_channels
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size,
                dilation=dilation, padding=padding, dropout=dropout
            ))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.out_channels = num_channels[-1] if num_channels else in_channels

    def forward(self, x):
        return self.network(x)


class LeafEmbedding(nn.Module):
    def __init__(self, n_trees, num_leaves, embed_dim=4, output_dim=64):
        super().__init__()
        self.n_trees = n_trees
        self.num_leaves = num_leaves
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.embedding = nn.ModuleList([
            nn.Embedding(num_leaves, embed_dim)
            for _ in range(n_trees)
        ])
        self.proj = nn.Linear(n_trees * embed_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, leaf_indices):
        if leaf_indices is None:
            return None
        embs = []
        for i in range(self.n_trees):
            idx = leaf_indices[:, i].clamp(0, self.num_leaves - 1)
            embs.append(self.embedding[i](idx))
        x = torch.cat(embs, dim=-1)
        x = self.proj(x)
        x = self.layer_norm(x)
        return x


class TCNWithLeafEmbedding(nn.Module):
    def __init__(self, in_channels, tcn_channels, kernel_size=7, dropout=0.3,
                 n_trees=100, num_leaves=31, leaf_embed_dim=4, leaf_output_dim=64,
                 use_prior=False):
        super().__init__()
        self.tcn = TCNEncoder(in_channels, tcn_channels, kernel_size, dropout)
        tcn_out = tcn_channels[-1] if tcn_channels else in_channels

        self.leaf_embed = LeafEmbedding(n_trees, num_leaves, leaf_embed_dim, leaf_output_dim)

        self.gate = nn.Sequential(
            nn.Linear(leaf_output_dim, leaf_output_dim),
            nn.Sigmoid()
        )

        self.use_prior = use_prior
        prior_dim = 1 if use_prior else 0
        self.classifier = nn.Sequential(
            nn.Linear(tcn_out + leaf_output_dim + prior_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

        self.use_leaf = True

    def forward(self, x, leaf_indices=None, prior=None):
        tcn_out = self.tcn(x)
        pooled = torch.mean(tcn_out, dim=2)

        parts = [pooled]

        if leaf_indices is not None and self.use_leaf:
            leaf_emb = self.leaf_embed(leaf_indices)
            gate = self.gate(leaf_emb)
            leaf_emb = leaf_emb * gate
            parts.append(leaf_emb)

        if self.use_prior and prior is not None:
            parts.append(prior.reshape(-1, 1))

        combined = torch.cat(parts, dim=1)

        logit = self.classifier(combined)
        return logit.squeeze(-1)


class RecallOrientedFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, recall_weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.recall_weight = recall_weight

    def forward(self, logits, targets, pos_weight=None, label_smoothing=0.0):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)

        # Label smoothing
        if label_smoothing > 0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

        # Standard focal loss
        loss_pos = -self.alpha * torch.pow(1.0 - probs, self.gamma) * torch.log(probs) * targets
        loss_neg = -(1.0 - self.alpha) * torch.pow(probs, self.gamma) * torch.log(1.0 - probs) * (1.0 - targets)

        # Optional positive weighting (e.g. for class imbalance)
        if pos_weight is not None:
            loss_pos = loss_pos * pos_weight

        # Extra recall-oriented penalty on false negatives
        fn_penalty = self.recall_weight * torch.pow(1.0 - probs, 2) * targets * (-torch.log(probs + 1e-8))

        loss = loss_pos + loss_neg + fn_penalty
        return loss.mean()


class GeneralizedCrossEntropyLoss(nn.Module):
    """Generalized Cross Entropy (Zhang & Sabuncu, 2018) for robust label-noise training.

    For binary classification:
        L_q = ((1 - p_y)^q) / q,
    where p_y = p for y=1 and 1-p for y=0.  q -> 0 recovers CE; q in (0,1)
    is more robust to noisy labels.
    """
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets, pos_weight=None, label_smoothing=0.0):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        if label_smoothing > 0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        p_y = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = (torch.pow(1.0 - p_y, self.q)) / (self.q + 1e-8)
        return loss.mean()


class RecallOrientedGCELoss(nn.Module):
    """GCE + extra false-negative penalty to keep recall high."""
    def __init__(self, q=0.7, recall_weight=3.0):
        super().__init__()
        self.q = q
        self.recall_weight = recall_weight

    def forward(self, logits, targets, pos_weight=None, label_smoothing=0.0):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        if label_smoothing > 0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        p_y = probs * targets + (1.0 - probs) * (1.0 - targets)
        gce = torch.pow(1.0 - p_y, self.q) / (self.q + 1e-8)
        fn_penalty = self.recall_weight * torch.pow(1.0 - probs, 2) * targets * (-torch.log(probs + 1e-8))
        return (gce + fn_penalty).mean()


def create_model(config_name='sgcc', in_channels=4):
    from config import SGCC_CONFIG, OEDI_CONFIG
    if config_name == 'sgcc':
        cfg = SGCC_CONFIG
    else:
        cfg = OEDI_CONFIG

    tcn_params = cfg['tcn_params']
    model = TCNWithLeafEmbedding(
        in_channels=in_channels,
        tcn_channels=tcn_params['num_channels'],
        kernel_size=tcn_params['kernel_size'],
        dropout=tcn_params['dropout'],
        n_trees=tcn_params['n_trees'],
        num_leaves=tcn_params['num_leaves'],
        leaf_embed_dim=tcn_params['leaf_embed_dim'],
        leaf_output_dim=tcn_params['leaf_embed_dim'],
    )
    return model


if __name__ == '__main__':
    from config import SGCC_CONFIG, OEDI_CONFIG

    print("Testing SGCC model...")
    model_sgcc = create_model('sgcc', in_channels=4)
    x = torch.randn(8, 4, 1035)
    leaf = torch.randint(0, 64, (8, 200))
    out = model_sgcc(x, leaf)
    print(f"  Input: {x.shape}, Leaf: {leaf.shape}")
    print(f"  Output: {out.shape}")
    n_params = sum(p.numel() for p in model_sgcc.parameters())
    print(f"  Parameters: {n_params:,}")

    print("\nTesting OEDI model...")
    model_oedi = create_model('oedi', in_channels=11)
    x = torch.randn(8, 11, 720)
    leaf = torch.randint(0, 64, (8, 200))
    out = model_oedi(x, leaf)
    print(f"  Input: {x.shape}, Leaf: {leaf.shape}")
    print(f"  Output: {out.shape}")
    n_params = sum(p.numel() for p in model_oedi.parameters())
    print(f"  Parameters: {n_params:,}")