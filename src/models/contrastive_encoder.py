import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLSTMEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_size=32, d_model=96, num_layers=2, dropout=0.1):
        super().__init__()
        lstm_out = hidden_size * 2
        self.lstm = nn.LSTM(
            1, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fusion = nn.Linear(in_channels * lstm_out, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T = x.shape
        x = x.reshape(B * C, T, 1)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = out.reshape(B, -1)
        return self.norm(self.fusion(out))


class ProjectionHead(nn.Module):
    def __init__(self, d_model, proj_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        return self.net(x).reshape(-1)


class ContrastiveTemporalEncoder(nn.Module):
    def __init__(self, in_channels=5, d_model=96, lstm_hidden=32, lstm_layers=2,
                 proj_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = ChannelLSTMEncoder(in_channels, lstm_hidden, d_model, lstm_layers, dropout)
        self.projector = ProjectionHead(d_model, proj_dim, dropout)
        self.classifier = ClassifierHead(d_model, dropout)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, mode='classify'):
        z = self.encode(x)
        if mode == 'contrastive':
            return F.normalize(self.projector(z), dim=-1, eps=1e-8)
        elif mode == 'embed':
            return z
        return self.classifier(z)


def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0).view(2 * B, 1)
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, float('-inf'))
    neg = sim.logsumexp(dim=1, keepdim=True)
    return -(pos - neg).mean()


def knn_contrastive_loss(z, labels, knn_mask, temperature=0.5):
    """Label-aware contrastive loss with KNN-derived hard negatives.

    z:        (B, D)  normalized embeddings
    labels:   (B,)    class labels (0/1)
    knn_mask: (B, B)  bool, mask[i][j]==True means j is a stat-neighbor of i
    """
    B = z.shape[0]
    if not knn_mask.any():
        return z.new_zeros(())
    sim = torch.mm(z, z.T).div(temperature).clamp(-50, 50)
    sim = sim.masked_fill(torch.eye(B, device=z.device).bool(), float('-inf'))
    label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1))
    pos_mask = knn_mask & label_eq
    has_neighbor = knn_mask.any(dim=1)
    has_pos = pos_mask.any(dim=1)
    valid = has_pos & has_neighbor
    if not valid.any():
        return z.new_zeros(())
    neigh_sim = sim.masked_fill(~knn_mask, -1e4)
    neigh_lse = neigh_sim.logsumexp(dim=1)
    pos_sim = sim.masked_fill(~pos_mask, -1e4)
    pos_lse = pos_sim.logsumexp(dim=1)
    return (neigh_lse - pos_lse)[valid].clamp(max=20.0).mean()


def batch_label_contrastive_loss(z, labels, temperature=0.5):
    """Fallback: use all same-label samples in batch as positives."""
    B = z.shape[0]
    sim = torch.mm(z, z.T).div(temperature).clamp(-50, 50)
    sim = sim.masked_fill(torch.eye(B, device=z.device).bool(), float('-inf'))
    label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1))
    has_pos = label_eq.sum(dim=1) > 1
    if not has_pos.any():
        return z.new_zeros(())
    pos_sim = sim.masked_fill(~label_eq, -1e4)
    pos_lse = pos_sim.logsumexp(dim=1)
    all_lse = sim.logsumexp(dim=1)
    return (all_lse - pos_lse)[has_pos].mean()


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
            pw = torch.as_tensor(float(self.pos_weight), dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pw)
        p = torch.sigmoid(logits)
        loss = self.alpha * targets * (1 - p).pow(self.gamma_pos) * bce \
            + (1 - self.alpha) * (1 - targets) * p.pow(self.gamma_neg) * bce
        return loss.mean()
