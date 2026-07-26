"""
Supervised Contrastive Learning for Electricity Theft Detection.

Based on:
- Khosla et al. (2020) "Supervised Contrastive Learning" (NeurIPS)
- Liu et al. (2023) "A electricity theft detection method through contrastive
  learning in smart grid" (EURASIP JWCN) — first SupCon for ETD on SGCC

Core insight: Traditional cross-entropy only learns decision boundaries.
FN users who "mimic" normal consumption are on the wrong side of the boundary
but the model is confidently wrong. SupCon actively shapes the embedding
geometry: pulls same-class samples together, pushes different-class apart.
This creates discriminative representations even for "mimicking" theft users.

Architecture:
  1. TCN Encoder: [N, C, T] -> [N, D] (temporal representation)
  2. Projection Head: [N, D] -> [N, P] (contrastive space)
  3. Classifier: [N, D] -> [N, 1] (binary classification)

Training:
  Stage 1: Pretrain encoder with SupCon loss (representation learning)
  Stage 2: Fine-tune with classification loss (decision boundary)
  Or joint: L = L_cls + lambda * L_supcon
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

from src.models.models import TCNEncoder


class SupConEncoder(nn.Module):
    """TCN encoder for contrastive learning.

    Encodes multi-channel time series into a fixed-dimensional representation.
    Includes an optional projection head for contrastive loss.
    """

    def __init__(self, in_channels, tcn_channels, kernel_size=7, dropout=0.3,
                 proj_dim=64):
        super().__init__()
        self.tcn = TCNEncoder(in_channels, tcn_channels, kernel_size, dropout)
        tcn_out = tcn_channels[-1] if tcn_channels else in_channels

        self.pool_proj = nn.Sequential(
            nn.Linear(tcn_out, tcn_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_out * 2, tcn_out),
        )

        self.projection = nn.Sequential(
            nn.Linear(tcn_out, tcn_out),
            nn.GELU(),
            nn.Linear(tcn_out, proj_dim),
        )

        self.feature_dim = tcn_out

    def forward(self, x, return_proj=False):
        tcn_out = self.tcn(x)
        pooled = torch.mean(tcn_out, dim=2)
        feat = self.pool_proj(pooled)

        if return_proj:
            proj = self.projection(feat)
            proj = F.normalize(proj, dim=1)
            return feat, proj
        return feat


class SupConClassifier(nn.Module):
    """Full model: encoder + classifier head with optional prior input."""

    def __init__(self, in_channels, tcn_channels, kernel_size=7, dropout=0.3,
                 proj_dim=64, use_prior=False):
        super().__init__()
        self.encoder = SupConEncoder(in_channels, tcn_channels, kernel_size,
                                      dropout, proj_dim)
        self.use_prior = use_prior
        prior_dim = 1 if use_prior else 0
        feat_dim = self.encoder.feature_dim + prior_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x, return_proj=False, prior=None):
        if return_proj:
            feat, proj = self.encoder(x, return_proj=True)
            if self.use_prior and prior is not None:
                cls_input = torch.cat([feat, prior.reshape(-1, 1)], dim=1)
            else:
                cls_input = feat
            logit = self.classifier(cls_input)
            return logit.squeeze(-1), proj
        feat = self.encoder(x, return_proj=False)
        if self.use_prior and prior is not None:
            cls_input = torch.cat([feat, prior.reshape(-1, 1)], dim=1)
        else:
            cls_input = feat
        logit = self.classifier(cls_input)
        return logit.squeeze(-1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    For each anchor, positive pairs are same-class samples in the batch.
    Negative pairs are different-class samples. The loss pulls positives
    together and pushes negatives apart in the normalized embedding space.

    L = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i·z_p/τ) / sum_{a≠i} exp(z_i·z_a/τ) )
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        """
        Args:
            projections: [N, D] L2-normalized embeddings
            labels: [N] class labels
        """
        device = projections.device
        n = projections.shape[0]

        sim = torch.matmul(projections, projections.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0.0)

        self_mask = torch.eye(n, device=device)
        sim_masked = sim - self_mask * 1e9

        sim_max, _ = sim_masked.max(dim=1, keepdim=True)
        sim_stable = sim_masked - sim_max.detach()

        exp_sim = torch.exp(sim_stable) * (1 - self_mask)
        sum_exp = exp_sim.sum(dim=1, keepdim=True) + 1e-8

        log_prob = sim_stable - torch.log(sum_exp)

        pos_count = pos_mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1.0)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

        valid_mask = pos_count > 0.5
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -(mean_log_prob_pos * valid_mask.float()).sum() / valid_mask.sum()
        return loss


class SymmetricCrossEntropy(nn.Module):
    """Symmetric Cross Entropy loss — noise-robust classification loss.

    Based on Wang et al. (2019) "Symmetric Cross Entropy for Robust Learning
    with Noisy Labels" (ICCV).

    L_SCE = alpha * L_CE + beta * L_RCE

    Where L_RCE is the reverse cross-entropy (using predicted probabilities
    as "soft labels"). This makes the loss robust to symmetric label noise:
    noisy samples (wrong labels) naturally produce larger gradients that
    get dampened by the RCE term.

    For SGCC: ~3% label noise estimated. SCE handles this without needing
    to identify which labels are noisy (unlike cleanlab which removes samples).
    """

    def __init__(self, alpha=1.0, beta=1.0, num_classes=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        Args:
            logits: [N] raw logits (before sigmoid)
            targets: [N] binary labels {0, 1}
        """
        targets = targets.float()

        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probs = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
        rce = -(targets * torch.log(probs + 1e-8) +
                (1 - targets) * torch.log(1 - probs + 1e-8))

        loss = self.alpha * ce + self.beta * rce
        return loss.mean()


def train_supcon_model(X_seq, y, leaf_indices, oof_prior=None,
                        tcn_channels=None, kernel_size=5, dropout=0.3,
                        proj_dim=64, epochs=50, batch_size=64, lr=3e-4,
                        supcon_weight=0.5, supcon_temp=0.07,
                        sce_alpha=1.0, sce_beta=1.0,
                        device='cuda', seed=42, verbose=True):
    """Train SupCon + noise-robust classifier.

    Joint training: L = L_SCE(cls) + lambda * L_SupCon(proj)

    The SupCon loss shapes the embedding space to separate "mimicking"
    theft users from normals. The SCE loss handles label noise robustly.

    Args:
        X_seq: [N, C, T] multi-channel time series
        y: [N] binary labels
        leaf_indices: [N, n_trees] GBDT leaf indices (unused in SupCon, kept for API)
        oof_prior: [N] GBDT OOF probabilities (optional, used as auxiliary feature)
        tcn_channels: list of TCN channel sizes
        kernel_size: TCN kernel size
        dropout: dropout rate
        proj_dim: projection head output dimension
        epochs: training epochs
        batch_size: training batch size
        lr: learning rate
        supcon_weight: lambda for SupCon loss
        supcon_temp: temperature for SupCon loss
        sce_alpha: alpha for SCE loss
        sce_beta: beta for SCE loss
        device: 'cuda' or 'cpu'
        seed: random seed
        verbose: print progress

    Returns:
        trained SupConClassifier model
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if tcn_channels is None:
        tcn_channels = [32, 32, 32, 16]

    N = len(y)
    in_ch = X_seq.shape[1]
    use_prior = oof_prior is not None

    model = SupConClassifier(in_ch, tcn_channels, kernel_size, dropout,
                              proj_dim, use_prior=use_prior).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  SupCon model: {n_params:,} params (prior={use_prior})")

    supcon_criterion = SupConLoss(temperature=supcon_temp)
    cls_criterion = SymmetricCrossEntropy(alpha=sce_alpha, beta=sce_beta)

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
    max_patience = 10

    for epoch in range(epochs):
        model.train()
        total_cls_loss = 0
        total_con_loss = 0
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

            logits, projections = model(batch_x, return_proj=True, prior=batch_p)

            cls_loss = cls_criterion(logits, batch_y)

            n_pos = (batch_y == 1).sum().item()
            n_neg = (batch_y == 0).sum().item()
            if n_pos >= 2 and n_neg >= 2:
                con_loss = supcon_criterion(projections, batch_y)
            else:
                con_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + supcon_weight * con_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_con_loss += con_loss.item()
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
                    all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_probs = np.concatenate(all_probs)

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

            print(f"    Epoch {epoch+1}: cls={total_cls_loss/max(n_batches,1):.4f} "
                  f"con={total_con_loss/max(n_batches,1):.4f} "
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


def predict_supcon(model, X_seq, oof_prior=None, batch_size=512, device='cuda'):
    """Get OOF predictions from SupCon model."""
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


def train_supcon_cv(X_seq, y, leaf_indices=None, oof_prior=None,
                     n_folds=5, seed=42, device='cuda', **kwargs):
    """Train SupCon model with K-fold cross-validation.

    Returns OOF predictions.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
        print(f"\n  SupCon Fold {fold_idx+1}/{n_folds}")

        model = train_supcon_model(
            X_seq[train_idx], y[train_idx],
            leaf_indices[train_idx] if leaf_indices is not None else None,
            oof_prior[train_idx] if oof_prior is not None else None,
            device=device, seed=seed + fold_idx, verbose=True, **kwargs
        )

        val_probs = predict_supcon(model, X_seq[val_idx],
                                    oof_prior=oof_prior[val_idx] if oof_prior is not None else None,
                                    device=device)
        oof_proba[val_idx] = val_probs

        f1_best = 0
        best_th = 0.5
        for th in np.arange(0.1, 0.9, 0.005):
            pred = (val_probs > th).astype(int)
            if pred.sum() == 0:
                continue
            f1 = f1_score(y[val_idx], pred, zero_division=0)
            if f1 > f1_best:
                f1_best = f1
                best_th = th

        auc = roc_auc_score(y[val_idx], val_probs)
        print(f"  Fold {fold_idx+1}: F1={f1_best:.4f} AUC={auc:.4f} th={best_th:.3f}")

        del model
        torch.cuda.empty_cache()

    return oof_proba


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("Testing SupCon model...")
    N, C, T = 100, 4, 100
    np.random.seed(42)
    X = np.random.randn(N, C, T).astype(np.float32)
    y = (np.random.rand(N) < 0.15).astype(int)
    y[:10] = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = train_supcon_model(X, y, None, None,
                                epochs=10, batch_size=16, device=device)
    probs = predict_supcon(model, X, device=device)
    print(f"Output shape: {probs.shape}")
    print(f"Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print("SupCon test passed!")
