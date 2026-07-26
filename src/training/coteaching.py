"""
Co-teaching for noise-robust electricity theft detection.

Based on Han et al. (2018) "Co-teaching: Robust Training of Deep Neural
Networks with Extremely Noisy Labels" (NeurIPS).

Core idea: Train two networks simultaneously. In each batch, each network
selects the small-loss samples (likely correctly labeled) and teaches them
to the other network. Large-loss samples (likely noisy labels) are naturally
filtered out.

For SGCC: ~3% label noise estimated. Co-teaching handles this without
needing to identify which labels are noisy. The forget rate schedule
gradually increases from 0 to the estimated noise rate.

Combined with SupCon + SCE for maximum robustness:
  - SupCon: shapes embedding space (separates mimicking theft)
  - SCE: noise-robust loss (handles noisy labels)
  - Co-teaching: filters noisy samples during training
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

from src.models.supcon_model import SupConClassifier, SupConLoss, SymmetricCrossEntropy


def forget_rate_schedule(epoch, max_epoch, initial_rate=0.0, final_rate=0.3,
                          warmup_epochs=10):
    """Linear forget rate schedule with warmup.

    During warmup, forget_rate = 0 (use all samples).
    After warmup, linearly increase to final_rate (estimated noise rate).
    """
    if epoch < warmup_epochs:
        return initial_rate
    progress = (epoch - warmup_epochs) / max(max_epoch - warmup_epochs, 1)
    return initial_rate + (final_rate - initial_rate) * min(progress, 1.0)


def train_coteaching(X_seq, y, leaf_indices=None, oof_prior=None,
                      tcn_channels=None, kernel_size=5, dropout=0.3,
                      proj_dim=64, epochs=50, batch_size=64, lr=3e-4,
                      supcon_weight=0.3, supcon_temp=0.07,
                      sce_alpha=1.0, sce_beta=0.5,
                      forget_rate=0.15, warmup_epochs=10,
                      device='cuda', seed=42, verbose=True):
    """Train with co-teaching: two networks filter noisy labels for each other.

    Args:
        X_seq: [N, C, T] time series
        y: [N] labels
        forget_rate: estimated label noise rate (max forget rate)
        warmup_epochs: epochs before forgetting starts
        Other args: same as train_supcon_model

    Returns:
        (model1, model2): both trained networks
    """
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

    model1 = SupConClassifier(in_ch, tcn_channels, kernel_size, dropout,
                               proj_dim, use_prior=use_prior).to(device)
    model2 = SupConClassifier(in_ch, tcn_channels, kernel_size, dropout,
                               proj_dim, use_prior=use_prior).to(device)

    supcon_criterion = SupConLoss(temperature=supcon_temp)
    cls_criterion = SymmetricCrossEntropy(alpha=sce_alpha, beta=sce_beta)

    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=lr, weight_decay=1e-4)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=lr, weight_decay=1e-4)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=epochs, eta_min=1e-6)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=epochs, eta_min=1e-6)

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
    best_state1 = None
    best_state2 = None
    patience = 0

    for epoch in range(epochs):
        model1.train()
        model2.train()

        cur_forget = forget_rate_schedule(epoch, epochs, 0.0, forget_rate,
                                           warmup_epochs)

        total_loss = 0
        n_batches = 0
        n_kept = 0
        n_total = 0

        for batch_data in loader:
            if use_prior:
                batch_x, batch_y, batch_p = batch_data
                batch_p = batch_p.to(device)
            else:
                batch_x, batch_y = batch_data
                batch_p = None
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.shape[0]

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            logits1, proj1 = model1(batch_x, return_proj=True, prior=batch_p)
            logits2, proj2 = model2(batch_x, return_proj=True, prior=batch_p)

            loss1 = cls_criterion(logits1, batch_y)
            loss2 = cls_criterion(logits2, batch_y)

            per_sample_loss1 = cls_criterion(logits1, batch_y)
            per_sample_loss2 = cls_criterion(logits2, batch_y)

            with torch.no_grad():
                loss1_indiv = F_loss_indiv(cls_criterion, logits1, batch_y)
                loss2_indiv = F_loss_indiv(cls_criterion, logits2, batch_y)

                n_remember = max(int(batch_size * (1 - cur_forget)), 1)

                _, idx1 = torch.topk(loss2_indiv, n_remember, largest=False)
                _, idx2 = torch.topk(loss1_indiv, n_remember, largest=False)

            clean_logits1 = logits1[idx2]
            clean_y1 = batch_y[idx2]
            clean_proj1 = proj1[idx2]

            clean_logits2 = logits2[idx1]
            clean_y2 = batch_y[idx1]
            clean_proj2 = proj2[idx1]

            cls_loss1 = cls_criterion(clean_logits1, clean_y1)
            cls_loss2 = cls_criterion(clean_logits2, clean_y2)

            n_pos1 = (clean_y1 == 1).sum().item()
            n_neg1 = (clean_y1 == 0).sum().item()
            if n_pos1 >= 2 and n_neg1 >= 2:
                con_loss1 = supcon_criterion(clean_proj1, clean_y1)
            else:
                con_loss1 = torch.tensor(0.0, device=device)

            n_pos2 = (clean_y2 == 1).sum().item()
            n_neg2 = (clean_y2 == 0).sum().item()
            if n_pos2 >= 2 and n_neg2 >= 2:
                con_loss2 = supcon_criterion(clean_proj2, clean_y2)
            else:
                con_loss2 = torch.tensor(0.0, device=device)

            loss1_final = cls_loss1 + supcon_weight * con_loss1
            loss2_final = cls_loss2 + supcon_weight * con_loss2

            loss1_final.backward()
            loss2_final.backward()

            torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)

            optimizer1.step()
            optimizer2.step()

            total_loss += (loss1_final.item() + loss2_final.item()) / 2
            n_batches += 1
            n_kept += n_remember * 2
            n_total += batch_size * 2

        scheduler1.step()
        scheduler2.step()

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            model1.eval()
            model2.eval()
            with torch.no_grad():
                probs1 = []
                probs2 = []
                for start in range(0, N, 512):
                    end = min(start + 512, N)
                    xb = torch.FloatTensor(X_seq[start:end]).to(device)
                    pb = torch.FloatTensor(oof_prior[start:end]).to(device) if use_prior else None
                    probs1.append(torch.sigmoid(model1(xb, prior=pb)).cpu().numpy())
                    probs2.append(torch.sigmoid(model2(xb, prior=pb)).cpu().numpy())
                probs1 = np.concatenate(probs1)
                probs2 = np.concatenate(probs2)
                avg_probs = (probs1 + probs2) / 2

            auc = roc_auc_score(y, avg_probs)
            best_th = 0.5
            best_f1_val = 0
            for th in np.arange(0.1, 0.9, 0.01):
                pred = (avg_probs > th).astype(int)
                if pred.sum() == 0:
                    continue
                f1 = f1_score(y, pred, zero_division=0)
                if f1 > best_f1_val:
                    best_f1_val = f1
                    best_th = th

            keep_rate = n_kept / max(n_total, 1)
            print(f"    Epoch {epoch+1}: loss={total_loss/max(n_batches,1):.4f} "
                  f"forget={cur_forget:.2f} keep={keep_rate:.2f} "
                  f"F1={best_f1_val:.4f} AUC={auc:.4f}")

            if best_f1_val > best_f1:
                best_f1 = best_f1_val
                best_state1 = {k: v.cpu().clone() for k, v in model1.state_dict().items()}
                best_state2 = {k: v.cpu().clone() for k, v in model2.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if patience >= 10:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1}")
                break

    if best_state1 is not None:
        model1.load_state_dict(best_state1)
        model2.load_state_dict(best_state2)

    return model1, model2


def F_loss_indiv(criterion, logits, targets):
    """Compute per-sample loss (not averaged)."""
    probs = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    targets = targets.float()
    ce = -(targets * torch.log(probs + 1e-8) +
           (1 - targets) * torch.log(1 - probs + 1e-8))
    return ce


def predict_coteaching(model1, model2, X_seq, oof_prior=None, batch_size=512, device='cuda'):
    """Average predictions from both co-teaching networks."""
    model1.eval()
    model2.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            end = min(start + batch_size, len(X_seq))
            xb = torch.FloatTensor(X_seq[start:end]).to(device)
            pb = torch.FloatTensor(oof_prior[start:end]).to(device) if oof_prior is not None else None
            p1 = torch.sigmoid(model1(xb, prior=pb)).cpu().numpy()
            p2 = torch.sigmoid(model2(xb, prior=pb)).cpu().numpy()
            all_probs.append((p1 + p2) / 2)
    return np.concatenate(all_probs)


def train_coteaching_cv(X_seq, y, leaf_indices=None, oof_prior=None,
                         n_folds=5, seed=42, device='cuda', **kwargs):
    """Train co-teaching with K-fold CV. Returns OOF predictions."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
        print(f"\n  Co-teaching Fold {fold_idx+1}/{n_folds}")

        model1, model2 = train_coteaching(
            X_seq[train_idx], y[train_idx],
            leaf_indices[train_idx] if leaf_indices is not None else None,
            oof_prior[train_idx] if oof_prior is not None else None,
            device=device, seed=seed + fold_idx, verbose=True, **kwargs
        )

        val_probs = predict_coteaching(model1, model2, X_seq[val_idx],
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

        del model1, model2
        torch.cuda.empty_cache()

    return oof_proba


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("Testing Co-teaching...")
    N, C, T = 100, 4, 100
    np.random.seed(42)
    X = np.random.randn(N, C, T).astype(np.float32)
    y = (np.random.rand(N) < 0.15).astype(int)
    y[:10] = 1
    y[50] = 0  # Inject label noise

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m1, m2 = train_coteaching(X, y, epochs=10, batch_size=16, device=device)
    probs = predict_coteaching(m1, m2, X, device=device)
    print(f"Output: {probs.shape}, range=[{probs.min():.4f}, {probs.max():.4f}]")
    print("Co-teaching test passed!")
