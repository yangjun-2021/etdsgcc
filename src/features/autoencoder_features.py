"""
Deep Autoencoder anomaly features for SGCC electricity theft detection.

Ported from V71's train_v71.py (DeepAE class + train_deep_ae + extract_ae_features).

Core idea: Train a deep convolutional autoencoder on NORMAL users only.
The AE learns the "normal consumption manifold". For any user, high
reconstruction error = anomalous = potential theft.

Produces 20-dim features:
  1-4:   Reconstruction error (full, early, mid, late segments)
  5-16:  Monthly reconstruction error (12 months)
  17:    Latent vector norm
  18:    KNN distance to nearest neighbor in latent space
  19:    Anomaly score relative to population median error
  20:    Max single-day reconstruction error (spike detection)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors


class DeepAE(nn.Module):
    """U-Net inspired deep autoencoder for 1D time series."""

    def __init__(self, input_len=1034):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1))
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.1))
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.1))
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.1))

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            en = self.enc4(self.enc3(self.enc2(self.enc1(dummy))))
            self.flat_dim = en.numel()
            self.conv_len = en.shape[2]

        self.bottleneck = nn.Sequential(
            nn.Linear(self.flat_dim, 256), nn.LeakyReLU(0.1),
            nn.Linear(256, 128))

        self.dec_fc = nn.Sequential(
            nn.Linear(128, 256), nn.LeakyReLU(0.1),
            nn.Linear(256, self.flat_dim), nn.LeakyReLU(0.1))

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.1))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.1))
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1))
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(64, 1, 5, stride=2, padding=2, output_padding=1))

    def forward(self, x, return_latent=False):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        flat = e4.reshape(e4.size(0), -1)
        latent = self.bottleneck(flat)
        di = self.dec_fc(latent).reshape(e4.size(0), -1, self.conv_len)
        d1 = self.dec1(di)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        out = self.dec4(d3)
        if out.shape[2] > x.shape[2]:
            out = out[:, :, :x.shape[2]]
        elif out.shape[2] < x.shape[2]:
            pad = torch.zeros(out.size(0), out.size(1),
                              x.shape[2] - out.shape[2], device=out.device)
            out = torch.cat([out, pad], dim=2)
        if return_latent:
            return out, latent, flat
        return out


def train_deep_ae(X_normal, epochs=300, batch_size=256, val_ratio=0.1,
                   device='cuda', seed=42, verbose=True):
    """Train deep AE on normal users only.

    Args:
        X_normal: [N_normal, T] consumption data for normal users
        epochs: max training epochs
        batch_size: training batch size
        val_ratio: validation split ratio
        device: 'cuda' or 'cpu'
        seed: random seed
        verbose: print progress

    Returns:
        trained DeepAE model
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_n = X_normal.astype(np.float32)
    n_n = X_n.shape[0]

    um = X_n.mean(1, keepdims=True)
    us = X_n.std(1, keepdims=True) + 1e-6
    X_norm = np.nan_to_num((X_n - um) / us, nan=0.0)

    idx = np.random.RandomState(seed).permutation(n_n)
    n_tr = int(n_n * (1 - val_ratio))
    X_tr_n, X_val_n = X_norm[idx[:n_tr]], X_norm[idx[n_tr:]]

    model = DeepAE(input_len=X_normal.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_tr_n).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_n).unsqueeze(1).to(device)
    ld = DataLoader(TensorDataset(X_tr_t), batch_size=batch_size, shuffle=True)

    best_val = float('inf')
    best_state = None
    patience = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0
        for (bx,) in ld:
            bx = bx.to(device)
            optimizer.zero_grad()
            rec = model(bx)
            loss = criterion(rec, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vr = model(X_val_t)
            val_loss = criterion(vr, X_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if verbose and (ep + 1) % 60 == 0:
            print(f'    AE epoch {ep+1}: train={total_loss/n_tr:.6f} val={val_loss:.6f}')

        if patience >= 50:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def extract_ae_features(model, X_all, batch_size=512, dates_month=None,
                         device='cuda', seed=42):
    """Extract 20-dim AE features for all users.

    Args:
        model: trained DeepAE model
        X_all: [N, T] consumption data for ALL users
        batch_size: inference batch size
        dates_month: [T] array of month numbers (1-12) for monthly error
        device: 'cuda' or 'cpu'
        seed: random seed for KNN sampling

    Returns:
        [N, 20] feature matrix
    """
    n, T = X_all.shape
    feats = np.zeros((n, 20), dtype=np.float32)

    um = X_all.mean(1, keepdims=True)
    us = X_all.std(1, keepdims=True) + 1e-6
    X_norm = np.nan_to_num((X_all - um) / us, nan=0.0)

    X_t = torch.FloatTensor(X_norm).unsqueeze(1)

    recs = np.zeros((n, T), dtype=np.float32)
    latents = np.zeros((n, 128), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            bx = X_t[i:end].to(device)
            rec, lat, _ = model(bx, return_latent=True)
            recs[i:end] = rec.squeeze(1).cpu().numpy()
            latents[i:end] = lat.cpu().numpy()

    diff = (recs - X_norm) ** 2

    T3 = T // 3
    feats[:, 0] = diff.mean(1)
    if T3 > 0:
        feats[:, 1] = diff[:, :T3].mean(1)
        feats[:, 2] = diff[:, T3:2*T3].mean(1)
        feats[:, 3] = diff[:, 2*T3:].mean(1)

    if dates_month is not None:
        for m in range(12):
            m_mask = dates_month == (m + 1)
            if m_mask.sum() > 0:
                feats[:, 4 + m] = diff[:, m_mask].mean(1)

    feats[:, 16] = np.sqrt((latents ** 2).sum(1))

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(n, min(5000, n), replace=False)
    nbrs = NearestNeighbors(n_neighbors=2, metric='cosine')
    nbrs.fit(latents[sample_idx])
    dists, _ = nbrs.kneighbors(latents)
    feats[:, 17] = dists[:, 1]

    mean_errors = feats[:, 0]
    median_error = np.median(mean_errors)
    feats[:, 18] = mean_errors / (median_error + 1e-10)

    feats[:, 19] = diff.max(1)

    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=0.0)
    feats = np.clip(feats, -10, 10)
    return feats


def compute_autoencoder_features(X_interp, flags, date_cols,
                                  epochs=200, batch_size=256, device='cuda',
                                  seed=42, verbose=True):
    """Full pipeline: train AE on normal users, extract features for all.

    Args:
        X_interp: [N, T] interpolated consumption data (NaN-filled)
        flags: [N] binary labels (0=normal, 1=theft)
        date_cols: list of date column strings (M/D/YYYY)
        epochs: AE training epochs
        batch_size: AE training batch size
        device: 'cuda' or 'cpu'
        seed: random seed
        verbose: print progress

    Returns:
        [N, 20] AE feature matrix
    """
    normal_mask = flags == 0
    X_normal = X_interp[normal_mask]

    if verbose:
        print(f"  Training AE on {normal_mask.sum()} normal users...")

    model = train_deep_ae(X_normal, epochs=epochs, batch_size=batch_size,
                           device=device, seed=seed, verbose=verbose)

    dates_month = np.array([int(str(c).split('/')[0]) for c in date_cols])

    if verbose:
        print(f"  Extracting AE features for all {len(flags)} users...")

    ae_feats = extract_ae_features(model, X_interp, dates_month=dates_month,
                                     device=device, seed=seed)

    del model
    torch.cuda.empty_cache()

    return ae_feats


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    np.random.seed(42)
    N, T = 200, 365
    X = np.random.rand(N, T) * 50
    flags = (np.random.rand(N) < 0.1).astype(int)
    date_cols = [f'{(i%12)+1}/1/2014' for i in range(T)]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    ae_feats = compute_autoencoder_features(
        X, flags, date_cols, epochs=50, device=device, verbose=True
    )
    print(f"AE features: {ae_feats.shape}")
    print(f"Normal users mean recon error: {ae_feats[flags==0, 0].mean():.4f}")
    print(f"Theft users mean recon error:  {ae_feats[flags==1, 0].mean():.4f}")
    print("Test passed!")
