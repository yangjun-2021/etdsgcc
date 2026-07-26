"""
WGAN-GP Discriminator Feature Extraction
==========================================
Train WGAN-GP on PAA-compressed electricity time series.
Extract discriminator's penultimate layer as GAN-learned features
that capture theft-specific generative patterns.

Architecture:
  Generator: z(32) → 128 → 50 (PAA)
  Discriminator: 50 → 128 → 32 → 1 (Wasserstein)

Output: 32-dim adversarial features added to ensemble pool.
"""
import os, time, warnings
import numpy as np, pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.utils.data import DataLoader, TensorDataset

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ─── GAN Architecture ──────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, noise_dim=32, out_dim=50, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim=50, hidden=128, feat_dim=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.feat = nn.Sequential(
            nn.Linear(hidden, feat_dim),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x, return_feat=False):
        h = self.body(x)
        f = self.feat(h)
        out = self.head(f)
        if return_feat:
            return out, f
        return out


# ─── WGAN-GP Training ──────────────────────────────────────────────
def gradient_penalty(D, real, fake):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=real.device)
    alpha = alpha.expand_as(real)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    d_interpolated = D(interpolated)
    grad = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True,
    )[0]
    grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def train_wgan_gp(X_paa, y, noise_dim=32, hidden=128, feat_dim=32,
                  epochs=200, batch_size=128, lr=2e-4, gp_lambda=10,
                  n_critic=5, device=DEV):
    """Train WGAN-GP and extract discriminator features."""

    # Use all data for representation learning
    X_t = torch.tensor(X_paa.astype(np.float32), device=device)
    n = len(X_t)
    dl = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    G = Generator(noise_dim, X_paa.shape[1], hidden).to(device)
    D = Discriminator(X_paa.shape[1], hidden, feat_dim).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    print(f"  WGAN-GP: {X_paa.shape[1]}→{hidden}→{feat_dim}   epoch={epochs}  bs={batch_size}")
    t0 = time.time()

    for epoch in range(epochs):
        d_loss_total = 0.0
        for _ in range(n_critic):
            for (x_real,) in dl:
                b = x_real.size(0)
                z = torch.randn(b, noise_dim, device=device)
                with torch.no_grad():
                    x_fake = G(z)
                d_real = D(x_real)
                d_fake = D(x_fake)
                gp = gradient_penalty(D, x_real, x_fake)
                d_loss = d_fake.mean() - d_real.mean() + gp_lambda * gp
                d_opt.zero_grad(); d_loss.backward(); d_opt.step()
                d_loss_total += d_loss.item()

        z = torch.randn(batch_size, noise_dim, device=device)
        g_loss = -D(G(z)).mean()
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        if (epoch + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1:3d}: D_loss={d_loss_total/n_critic/len(dl):.4f}  "
                  f"G_loss={g_loss.item():.4f}  {elapsed:.0f}s")

    # Extract discriminator features for ALL samples
    D.eval()
    feat_list = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_t[i:i+batch_size]
            _, feats = D(batch, return_feat=True)
            feat_list.append(feats.cpu().numpy())
    gan_feats = np.concatenate(feat_list, axis=0)
    print(f"  GAN features extracted: {gan_feats.shape[1]} dims  |  "
          f"Total time: {time.time()-t0:.0f}s")

    return gan_feats, D, G


# ─── Integration Test ──────────────────────────────────────────────
def compute_paa(raw, n_seg=50):
    n, nd = raw.shape
    seg = nd / n_seg
    out = np.zeros((n, n_seg), dtype=np.float32)
    for i in range(n_seg):
        s = int(round(i * seg)); e = int(round((i + 1) * seg))
        out[:, i] = np.nanmean(raw[:, s:max(e, s+1)], axis=1)
    return np.nan_to_num(out, nan=0)


if __name__ == '__main__':
    print(f"Device: {DEV}")
    print("=" * 60)
    print("  WGAN-GP Discriminator Feature Extraction")
    print("=" * 60)

    t0 = time.time()

    # Load and preprocess
    print("\n[1] Loading data...")
    raw_df = pd.read_csv('data/raw_data.csv')
    date_cols = [c for c in raw_df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = raw_df[date_cols].values.astype(float)
    y = raw_df['FLAG'].values.astype(int)
    del raw_df
    print(f"    Raw: {raw.shape}  theft={y.sum()}/{len(y)}")

    # PAA compression
    print("\n[2] PAA compression (1034→50)...")
    X_paa = compute_paa(raw, n_seg=50)
    # Per-sample standardize
    mean = X_paa.mean(axis=1, keepdims=True)
    std = X_paa.std(axis=1, keepdims=True) + 1e-6
    X_norm = (X_paa - mean) / std
    X_norm = np.clip(X_norm, -5, 5)
    print(f"    PAA shape: {X_norm.shape}")

    # Train WGAN-GP
    print("\n[3] Training WGAN-GP...")
    gan_feats, D, G = train_wgan_gp(
        X_norm, y,
        noise_dim=32, hidden=128, feat_dim=32,
        epochs=200, batch_size=128, lr=2e-4, gp_lambda=10,
        n_critic=5, device=DEV,
    )

    # Quick evaluation: train tiny GBDT on gan_feats only
    print("\n[4] Quick evaluation: GBDT on GAN features (32 dims)...")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score
    import lightgbm as lgb

    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fi, (ti, vi) in enumerate(skf.split(gan_feats, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = lgb.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                scale_pos_weight=pw, random_state=SEED + fi, verbose=-1)
        m.fit(gan_feats[ti], y[ti])
        oof[vi] = m.predict_proba(gan_feats[vi])[:, 1]

    bf = max(f1_score(y, (oof > th).astype(int), zero_division=0)
             for th in np.arange(0.05, 0.95, 0.001))
    auc = roc_auc_score(y, oof)
    print(f"    GAN features only: F1={bf:.4f} AUC={auc:.4f}")

    # Save
    np.savez('output/gan_features.npz', features=gan_feats, y=y,
             f1=bf, auc=auc, feat_dim=gan_feats.shape[1])
    print(f"\n  Features saved to output/gan_features.npz")
    print(f"  Total time: {(time.time() - t0) / 60:.1f} min")
