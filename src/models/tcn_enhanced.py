"""
Enhanced TCN: Multi-resolution + Self-Attention + SWA + ASL
=============================================================
Key improvements over baseline TCN (F1=0.843):
  1. Multi-resolution PAA input (100+50+25 segments) instead of raw 1034
  2. Multi-head self-attention after TCN encoder
  3. Squeeze-and-Excitation channel attention
  4. Asymmetric Loss (ASL) for better imbalance handling
  5. Stochastic Weight Averaging (SWA)
  6. Knowledge distillation to V225 soft labels

Target: TCN F1 from 0.843 → 0.85-0.86
"""
import os, time, glob, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from src.utils.utils import seed_everything, best_f1_score

SEED = 42
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEV}')
seed_everything(SEED)


# ═══════════════════════════════════════════════════════════════════
# ENHANCED MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, t = x.shape
        y = x.mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding))
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding))
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.pad = padding
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        out = self.conv1(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        out = F.gelu(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        out = F.gelu(self.bn2(out))
        out = self.se(out)
        out = self.dropout(out)
        return F.gelu(out + res)


class EnhancedTCNEncoder(nn.Module):
    """TCN with SE blocks, deeper channels."""
    def __init__(self, in_channels, channels, kernel_size=5, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = in_channels
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over time dimension."""
    def __init__(self, dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, t = x.shape
        x_t = x.transpose(1, 2)
        qkv = self.qkv(x_t).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
                   for z in qkv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, t, c)
        out = self.out(out)
        return out.transpose(1, 2)


class EnhancedTCNClassifier(nn.Module):
    """Enhanced TCN: V108 input → TCN → adaptive pool → attention → classifier."""
    def __init__(self, in_channels, tcn_channels, kernel_size=7, dropout=0.3,
                 attn_heads=4, pool_size=128, use_prior=True):
        super().__init__()
        self.tcn = EnhancedTCNEncoder(in_channels, tcn_channels, kernel_size, dropout)
        self.pool_size = pool_size

        tcn_out = tcn_channels[-1]
        self.attn = SelfAttention1D(tcn_out, n_heads=attn_heads, dropout=dropout)

        prior_dim = 1 if use_prior else 0
        self.use_prior = use_prior

        self.classifier = nn.Sequential(
            nn.Linear(tcn_out + prior_dim, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(48, 1),
        )

    def forward(self, x, prior=None):
        h = self.tcn(x)
        if h.shape[2] > self.pool_size:
            h = F.adaptive_avg_pool1d(h, self.pool_size)
        h = self.attn(h)
        pooled = torch.mean(h, dim=2)

        parts = [pooled]
        if self.use_prior and prior is not None:
            parts.append(prior.reshape(-1, 1))

        combined = torch.cat(parts, dim=1)
        return self.classifier(combined).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# ASYMMETRIC LOSS
# ═══════════════════════════════════════════════════════════════════
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss: gamma_neg > gamma_pos to reduce easy negative impact."""
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        xs_pos = probs
        xs_neg = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = -targets * torch.log(xs_pos) * torch.pow(1.0 - xs_pos, self.gamma_pos)
        loss_neg = -(1.0 - targets) * torch.log(xs_neg) * torch.pow(xs_neg, self.gamma_neg)

        return loss_pos.mean() + loss_neg.mean()


# ═══════════════════════════════════════════════════════════════════
# MULTI-RESOLUTION PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
def build_multi_resolution_input(raw, resolutions=[25, 50, 100]):
    """
    Build multi-resolution PAA channels.
    Returns: [N, n_channels, max_len] where n_channels = len(resolutions)
    Each channel is a different PAA resolution, padded to max_len.
    """
    n, nd = raw.shape
    channels = []
    max_len = 0

    for res in resolutions:
        seg = nd / res
        paa = np.zeros((n, res), dtype=np.float32)
        for i in range(res):
            s = int(round(i * seg))
            e = int(round((i + 1) * seg))
            if e > s:
                paa[:, i] = np.nanmean(raw[:, s:e], axis=1)
            elif s < nd:
                paa[:, i] = raw[:, s]
        paa = np.nan_to_num(paa, nan=0)
        # Per-sample standardize
        mean = paa.mean(axis=1, keepdims=True)
        std = paa.std(axis=1, keepdims=True) + 1e-6
        paa = (paa - mean) / std
        paa = np.clip(paa, -5, 5)
        channels.append(paa)
        max_len = max(max_len, res)

    # Pad shorter sequences to max_len
    result = np.zeros((n, len(resolutions), max_len), dtype=np.float32)
    for i, ch in enumerate(channels):
        result[:, i, :ch.shape[1]] = ch

    return result


def build_aux_channels(raw, resolutions=[25, 50, 100]):
    """Build auxiliary channels: missing_mask and zero_mask at multi-resolutions."""
    n, nd = raw.shape
    channels = []
    max_len = 0

    for res in resolutions:
        seg = nd / res
        miss = np.zeros((n, res), dtype=np.float32)
        zero = np.zeros((n, res), dtype=np.float32)
        for i in range(res):
            s = int(round(i * seg))
            e = int(round((i + 1) * seg))
            if e > s:
                seg_raw = raw[:, s:e]
                miss[:, i] = np.isnan(seg_raw).mean(axis=1)
                zero[:, i] = ((seg_raw == 0) | np.isnan(seg_raw)).mean(axis=1)
        channels.extend([miss, zero])
        max_len = max(max_len, res)

    result = np.zeros((n, len(channels), max_len), dtype=np.float32)
    for i, ch in enumerate(channels):
        result[:, i, :ch.shape[1]] = ch

    return result


# ═══════════════════════════════════════════════════════════════════
# TRAINING WITH SWA
# ═══════════════════════════════════════════════════════════════════
class SWA(nn.Module):
    """Stochastic Weight Averaging."""
    def __init__(self, model, start_epoch=20):
        super().__init__()
        self.model = model
        self.start_epoch = start_epoch
        self.swa_model = None
        self.n_averaged = 0

    def update(self, model, epoch):
        if epoch < self.start_epoch:
            return
        if self.swa_model is None:
            self.swa_model = {k: v.clone().detach() for k, v in model.state_dict().items()}
            self.n_averaged = 1
        else:
            for k in self.swa_model:
                self.swa_model[k] = (self.swa_model[k] * self.n_averaged +
                                     model.state_dict()[k].clone().detach()) / (self.n_averaged + 1)
            self.n_averaged += 1

    def get_swa_model(self):
        if self.swa_model is None:
            return self.model
        model_copy = type(self.model)(**self._get_init_args())
        model_copy.load_state_dict(self.swa_model)
        return model_copy


def train_enhanced_tcn(X_seq, y, oof_prior=None, epochs=60, lr=8e-4, batch_size=64,
                        kd_weight=0.3, swa_start=25, device='cuda', seed=42):
    """Train Enhanced TCN with SWA + ASL + optional KD."""
    has_prior = oof_prior is not None
    n = len(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(n)
    fold_results = []

    print(f'\n{"=" * 60}')
    print(f'  Enhanced TCN: {X_seq.shape[1]}ch x {X_seq.shape[2]}t')
    print(f'  TCN: [48,48,32,32,16]  kernel=7  attn=4heads  SE')
    print(f'  ASL(gamma_pos=1,gamma_neg=4)  KD_alpha={kd_weight}  SWA@{swa_start}')
    prior_str = 'YES' if has_prior else 'NO'
    print(f'  Prior: {prior_str}  epochs={epochs}  lr={lr}  bs={batch_size}')
    print(f'{"=" * 60}')

    for fi, (ti, vi) in enumerate(skf.split(X_seq, y)):
        tf = time.time()
        torch.cuda.empty_cache()
        torch.manual_seed(seed + fi)

        model = EnhancedTCNClassifier(
            in_channels=X_seq.shape[1],
            tcn_channels=[48, 48, 32, 32, 16],
            kernel_size=7, dropout=0.3, attn_heads=4,
            pool_size=128,
            use_prior=has_prior,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f'\n  Fold {fi+1} ({n_params:,} params)...')

        crit_ce = AsymmetricLoss(gamma_pos=1.0, gamma_neg=4.0)
        crit_kd = nn.MSELoss()

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        ds = TensorDataset(
            torch.FloatTensor(X_seq[ti]),
            torch.FloatTensor(y[ti]),
            torch.FloatTensor(oof_prior[ti]) if has_prior else torch.FloatTensor(y[ti]),
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        swa_state = {}
        n_swa = 0
        best_val_f1, best_state, patience = 0, None, 0

        for ep in range(epochs):
            model.train()
            for batch_data in dl:
                if has_prior:
                    bx, by, bp = batch_data
                    bx, by, bp = bx.to(device), by.to(device), bp.to(device)
                else:
                    bx, by, _ = batch_data
                    bx, by = bx.to(device), by.to(device)
                    bp = None
                opt.zero_grad()
                logits = model(bx, bp)
                loss_ce = crit_ce(logits, by)
                if has_prior and kd_weight > 0:
                    probs = torch.sigmoid(logits)
                    loss_kd = crit_kd(probs, bp)
                    loss = loss_ce + kd_weight * loss_kd
                else:
                    loss = loss_ce
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

            # SWA update
            if ep >= swa_start:
                if n_swa == 0:
                    swa_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
                    n_swa = 1
                else:
                    for k in swa_state:
                        swa_state[k] = (swa_state[k] * n_swa + model.state_dict()[k].clone().detach()) / (n_swa + 1)
                    n_swa += 1

            model.eval()
            with torch.no_grad():
                xv = torch.FloatTensor(X_seq[vi]).to(device)
                if has_prior:
                    pv = torch.FloatTensor(oof_prior[vi]).to(device)
                else:
                    pv = None
                val_probs = torch.sigmoid(model(xv, pv)).cpu().numpy()

            val_f1 = max(f1_score(y[vi], (val_probs > th).astype(int), zero_division=0)
                         for th in np.arange(0.2, 0.8, 0.01))

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1

            if (ep + 1) % 15 == 0:
                auc_fold = roc_auc_score(y[vi], val_probs)
                print(f'    E{ep+1:2d}: F1={val_f1:.4f} AUC={auc_fold:.4f}  '
                      f'patience={patience}/{10}  SWA={n_swa}')

            if patience >= 10:
                break

        # Use SWA model for final prediction
        if n_swa > 0:
            model.load_state_dict(swa_state)

        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(X_seq[vi]).to(device)
            if has_prior:
                pv = torch.FloatTensor(oof_prior[vi]).to(device)
            else:
                pv = None
            oof_fold = torch.sigmoid(model(xv, pv)).cpu().numpy()

        oof[vi] = oof_fold
        f1_fold, th_fold, rec_fold, prec_fold = best_f1_score(y[vi], oof_fold)
        auc_fold = roc_auc_score(y[vi], oof_fold)
        fold_results.append((f1_fold, auc_fold))
        elapsed = time.time() - tf
        print(f'  Fold {fi+1}: F1={f1_fold:.4f} AUC={auc_fold:.4f}  '
              f'SWA={n_swa}  [{elapsed:.0f}s]')
        del model
        torch.cuda.empty_cache()

    f1_all, th_all, rec_all, prec_all = best_f1_score(y, oof)
    auc_all = roc_auc_score(y, oof)
    tp = ((oof > th_all) & (y == 1)).sum()
    fp = ((oof > th_all) & (y == 0)).sum()
    fn = ((oof <= th_all) & (y == 1)).sum()

    fold_f1s = [f for f, a in fold_results]
    print(f'\n  {"=" * 60}')
    print(f'  ENHANCED TCN 5-FOLD RESULTS')
    print(f'  {"=" * 60}')
    print(f'  AUC: {auc_all:.4f}')
    print(f'  F1:  {f1_all:.4f} (th={th_all:.3f})')
    print(f'  Rec: {rec_all:.4f}  Prec: {prec_all:.4f}')
    print(f'  TP={tp} FP={fp} FN={fn}')
    print(f'  Folds: {fold_f1s}')
    print(f'  vs baseline TCN (0.8433): +{f1_all - 0.8433:+.4f}')
    print(f'  vs baseline AUC (0.9783): +{auc_all - 0.9783:+.4f}')

    return oof, f1_all, auc_all


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    OUT = 'output'

    # Load external OOFs from bundled files (self-contained, no external path)
    bundled = np.load(os.path.join(OUT, 'bundled_oofs.npz'), allow_pickle=True)
    v71_oofs = np.column_stack([bundled[f'V71_{k}'] for k in ['lgb', 'xgb', 'cat', 'tcn', 'innov']])
    ext_oofs = np.column_stack([
        bundled['V213'], bundled['V216'], bundled['V219'], bundled['V225'],
    ])

    print('Loading raw data...')
    df = pd.read_csv('data/raw_data.csv')
    dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
    raw = df[dc].values.astype(float)
    y = df['FLAG'].values.astype(int)
    n = len(y)
    del df

    # V108 preprocessing (keep full 1034 timesteps)
    print('\nV108 preprocessing (1034 timesteps)...')
    nmk = np.isnan(raw)
    Xf = np.nan_to_num(raw, nan=0.0)
    Xl = np.log1p(np.maximum(Xf, 0))
    sc = StandardScaler()
    Xs = np.clip(sc.fit_transform(Xl).astype(np.float32), -5, 5)
    X_seq = np.stack([
        Xs,
        nmk.astype(np.float32),
        (Xf == 0).astype(np.float32),
    ], axis=1)

    # GBDT prior (same as before)
    print('\nBuilding GBDT prior...')
    base = np.load(os.path.join(OUT, 'sgcc_preprocessed.npz'))
    stat = np.nan_to_num(base['stat_features'], nan=0, posinf=0, neginf=0)
    mr = base['impute_mask'].mean(axis=1).reshape(-1, 1)

    X_gbdt = np.column_stack([stat, mr, v71_oofs, ext_oofs]).astype(np.float32)
    X_gbdt = np.nan_to_num(X_gbdt, nan=0, posinf=0, neginf=0)

    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_prior = np.zeros(n)
    for fi, (ti, vi) in enumerate(skf.split(X_gbdt, y)):
        pw = (y[ti] == 0).sum() / max((y[ti] == 1).sum(), 1)
        m = lgb.LGBMClassifier(n_estimators=1000, max_depth=7, learning_rate=0.05,
                                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=pw, random_state=SEED, verbose=-1)
        m.fit(X_gbdt[ti], y[ti], eval_set=[(X_gbdt[vi], y[vi])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_prior[vi] = m.predict_proba(X_gbdt[vi])[:, 1]

    # Train Enhanced TCN WITHOUT prior (maximize diversity)
    oof_tcn, f1_tcn, auc_tcn = train_enhanced_tcn(
        X_seq, y, oof_prior=None,
        epochs=80, lr=3e-4, batch_size=32, kd_weight=0.0,
        swa_start=35, device=DEV, seed=SEED,
    )

    # Save
    np.savez(os.path.join(OUT, 'tcn_enhanced.npz'),
             oof_tcn=oof_tcn, y=y, f1=f1_tcn, auc=auc_tcn)
    print(f'\nSaved to output/tcn_enhanced.npz')
    print(f'Total time: {(time.time() - t0) / 60:.1f} min')
