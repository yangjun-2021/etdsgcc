"""V3 voter: fast single TCN (SupCon encoder) with AMP, ORIGINAL labels, NO prior.

Replacement for the infeasibly slow fp32 AMST voter on the laptop RTX 5060:
single TCN, batch 256, mixed precision, 20 epochs -> ~10 min/fold.

Usage:
    conda run -n ml python experiments/v3_tcn_fast_orig.py

Requires: GPU.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from config import OUTPUT_DIR, SEED, N_FOLDS
from src.utils.utils import seed_everything
from src.models.supcon_model import SupConClassifier

EPOCHS = 20
BATCH = 256
LR = 1e-3
WD = 1e-4


def train_fold(X_tr, y_tr, X_va, y_va, device, seed):
    seed_everything(seed)
    model = SupConClassifier(X_tr.shape[1], [64, 64, 64, 32], kernel_size=5,
                             dropout=0.3, proj_dim=64, use_prior=False).to(device)
    pw = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                      dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.astype(np.float32)))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)

    X_va_t = torch.FloatTensor(X_va)

    def predict():
        model.eval()
        probs = []
        with torch.no_grad():
            for s in range(0, len(X_va_t), 1024):
                xb = X_va_t[s:s + 1024].to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    probs.append(torch.sigmoid(model(xb).squeeze(-1)).float().cpu().numpy())
        return np.concatenate(probs)

    best_auc, best_state = 0.0, None
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                loss = criterion(model(xb).squeeze(-1), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(xb)
        sched.step()

        p = predict()
        auc = roc_auc_score(y_va, p)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'    ep{ep+1}: loss={tot/len(ds):.4f} val_AUC={auc:.4f} (best={best_auc:.4f})',
              flush=True)

    model.load_state_dict(best_state)
    return predict()


def main():
    seed_everything(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}, torch={torch.__version__}')

    pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
    X = pre['X_seq'].astype(np.float32)
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y = cl['y_orig'].astype(int)
    print(f'X={X.shape}, pos={y.sum()} ({y.mean()*100:.2f}%)')

    out_prefix = 'v3voter_tcn_fast'
    oof = np.zeros(len(y), dtype=np.float32)
    done = []
    for fi in range(N_FOLDS):
        fp = os.path.join(OUTPUT_DIR, f'{out_prefix}_fold{fi}.npz')
        if os.path.exists(fp):
            d = np.load(fp)
            oof[d['vi']] = d['oof']
            done.append(fi)
            print(f'Resumed fold {fi+1}')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fi, (ti, vi) in enumerate(skf.split(X, y)):
        if fi in done:
            continue
        print(f'\n=== Fold {fi+1}/{N_FOLDS} ===', flush=True)
        oof[vi] = train_fold(X[ti], y[ti], X[vi], y[vi], device, SEED + fi)

        best_f1, best_th = 0, 0.5
        for th in np.arange(0.05, 0.95, 0.005):
            f = f1_score(y[vi], (oof[vi] > th).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_th = f, th
        pred = (oof[vi] > best_th).astype(int)
        print(f'  Fold {fi+1}: F1={best_f1:.4f} Rec={recall_score(y[vi], pred):.4f} '
              f'Prec={precision_score(y[vi], pred, zero_division=0):.4f} '
              f'AUC={roc_auc_score(y[vi], oof[vi]):.4f} th={best_th:.3f} '
              f'({(time.time()-t0)/60:.1f}min)', flush=True)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f'{out_prefix}_fold{fi}.npz'),
                            oof=oof[vi], vi=vi, ti=ti)
        torch.cuda.empty_cache()

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y, (oof > th).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, th
    pred = (oof > best_th).astype(int)
    print(f'\n=== V3-voter fast TCN (original, no prior): F1={best_f1:.4f}, '
          f'Rec={recall_score(y, pred):.4f}, Prec={precision_score(y, pred, zero_division=0):.4f}, '
          f'AUC={roc_auc_score(y, oof):.4f}, th={best_th:.3f}')

    np.savez_compressed(os.path.join(OUTPUT_DIR, f'{out_prefix}_oof.npz'),
                        oof_v3voter_tcn_fast=oof, y_orig=y)
    print(f'Saved {out_prefix}_oof.npz, total {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
