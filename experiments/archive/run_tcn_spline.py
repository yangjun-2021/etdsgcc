"""TCN with Spline + Newton + tanh preprocessing + Knowledge Distillation.

Key improvements over run_tcn_kd.py:
  1. Cubic spline interpolation → smooth continuous time series
  2. Newton polynomial residual → theft deviation signal (4th channel)
  3. tanh robust normalization → no gradient vanishing
  4. 4-channel input: [tanh_norm, missing_mask, newton_residual, zero_mask]
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np, time, glob
import pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from utils import seed_everything, best_f1_score
from models import TCNWithLeafEmbedding, RecallOrientedFocalLoss
from tc_preprocess import build_tcn_input

seed_everything(42)
t0 = time.time()

# Load teacher OOF (V225 for KD)
OD = r'D:\Project\ThiefElectricity\output'
v225_data = np.load(sorted(glob.glob(f'{OD}/v225_results_*.npz'), reverse=True)[0],
                     allow_pickle=True)
teacher_oof = v225_data['oof_final']
print(f'Teacher: V225, AUC={np.float64(0)}')  # placeholder

# Load raw data
print('Loading raw data...')
df = pd.read_csv('data/raw_data.csv')
dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
X_raw = df[dc].values.astype(float)
y = df['FLAG'].values.astype(np.int32)
del df

# NEW: Spline + Newton + tanh preprocessing
print('\nBuilding TCN input (spline + Newton + tanh)...')
X_seq, impute_mask = build_tcn_input(X_raw)

# Load GBDT leaf indices
print('\nLoading GBDT features...')
g = np.load('output/gbdt_stage.npz')
leaf_all = g['leaf_all']
del g

DEV = 'cuda'
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_offs = []

print(f'\nTraining TCN with spline+Newton+tanh input ({X_seq.shape[1]} channels)...')

for fi, (ti, vi) in enumerate(skf.split(X_seq, y)):
    tf = time.time()
    torch.cuda.empty_cache()
    print(f'  Fold {fi+1}...', end=' ', flush=True)

    model = TCNWithLeafEmbedding(
        in_channels=X_seq.shape[1],  # 4 channels
        tcn_channels=[32, 32, 32, 16],
        kernel_size=5, dropout=0.3, n_trees=200, num_leaves=31,
        leaf_embed_dim=4, leaf_output_dim=32, use_prior=True,
    ).to(DEV)

    crit_ce = RecallOrientedFocalLoss(alpha=0.75, gamma=2.0, recall_weight=3.0)
    crit_kd = torch.nn.MSELoss()
    kd_weight = 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

    ds = TensorDataset(
        torch.FloatTensor(X_seq[ti]),
        torch.LongTensor(leaf_all[ti]),
        torch.FloatTensor(y[ti]),
        torch.FloatTensor(teacher_oof[ti]),
    )
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    bf_val = 0
    best_state = None
    patience = 0

    for ep in range(50):
        model.train()
        for bx, bl, by, bp in dl:
            bx, bl, by, bp = bx.to(DEV), bl.to(DEV), by.to(DEV), bp.to(DEV)
            opt.zero_grad()
            logits = model(bx, bl, bp)
            loss_ce = crit_ce(logits, by)
            probs = torch.sigmoid(logits)
            loss_kd = crit_kd(probs, bp)
            loss = loss_ce + kd_weight * loss_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(X_seq[vi]).to(DEV)
            lv = torch.LongTensor(leaf_all[vi]).to(DEV)
            pv = torch.FloatTensor(teacher_oof[vi]).to(DEV)
            probs_val = torch.sigmoid(model(xv, lv, pv)).cpu().numpy()

        v = max((f1_score(y[vi], (probs_val > t).astype(int), zero_division=0)
                 for t in np.arange(0.2, 0.8, 0.01)), default=0)
        if v > bf_val:
            bf_val = v
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        if patience >= 7:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xv = torch.FloatTensor(X_seq[vi]).to(DEV)
        lv = torch.LongTensor(leaf_all[vi]).to(DEV)
        pv = torch.FloatTensor(teacher_oof[vi]).to(DEV)
        oof_fold = torch.sigmoid(model(xv, lv, pv)).cpu().numpy()
    fold_offs.append((vi, oof_fold))
    auc_fold = roc_auc_score(y[vi], oof_fold)
    print(f'F1={bf_val:.4f} AUC={auc_fold:.4f} ({time.time()-tf:.0f}s)')

    del model, opt, sch, ds, dl
    torch.cuda.empty_cache()

oof_spline = np.zeros(len(y))
for vi, oof_fold in fold_offs:
    oof_spline[vi] = oof_fold

auc_all = roc_auc_score(y, oof_spline)
f1_all, th_all, rec_all, prec_all = best_f1_score(y, oof_spline)

print(f'\n{"="*60}')
print('  TCN Spline+Newton+tanh RESULTS')
print('=' * 60)
print(f'  AUC: {auc_all:.4f}')
print(f'  F1:  {f1_all:.4f} (th={th_all:.3f})')
print(f'  Rec: {rec_all:.4f}')
print(f'  Prec:{prec_all:.4f}')
tp = ((oof_spline > th_all) & (y == 1)).sum()
fp = ((oof_spline > th_all) & (y == 0)).sum()
fn = ((oof_spline <= th_all) & (y == 1)).sum()
print(f'  TP={tp} FP={fp} FN={fn}')
print(f'')
print(f'  vs V108 TCN+KD:    AUC=0.9783 F1=0.8433')
print(f'  Delta:              AUC={auc_all-0.9783:+.4f} F1={f1_all-0.8433:+.4f}')
print(f'  Time: {(time.time()-t0)/60:.1f} min')

np.savez('output/tcn_spline_results.npz', oof_spline=oof_spline, y=y)
