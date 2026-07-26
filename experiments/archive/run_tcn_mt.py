"""Multi-teacher KD: train TCN with v213+v219+v225 as teachers."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np, time, glob
import pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from utils import seed_everything, best_f1_score
from models import TCNWithLeafEmbedding, RecallOrientedFocalLoss

seed_everything(42)
t0 = time.time()

# Load multi-teacher OOFs
OD = r'D:\Project\ThiefElectricity\output'
def load(prefix, key):
    return np.load(sorted(glob.glob(f'{OD}/{prefix}*.npz'), reverse=True)[0],
                    allow_pickle=True)[key]

v213_oof = load('v213_results_', 'oof_v213')
v219_oof = load('v219_results_', 'oof_final')
v225_oof = load('v225_results_', 'oof_final')
teacher_oof = (v213_oof + v219_oof + v225_oof) / 3.0  # multi-teacher average

# Raw data + V108 preprocessing
df = pd.read_csv('data/raw_data.csv')
dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
X_raw = df[dc].values.astype(np.float32)
y = df['FLAG'].values.astype(np.int32)
del df
nmk = np.isnan(X_raw); Xf = np.nan_to_num(X_raw, nan=0.0)
Xl = np.log1p(np.maximum(Xf, 0)); sc = StandardScaler()
Xs = np.clip(sc.fit_transform(Xl).astype(np.float32), -5, 5)
X_seq = np.stack([Xs, nmk.astype(np.float32), (Xf == 0).astype(np.float32)], axis=1)
print(f'Loaded: X_seq={X_seq.shape}, teacher AUC={roc_auc_score(y,teacher_oof):.4f}')

# Load leaf indices
g = np.load('output/gbdt_stage.npz')
leaf_all = g['leaf_all']; del g

DEV = 'cuda'
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_offs = []

for fi, (ti, vi) in enumerate(skf.split(X_seq, y)):
    tf = time.time()
    torch.cuda.empty_cache()
    print(f'Fold {fi+1}...', end=' ', flush=True)

    model = TCNWithLeafEmbedding(
        in_channels=3, tcn_channels=[32, 32, 32, 16],
        kernel_size=5, dropout=0.3, n_trees=200, num_leaves=31,
        leaf_embed_dim=4, leaf_output_dim=32, use_prior=True,
    ).to(DEV)

    crit_ce = RecallOrientedFocalLoss(alpha=0.75, gamma=2.0, recall_weight=3.0)
    crit_kd = torch.nn.MSELoss()
    kd_weight = 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

    ds = TensorDataset(
        torch.FloatTensor(X_seq[ti]), torch.LongTensor(leaf_all[ti]),
        torch.FloatTensor(y[ti]), torch.FloatTensor(teacher_oof[ti]),
    )
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    bf, bs, pt = 0, None, 0
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
            probs = torch.sigmoid(model(xv, lv, pv)).cpu().numpy()
        v = max((f1_score(y[vi], (probs > t).astype(int), zero_division=0)
                 for t in np.arange(0.2, 0.8, 0.01)), default=0)
        if v > bf:
            bf, bs, pt = v, model.state_dict().copy(), 0
        else:
            pt += 1
        if pt >= 7:
            break

    model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        xv = torch.FloatTensor(X_seq[vi]).to(DEV)
        lv = torch.LongTensor(leaf_all[vi]).to(DEV)
        pv = torch.FloatTensor(teacher_oof[vi]).to(DEV)
        oof_fold = torch.sigmoid(model(xv, lv, pv)).cpu().numpy()
    fold_offs.append((vi, oof_fold))
    print(f'F1={bf:.4f} AUC={roc_auc_score(y[vi], oof_fold):.4f} ({time.time()-tf:.0f}s)')
    del model, opt, sch, ds, dl
    torch.cuda.empty_cache()

oof_mt = np.zeros(len(y))
for vi, oof_fold in fold_offs:
    oof_mt[vi] = oof_fold

auc_mt = roc_auc_score(y, oof_mt)
f1_mt, _, rec_mt, prec_mt = best_f1_score(y, oof_mt)

print(f'\nMulti-teacher TCN: AUC={auc_mt:.4f} F1={f1_mt:.4f} Rec={rec_mt:.4f} Prec={prec_mt:.4f}')
print(f'  vs V225 (KD): AUC=0.9783 F1=0.8433')
print(f'  Delta:          AUC={auc_mt-0.9783:+.4f} F1={f1_mt-0.8433:+.4f}')
print(f'  Time: {(time.time()-t0)/60:.1f} min')

np.savez('output/tcn_mt_results.npz', oof_mt=oof_mt, y=y)
