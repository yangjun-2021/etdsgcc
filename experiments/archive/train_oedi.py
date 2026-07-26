import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

print(f"Device: {DEVICE}")
print("Loading OEDI data...")
df = pd.read_csv('data/df.csv')

df = df[df['Class'] != '0'].reset_index(drop=True)

feat_cols = ['Electricity:Facility [kW](Hourly)', 'Fans:Electricity [kW](Hourly)',
             'Cooling:Electricity [kW](Hourly)', 'Heating:Electricity [kW](Hourly)',
             'InteriorLights:Electricity [kW](Hourly)', 'InteriorEquipment:Electricity [kW](Hourly)',
             'Gas:Facility [kW](Hourly)', 'Heating:Gas [kW](Hourly)',
             'InteriorEquipment:Gas [kW](Hourly)', 'Water Heater:WaterSystems:Gas [kW](Hourly)']

df['theft_binary'] = (df['theft'] != 'Normal').astype(int)

buiding_types = sorted(df['Class'].unique())
theft_types = sorted(df['theft'].unique())
print(f"Building types: {buiding_types}")
print(f"Theft types: {theft_types}")

w, s = 720, 168
X_list, y_list, groups = [], [], []

for bt in buiding_types:
    for tt in theft_types:
        mask = (df['Class'] == bt) & (df['theft'] == tt)
        sub = df.loc[mask, feat_cols].values.astype(np.float32)
        if len(sub) < w:
            continue
        nw = (len(sub) - w) // s + 1
        for wi in range(nw):
            X_list.append(sub[wi*s:wi*s+w])
            y_list.append(0 if tt == 'Normal' else 1)
            groups.append(bt)

X_seq = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
groups = np.array(groups)

feat_means = X_seq.mean(axis=(0,1))
feat_stds = X_seq.std(axis=(0,1)) + 1e-8
X_seq = (X_seq - feat_means[np.newaxis, np.newaxis, :]) / feat_stds[np.newaxis, np.newaxis, :]
X_seq = X_seq.transpose(0, 2, 1)

mask_ch = np.ones((X_seq.shape[0], 1, X_seq.shape[2]), dtype=np.float32)
X_seq = np.concatenate([X_seq, mask_ch], axis=1)

print(f"X_seq: {X_seq.shape}, y: {y.shape}, theft_rate: {y.mean()*100:.2f}%")

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_assignments = np.full(len(y), -1, dtype=int)
for fi, (_, vi) in enumerate(sgkf.split(X_seq, y, groups)):
    fold_assignments[vi] = fi

class SimpleTCN(nn.Module):
    def __init__(self, in_ch, hidden=32, dropout=0.3):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=16, dilation=8), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
    
    def forward(self, x):
        return self.head(self.tcn(x).mean(dim=2)).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.55, gamma=1.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        return (-self.alpha*(1-p)**self.gamma*torch.log(p)*targets - (1-self.alpha)*p**self.gamma*torch.log(1-p)*(1-targets)).mean()

def pred_batch(model, X_np, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_np), bs):
            out.append(torch.sigmoid(model(torch.FloatTensor(X_np[i:i+bs]).to(DEVICE))).cpu().numpy())
    return np.concatenate(out)

def evaluate_proba(y, prob, pref=''):
    best, bt = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.005):
        pred = (prob > t).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred)
        if f > best: best, bt = f, t
    pred = (prob > bt).astype(int)
    f, r, p, a = f1_score(y, pred), recall_score(y, pred), precision_score(y, pred), roc_auc_score(y, prob)
    print(f'{pref}F1={f:.4f} Recall={r:.4f} Prec={p:.4f} AUC={a:.4f} th={bt:.3f}')
    return {'f1': f, 'recall': r, 'precision': p, 'auc': a, 'th': bt}

print("\n" + "="*60)
print("OEDI: TCN Only (5-fold CV by building type)")
print("="*60)

oof_tcn = np.zeros(len(y))
for fi in range(N_FOLDS):
    ti = np.where(fold_assignments != fi)[0]
    vi = np.where(fold_assignments == fi)[0]
    print(f"\n  Fold {fi+1}: train={len(ti)}, val={len(vi)}")
    
    model = SimpleTCN(in_ch=11, hidden=32).to(DEVICE)
    crit = FocalLoss(alpha=0.55, gamma=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-6)
    
    ds = TensorDataset(torch.FloatTensor(X_seq[ti]), torch.FloatTensor(y[ti]))
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
    
    best_f1 = 0; best_state = None; patience = 0
    
    for epoch in range(30):
        model.train()
        for bx, by_ in dl:
            bx, by_ = bx.to(DEVICE), by_.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bx), by_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        
        vp = pred_batch(model, X_seq[vi])
        bf = 0
        for t in np.arange(0.1, 0.9, 0.01):
            f = f1_score(y[vi], (vp > t).astype(int))
            if f > bf: bf = f
        
        if bf > best_f1:
            best_f1 = bf; best_state = model.state_dict().copy(); patience = 0
        else:
            patience += 1
        
        if epoch % 5 == 0 or epoch == 29: print(f"    E{epoch+1}: F1={bf:.4f}")
        if patience >= 6: print(f"    Early stop"); break
    
    model.load_state_dict(best_state)
    oof_tcn[vi] = pred_batch(model, X_seq[vi])

res_tcn = evaluate_proba(y, oof_tcn, pref='  TCN: ')

print("\n" + "="*60)
print("OEDI: LightGBM Baseline")
print("="*60)

stat_features = []
for ch in range(X_seq.shape[1]):
    ch_data = X_seq[:, ch, :]
    stat_features.append([
        ch_data.mean(axis=1), ch_data.std(axis=1), np.percentile(ch_data, 25, axis=1),
        np.percentile(ch_data, 75, axis=1), np.percentile(ch_data, 10, axis=1),
        np.percentile(ch_data, 90, axis=1),
    ])
X_stat = np.column_stack([arr for feat in stat_features for arr in feat])
X_stat = np.nan_to_num(X_stat, nan=0.0)

import lightgbm as lgb
oof_lgb = np.zeros(len(y))
for fi in range(N_FOLDS):
    ti = np.where(fold_assignments != fi)[0]
    vi = np.where(fold_assignments == fi)[0]
    pw = (y[ti]==0).sum() / max((y[ti]==1).sum(), 1)
    m = lgb.LGBMClassifier(n_estimators=500, max_depth=6, num_leaves=31, scale_pos_weight=pw,
                            learning_rate=0.05, random_state=SEED, verbose=-1)
    m.fit(X_stat[ti], y[ti])
    oof_lgb[vi] = m.predict_proba(X_stat[vi])[:,1]

res_lgb = evaluate_proba(y, oof_lgb, pref='  LightGBM: ')

print("\n" + "="*60)
print("OEDI: Simple Ensemble + Meta")
print("="*60)

oof_avg = (oof_lgb + oof_tcn) / 2
res_avg = evaluate_proba(y, oof_avg, pref='  AVG(LGB+TCN): ')

import xgboost as xgb
oof_meta = np.zeros(len(y))
X_meta = np.column_stack([X_stat, oof_lgb.reshape(-1,1), oof_tcn.reshape(-1,1)])
for fi in range(N_FOLDS):
    ti = np.where(fold_assignments != fi)[0]
    vi = np.where(fold_assignments == fi)[0]
    pw = (y[ti]==0).sum() / max((y[ti]==1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            scale_pos_weight=pw, tree_method='hist', random_state=SEED, verbosity=0)
    m.fit(X_meta[ti], y[ti])
    oof_meta[vi] = m.predict_proba(X_meta[vi])[:,1]
res_meta = evaluate_proba(y, oof_meta, pref='  Meta: ')

print("\n" + "="*60)
print("OEDI Final Comparison")
print("="*60)
print(f"{'Component':<20s} {'F1':>8s} {'Recall':>8s} {'Precision':>8s} {'AUC':>8s}")
print("-"*55)
for name, r in [('LightGBM', res_lgb), ('TCN', res_tcn), ('AVG(LGB+TCN)', res_avg), ('Meta Stacking', res_meta)]:
    print(f"{name:<20s} {r['f1']:8.4f} {r['recall']:8.4f} {r['precision']:8.4f} {r['auc']:8.4f}")

np.savez_compressed(os.path.join(OUTPUT_DIR, 'oedi_final_oof.npz'), oof_lgb=oof_lgb, oof_tcn=oof_tcn, oof_meta=oof_meta, y=y)
print(f"\nSaved to {OUTPUT_DIR}/oedi_final_oof.npz")