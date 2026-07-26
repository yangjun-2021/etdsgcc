import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'D:\Project\ThiefElectricity')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')

SEED = 42; N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

from dl_data import load_raw_data, prepare_sequences, prepare_aggregates, best_f1, evaluate

print(f"Device: {DEVICE}")
print("Loading data...")
X_raw, y = load_raw_data()
X_seq = prepare_sequences(X_raw)
print(f"X_seq: {X_seq.shape}")

import glob
OOF_DIR = r'D:\Project\ThiefElectricity\output'
all_oof_files = sorted(glob.glob(os.path.join(OOF_DIR, 'v*_results*.npz')))

oof_dict = {}
for fpath in all_oof_files:
    try:
        d = np.load(fpath, allow_pickle=True)
        oof = d.get('oof_final')
        if oof is not None and len(oof) == len(y):
            vname = fpath.split('v')[-1].split('_')[0]
            oof_dict[vname] = oof
    except: pass

print(f"\nLoaded {len(oof_dict)} OOF sources:")
for k in sorted(oof_dict.keys(), key=lambda x: -float(x) if x.isdigit() else 0):
    o = oof_dict[k]
    bf, bt = best_f1(y, o)
    auc = roc_auc_score(y, o)
    pred = (o > bt).astype(int); r = recall_score(y, pred); p = precision_score(y, pred)
    print(f"  V{k}: F1={bf:.4f} AUC={auc:.4f} Rec={r:.4f} Prec={p:.4f} th={bt:.3f}")

class MegaTCN(nn.Module):
    def __init__(self, in_ch=3, hidden=32, dropout=0.3, n_priors=1):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, padding=3), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden*2, 7, padding=6, dilation=2), nn.BatchNorm1d(hidden*2), nn.GELU(),
            nn.Conv1d(hidden*2, hidden*2, 7, padding=12, dilation=4), nn.BatchNorm1d(hidden*2), nn.GELU(),
            nn.Conv1d(hidden*2, hidden, 7, padding=24, dilation=8), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 7, padding=48, dilation=16), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + n_priors, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, priors=None):
        x = self.tcn(x); x = torch.mean(x, dim=2)
        if priors is not None:
            x = torch.cat([x, priors], dim=1)
        return self.head(x).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0): super().__init__(); self.a, self.g = alpha, gamma
    def forward(self, logits, t):
        p = torch.sigmoid(logits).clamp(1e-6,1-1e-6)
        return (-self.a*(1-p)**self.g*torch.log(p)*t - (1-self.a)*p**self.g*torch.log(1-p)*(1-t)).mean()

def pred_batch(model, X_np, p_np, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X_np), bs):
            xb = torch.FloatTensor(X_np[i:i+bs]).to(DEVICE)
            pb = torch.FloatTensor(p_np[i:i+bs]).to(DEVICE) if p_np is not None else None
            out.append(torch.sigmoid(model(xb, pb)).cpu().numpy())
    return np.concatenate(out)

top_oof_keys = sorted(oof_dict.keys(), key=lambda k: best_f1(y, oof_dict[k])[0], reverse=True)[:5]
print(f"\nTraining TCN for top {len(top_oof_keys)} OOF sources...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
tcn_oofs = {}

for vk in top_oof_keys:
    print(f"\n{'='*50}\n  TCN with V{vk} as prior (F1={best_f1(y, oof_dict[vk])[0]:.4f})")
    oof_prior = oof_dict[vk]
    oof_tcn = np.zeros(len(y))
    
    for fi, (ti, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
        np.random.seed(SEED+fi); torch.manual_seed(SEED+fi)
        model = MegaTCN(in_ch=3, hidden=32, dropout=0.3, n_priors=1).to(DEVICE)
        crit = FocalLoss(alpha=0.75, gamma=2.0)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25, eta_min=1e-6)
        
        ds = TensorDataset(torch.FloatTensor(X_seq[ti]), torch.FloatTensor(oof_prior[ti]).reshape(-1,1), torch.FloatTensor(y[ti]))
        dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
        
        best_f1_fold = 0; best_state = None; patience = 0
        for ep in range(25):
            model.train()
            for bx, bp, by in dl:
                bx, bp, by = bx.to(DEVICE), bp.to(DEVICE), by.to(DEVICE)
                opt.zero_grad(); loss = crit(model(bx, bp), by)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sch.step()
            vp = pred_batch(model, X_seq[vi], oof_prior[vi].reshape(-1,1))
            bf = max((f1_score(y[vi], (vp>t).astype(int), zero_division=0) for t in np.arange(0.2,0.8,0.01)), default=0)
            if bf > best_f1_fold: best_f1_fold = bf; best_state = model.state_dict().copy(); patience = 0
            else: patience += 1
            if patience >= 5: break
        model.load_state_dict(best_state)
        oof_tcn[vi] = pred_batch(model, X_seq[vi], oof_prior[vi].reshape(-1,1))
        print(f"  Fold {fi+1}: F1={best_f1_fold:.4f}")
    
    bf, bt = best_f1(y, oof_tcn)
    print(f"  TCN_V{vk} OOF: best F1={bf:.4f} AUC={roc_auc_score(y, oof_tcn):.4f}")
    tcn_oofs[f'TCN_V{vk}'] = oof_tcn

print(f"\n{'='*60}")
print("MEGA ENSEMBLE: Hill-Climb over ALL OOFs")
print(f"{'='*60}")

all_oofs = {}
for k, v in oof_dict.items(): all_oofs[f'V{k}'] = v
for k, v in tcn_oofs.items(): all_oofs[k] = v

oof_names = sorted(all_oofs.keys())
n_oof = len(oof_names)
print(f"  Total OOF sources: {n_oof}")
for name in oof_names:
    bf, bt = best_f1(y, all_oofs[name])
    print(f"    {name:>12s}: F1={bf:.4f}")

weights = np.ones(n_oof) / n_oof

def ensemble_proba(weights):
    ep = np.zeros(len(y))
    for i, name in enumerate(oof_names):
        ep += weights[i] * all_oofs[name]
    return ep

def search_f1(weights):
    prob = ensemble_proba(weights)
    bf, _ = best_f1(y, prob)
    return bf

best_w = weights.copy(); best_f1_overall = search_f1(best_w)
print(f"\n  Initial (equal): F1={best_f1_overall:.4f}")

for iteration in range(1000):
    improved = False
    order = np.random.permutation(n_oof)
    for i in order:
        for delta in [0.005, -0.005, 0.01, -0.01, 0.02, -0.02]:
            test_w = best_w.copy()
            test_w[i] += delta
            test_w = np.maximum(test_w, 0.0)
            test_w = test_w / test_w.sum()
            f1_test = search_f1(test_w)
            if f1_test > best_f1_overall:
                best_f1_overall = f1_test; best_w = test_w.copy(); improved = True
    if iteration % 100 == 0:
        print(f"  Iter {iteration}: F1={best_f1_overall:.4f}")
    if not improved: break

print(f"\n  Final weights after hill-climb:")
for i, name in enumerate(oof_names):
    if best_w[i] > 0.01:
        print(f"    {name:>12s}: {best_w[i]:.4f}")

final_proba = ensemble_proba(best_w)
final_bf, final_bt = best_f1(y, final_proba)
final_pred = (final_proba > final_bt).astype(int)
final_recall = recall_score(y, final_pred)
final_precision = precision_score(y, final_pred)
final_auc = roc_auc_score(y, final_proba)

print(f"\n  MEGA ENSEMBLE RESULTS:")
print(f"    F1:       {final_bf:.4f}")
print(f"    Recall:   {final_recall:.4f}")
print(f"    Precision:{final_precision:.4f}")
print(f"    AUC:      {final_auc:.4f}")
print(f"    Threshold:{final_bt:.3f}")

tp = ((final_pred==1)&(y==1)).sum(); fp = ((final_pred==1)&(y==0)).sum()
fn = ((final_pred==0)&(y==1)).sum(); tn = ((final_pred==0)&(y==0)).sum()
print(f"    TP/FP/FN/TN: {tp}/{fp}/{fn}/{tn}")

print(f"\n  Recall-constrained search (Recall >= 0.90):")
rc_best_f1 = 0; rc_best_th = 0.5; rc_best_recall = 0; rc_best_precision = 0
for th in np.arange(0.05, 0.95, 0.002):
    pred = (final_proba > th).astype(int)
    r = recall_score(y, pred, zero_division=0)
    if r < 0.90: continue
    f1 = f1_score(y, pred, zero_division=0)
    if f1 > rc_best_f1:
        rc_best_f1 = f1; rc_best_th = th; rc_best_recall = r
        rc_best_precision = precision_score(y, pred, zero_division=0)

if rc_best_recall >= 0.90:
    print(f"    F1={rc_best_f1:.4f}, Recall={rc_best_recall:.4f}, Precision={rc_best_precision:.4f}, th={rc_best_th:.3f}")
else:
    print(f"    IMPOSSIBLE: max Recall achievable is below 0.90")
    best_rec_at_any_th = max(recall_score(y, (final_proba>t).astype(int)) for t in np.arange(0.01, 0.99, 0.002))
    print(f"    Max achievable Recall: {best_rec_at_any_th:.4f}")

print(f"\n  XGBoost Meta-Learner with all OOFs as features:")
X_meta = np.column_stack([all_oofs[name].reshape(-1,1) for name in oof_names] + 
                          [np.abs(all_oofs[a].reshape(-1,1) - all_oofs[b].reshape(-1,1))
                           for a in oof_names[:4] for b in oof_names[4:]])

oof_meta = np.zeros(len(y))
import xgboost as xgb
for fi, (ti, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
    Xt, Xv = X_meta[ti], X_meta[vi]; yt, yv = y[ti], y[vi]
    pw = (yt==0).sum() / max((yt==1).sum(),1)
    m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                          scale_pos_weight=pw, subsample=0.8, colsample_bytree=0.8,
                          tree_method='hist', random_state=SEED, verbosity=0)
    m.fit(Xt, yt); oof_meta[vi] = m.predict_proba(Xv)[:,1]

meta_bf, meta_bt = best_f1(y, oof_meta)
meta_pred = (oof_meta > meta_bt).astype(int)
print(f"    F1={meta_bf:.4f}, Recall={recall_score(y, meta_pred):.4f}, Precision={precision_score(y, meta_pred):.4f}, AUC={roc_auc_score(y, oof_meta):.4f}")

meta_rc_f1 = 0; meta_rc_th = 0.5
for th in np.arange(0.05, 0.95, 0.002):
    pred = (oof_meta > th).astype(int)
    r = recall_score(y, pred, zero_division=0)
    if r < 0.90: continue
    f1 = f1_score(y, pred, zero_division=0)
    if f1 > meta_rc_f1: meta_rc_f1=f1; meta_rc_th=th

if meta_rc_f1 > 0:
    print(f"    Recall-constrained F1={meta_rc_f1:.4f} (th={meta_rc_th:.3f})")
else:
    max_rec = max(recall_score(y, (oof_meta>t).astype(int)) for t in np.arange(0.01,0.99,0.002))
    print(f"    Max Recall: {max_rec:.4f} - Recall>=0.90 IMPOSSIBLE")

np.savez_compressed(os.path.join(OUTPUT_DIR, 'mega_ensemble_results.npz'),
                    final_proba=final_proba, oof_meta=oof_meta, weights=best_w,
                    oof_names=oof_names, y=y)
print(f"\nResults saved.")