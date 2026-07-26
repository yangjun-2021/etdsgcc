"""Final TCN training (V108-style) + hill-climb blend."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np, time
import pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from utils import seed_everything, best_f1_score
from models import TCNWithLeafEmbedding, RecallOrientedFocalLoss

seed_everything(42)
t0 = time.time()

# V108 preprocessing
print("Loading raw data...")
df = pd.read_csv('data/raw_data.csv')
dc = [c for c in df.columns if '/' in str(c) and len(str(c)) <= 10]
X_raw = df[dc].values.astype(np.float32)
y = df['FLAG'].values.astype(np.int32)
del df
nmk = np.isnan(X_raw)
Xf = np.nan_to_num(X_raw, nan=0.0)
Xl = np.log1p(np.maximum(Xf, 0))
sc = StandardScaler()
Xs = np.clip(sc.fit_transform(Xl).astype(np.float32), -5, 5)
X_seq = np.stack([Xs, nmk.astype(np.float32), (Xf == 0).astype(np.float32)], axis=1)

g = np.load('output/gbdt_stage.npz')
oof_stack = g['oof_stack']
oof_blend = g['oof_blend']
leaf_all = g['leaf_all']
del g

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

    crit = RecallOrientedFocalLoss(alpha=0.75, gamma=2.0, recall_weight=3.0)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

    ds = TensorDataset(
        torch.FloatTensor(X_seq[ti]), torch.LongTensor(leaf_all[ti]),
        torch.FloatTensor(y[ti]), torch.FloatTensor(oof_stack[ti]),
    )
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    bf, bs, pt = 0, None, 0
    for ep in range(50):
        model.train()
        for bx, bl, by, bp in dl:
            bx, bl, by, bp = bx.to(DEV), bl.to(DEV), by.to(DEV), bp.to(DEV)
            opt.zero_grad()
            loss = crit(model(bx, bl, bp), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(X_seq[vi]).to(DEV)
            lv = torch.LongTensor(leaf_all[vi]).to(DEV)
            pv = torch.FloatTensor(oof_stack[vi]).to(DEV)
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
        pv = torch.FloatTensor(oof_stack[vi]).to(DEV)
        oof_fold = torch.sigmoid(model(xv, lv, pv)).cpu().numpy()
    fold_offs.append((vi, oof_fold))
    print(f'F1={bf:.4f} AUC={roc_auc_score(y[vi], oof_fold):.4f} ({time.time()-tf:.0f}s)')
    del model, opt, sch, ds, dl
    torch.cuda.empty_cache()

oof_tcn = np.zeros(len(y))
for vi, oof in fold_offs:
    oof_tcn[vi] = oof

auc_t = roc_auc_score(y, oof_tcn)
f1_t, th_t, rec_t, prec_t = best_f1_score(y, oof_tcn)

# Hill-climb
sources = {'stacker': oof_stack, 'tcn': oof_tcn, 'blend': oof_blend}
names = list(sources.keys())
nw = len(names)
w = np.ones(nw) / nw

def score(wt):
    wt = np.maximum(wt, 0)
    wt = wt / wt.sum()
    p = sum(wt[i] * sources[names[i]] for i in range(nw))
    return best_f1_score(y, p)[0]

best_s = score(w)
best_w = w.copy()
for it in range(500):
    improved = False
    for i in range(nw):
        for dd in [0.01, -0.01, 0.03, -0.03, 0.05, -0.05]:
            tw = best_w.copy()
            tw[i] += dd
            tw = np.maximum(tw, 0)
            tw = tw / tw.sum()
            s = score(tw)
            if s > best_s + 1e-6:
                best_s, best_w = s, tw.copy()
                improved = True
    if not improved:
        break

p = sum(best_w[i] * sources[names[i]] for i in range(nw))
f1_h, th_h, rec_h, prec_h = best_f1_score(y, p)
auc_h = roc_auc_score(y, p)
tp = ((p > th_h) & (y == 1)).sum()
fp = ((p > th_h) & (y == 0)).sum()
fn = ((p <= th_h) & (y == 1)).sum()

print(f'\n{"="*60}')
print('  FINAL RESULTS (0% review)')
print('=' * 60)
print(f'  Stacker: AUC={roc_auc_score(y, oof_stack):.4f} F1={best_f1_score(y, oof_stack)[0]:.4f}')
print(f'  TCN:     AUC={auc_t:.4f} F1={f1_t:.4f} Rec={rec_t:.4f} Prec={prec_t:.4f}')
print(f'  Hill:    AUC={auc_h:.4f} F1={f1_h:.4f} Rec={rec_h:.4f} Prec={prec_h:.4f} th={th_h:.3f}')
print(f'  TP={tp} FP={fp} FN={fn}')
weights_str = {n: f'{w:.3f}' for n, w in zip(names, best_w) if w > 0.01}
print(f'  Weights: {weights_str}')
print(f'  V225:    AUC=0.9804 F1=0.8457 TP=2952 FP=414 FN=663')
print(f'  Gap:     AUC={0.9804-auc_h:.4f} F1={0.8457-f1_h:.4f}')
print(f'  Time: {(time.time()-t0)/60:.1f} min')

np.savez('output/final_v108.npz',
         oof_stack=oof_stack, oof_tcn=oof_tcn, oof_blend=oof_blend,
         oof_hill=p, y=y)
print('Saved to output/final_v108.npz')
