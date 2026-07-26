"""Recall-constrained analysis + comprehensive metrics. Fast version."""
import numpy as np, glob
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

OD = r'D:\Project\ThiefElectricity\output'
def load(prefix, key):
    return np.load(sorted(glob.glob(f'{OD}/{prefix}*.npz'), reverse=True)[0], allow_pickle=True)[key]

y = load('v225_results_', 'y')

models = {
    'Super-GBDT': np.load('output/super_gbdt.npz')['oof_super'],
    'V225': load('v225_results_', 'oof_final'),
    'V213': load('v213_results_', 'oof_v213'),
    'V71_Innov': load('v71_oofs_', 'innov'),
    'V219': load('v219_results_', 'oof_final'),
}

TH_COARSE = np.arange(0.05, 0.95, 0.005)
TH_FINE = np.arange(0.05, 0.95, 0.002)

SEP = '=' * 80
print(SEP)
print('  UNCONSTRAINED BEST F1')
print(SEP)
for nm, oof in models.items():
    bf, bt, br, bp = 0, 0.5, 0, 0
    for th in TH_COARSE:
        pred = (oof > th).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > bf: bf, bt, br, bp = f, th, recall_score(y,pred), precision_score(y,pred)
    tp = ((oof>bt)&(y==1)).sum(); fp = ((oof>bt)&(y==0)).sum(); fn = ((oof<=bt)&(y==1)).sum()
    auc = roc_auc_score(y, oof)
    print(f'  {nm:<14s}  AUC={auc:.4f}  F1={bf:.4f}  Rec={br:.4f}  Prec={bp:.4f}  TP={tp}  FP={fp}  FN={fn}  th={bt:.3f}')

# Recall-constrained (coarse grid for speed)
print(f'\n{SEP}')
print('  RECALL-CONSTRAINED F1')
print(SEP)
for target_rec in [0.80, 0.83, 0.85, 0.88, 0.90, 0.92]:
    print(f'\n  --- Target Recall >= {target_rec:.2f} ---')
    for nm, oof in models.items():
        bf, bt, br, bp = 0, 0.5, 0, 0
        for th in TH_FINE:
            pred = (oof > th).astype(int)
            if pred.sum() == 0: continue
            rec = recall_score(y, pred, zero_division=0)
            if rec < target_rec: continue
            f = f1_score(y, pred, zero_division=0)
            if f > bf: bf, bt, br, bp = f, th, rec, precision_score(y,pred,zero_division=0)
        if bf > 0:
            tp = ((oof>bt)&(y==1)).sum(); fp = ((oof>bt)&(y==0)).sum(); fn = ((oof<=bt)&(y==1)).sum()
            print(f'  {nm:<14s}  F1={bf:.4f}  Rec={br:.4f}  Prec={bp:.4f}  TP={tp}  FP={fp}  FN={fn}  th={bt:.3f}')
        else:
            max_rec = max(recall_score(y,(oof>th).astype(int)) for th in np.arange(0.01,0.99,0.005))
            print(f'  {nm:<14s}  NOT REACHABLE (max recall={max_rec:.4f})')

# Theoretical ceiling
print(f'\n{SEP}')
print('  THEORETICAL F1 CEILING (Super-GBDT)')
print(SEP)
oof = models['Super-GBDT']
pos_probs = oof[y == 1]; neg_probs = oof[y == 0]
sorted_pos = np.sort(pos_probs)[::-1]
for tr in [0.80, 0.83, 0.85, 0.88, 0.90, 0.92, 0.95]:
    k = int(len(pos_probs) * tr)
    th = sorted_pos[min(k, len(sorted_pos) - 1)] if k > 0 else 0
    fn = len(pos_probs) - k
    fp = int((neg_probs >= th).sum())
    f1c = 2 * k / (2 * k + fp + fn) if (2 * k + fp + fn) > 0 else 0
    print(f'  Rec={tr:.2f}: max F1={f1c:.4f}  TP={k}  FP={fp}  FN={fn}  th={th:.4f}')

