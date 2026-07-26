"""Selective AUC+F1 optimization over diverse OOFs."""
import numpy as np, glob, time
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from scipy.optimize import minimize

OD = r'D:\Project\ThiefElectricity\output'

def load_npz(prefix, key):
    files = sorted(glob.glob(f'{OD}/{prefix}*.npz'), reverse=True)
    return np.load(files[0], allow_pickle=True)[key]

t0 = time.time()
y = np.load(glob.glob(f'{OD}/v225_results_*.npz')[0], allow_pickle=True)['y']

# Mix: external strong + our diverse models
sources = {
    'v213': load_npz('v213_results_', 'oof_v213'),
    'v71_innov': load_npz('v71_oofs_', 'innov'),
    'v229_iso': load_npz('v229_results_', 'oof_iso'),
    'v219': load_npz('v219_results_', 'oof_final'),
    'v216': load_npz('v216_results_', 'oof_final'),
}
our = np.load('output/tcn_kd_results.npz')
sources['our_tcn_kd'] = our['oof_tcn_kd']
sources['our_stacker'] = our['oof_stacker']
sources['our_blend'] = our['oof_blend']

print(f'Diverse OOFs ({len(sources)}):')
for nm in sorted(sources.keys()):
    auc = roc_auc_score(y, sources[nm])
    bf = max((f1_score(y, (sources[nm] > th).astype(int), zero_division=0)
              for th in np.arange(0.05, 0.95, 0.005)), default=0)
    print(f'  {nm:15s}: AUC={auc:.4f} F1={bf:.4f}')

names = list(sources.keys())
n = len(names)
P = np.column_stack([sources[nm] for nm in names])

# Stage 1: AUC optimization (fast, differentiable via Nelder-Mead)
def neg_auc(w):
    w = np.abs(w); w = w / (w.sum() + 1e-10)
    try: return -roc_auc_score(y, P @ w)
    except: return 1.0

res = minimize(neg_auc, np.ones(n)/n, method='Nelder-Mead',
               options={'maxiter': 200, 'xatol': 1e-6})
w_auc = np.abs(res.x); w_auc = w_auc / w_auc.sum()
print(f'\nAUC-optimized: AUC={roc_auc_score(y, P @ w_auc):.4f}')

# Stage 2: F1 hill-climb from AUC-optimal
def f1_score_w(wt):
    wt = np.maximum(wt, 0); wt = wt / wt.sum()
    p = P @ wt
    return max((f1_score(y, (p > th).astype(int), zero_division=0)
                for th in np.arange(0.05, 0.95, 0.005)), default=0)

best_s = f1_score_w(w_auc)
best_w = w_auc.copy()

for it in range(200):
    improved = False
    for i in np.random.permutation(n):
        for d in [0.02, -0.02, 0.05, -0.05, 0.10, -0.10]:
            tw = best_w.copy(); tw[i] += d
            tw = np.maximum(tw, 0); tw = tw / tw.sum()
            s = f1_score_w(tw)
            if s > best_s + 1e-6:
                best_s, best_w = s, tw.copy(); improved = True
    if not improved: break

p = P @ best_w
bf, bt = 0, 0.5
for th in np.arange(0.05, 0.95, 0.001):
    pred = (p > th).astype(int)
    if pred.sum() == 0: continue
    f = f1_score(y, pred, zero_division=0)
    if f > bf: bf, bt = f, th

pred = (p > bt).astype(int)
tp = ((pred == 1) & (y == 1)).sum()
fp = ((pred == 1) & (y == 0)).sum()
fn = ((pred == 0) & (y == 1)).sum()

print(f'\nSMART BLEND ({n} OOFs):')
print(f'  F1={bf:.4f} AUC={roc_auc_score(y,p):.4f} Rec={recall_score(y,pred):.4f} Prec={precision_score(y,pred):.4f} th={bt:.3f}')
print(f'  TP={tp} FP={fp} FN={fn}')
print(f'  vs V225 (F1=0.8457): {bf-0.8457:+.4f}')
for i, nm in enumerate(names):
    if best_w[i] > 0.01:
        print(f'  {nm:15s}: w={best_w[i]:.4f}')
print(f'Time: {(time.time()-t0)/60:.1f} min')
np.savez('output/smart_blend.npz', oof_final=p, weights=best_w, names=names, y=y)
