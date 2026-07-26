import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pickle
from config import OUTPUT_DIR
from train_expert_b import train_tcn_sgcc, train_tcn_oedi

print("Loading data...")
d = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))
X_seq = d['X_seq']
sf = np.nan_to_num(d['stat_features'], nan=0.0, posinf=0.0, neginf=0.0)
flags = d['flags']
im = d['impute_mask']

ea = np.load(os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz'))
oof_a = ea['oof_proba']
leaf_lgb = ea['leaf_indices_lgb']
leaf_xgb = ea['leaf_indices_xgb']
leaf_combined = np.concatenate([leaf_lgb, leaf_xgb], axis=1)

print(f'X_seq: {X_seq.shape}, leaf: {leaf_combined.shape}')

oof_b = train_tcn_sgcc(X_seq, sf, flags, im, oof_a, leaf_combined)
print(f'\nTCN training complete. oof_b: {oof_b.shape}')
