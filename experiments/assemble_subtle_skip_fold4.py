"""Assemble subtle OOF from folds 0,1,2 (original) and fold 4 (fold5-only), skipping fold 3."""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR

flags = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags']
oof = np.zeros(len(flags), dtype=np.float32)

# Load original folds 0,1,2
for fi in [0, 1, 2]:
    fpath = os.path.join(OUTPUT_DIR, f'amst_3ch_synthetic_subtle_v3_fold{fi}.npz')
    d = np.load(fpath)
    oof[d['vi']] = d['oof']
    print(f'Loaded original fold {fi+1}: {len(d["vi"])} samples')

# Load fold5-only fold 4
d = np.load(os.path.join(OUTPUT_DIR, 'amst_3ch_synthetic_subtle_v3_fold5_only_fold4.npz'))
oof[d['vi']] = d['oof']
print(f'Loaded fold5-only fold 5: {len(d["vi"])} samples')

covered = (oof > 0).sum()
print(f'Covered samples: {covered}/{len(flags)} ({covered/len(flags)*100:.1f}%)')

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'amst_3ch_synthetic_subtle_v3_oof.npz'),
    oof_amst_3ch_synthetic_subtle_v3=oof,
    flags=flags,
)
print(f'Saved to {os.path.join(OUTPUT_DIR, "amst_3ch_synthetic_subtle_v3_oof.npz")}')
