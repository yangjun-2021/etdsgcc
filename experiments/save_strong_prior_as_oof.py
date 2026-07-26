"""Save strong GBDT prior in the same format as other OOFs."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR
import numpy as np

d = np.load(os.path.join(OUTPUT_DIR, 'strong_gbdt_prior.npz'))
np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'strong_gbdt_prior_oof.npz'),
    oof_strong_gbdt_prior=d['prior'],
    flags=d['flags'],
)
print('Saved strong GBDT prior as OOF.')
