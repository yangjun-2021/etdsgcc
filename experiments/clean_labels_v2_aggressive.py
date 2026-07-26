"""Aggressive label cleaning using strong OOF consensus."""
import os, sys
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything

seed_everything(SEED)


def main():
    y_orig = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz'))['flags'].astype(int)

    # Load strong OOFs
    oofs = {}
    files = [
        ('amst_3ch_recall10_oof.npz', 'oof_amst_3ch_recall10', 'AMST'),
        ('amst_3ch_strong_prior_oof.npz', 'oof_amst_3ch_strong_prior', 'AMST-strong'),
        ('informer_3ch_strong_prior_oof.npz', 'oof_informer_3ch_strong_prior', 'Informer'),
        ('patch_transformer_raw_3ch_recall_oof.npz', 'oof_patch_transformer_raw_3ch_recall', 'PatchT'),
        ('supcon_raw_3ch_oof.npz', 'oof_supcon_raw_3ch', 'SupCon'),
        ('mega_boost_enhanced.npz', 'oof_final', 'MegaBoost'),
    ]
    for fn, key, label in files:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, fn))
            oofs[label] = d[key]
        except Exception as e:
            print(f'Not loaded {label}: {e}')

    P = np.column_stack(list(oofs.values()))
    consensus = P.mean(axis=1)

    fp_th = 0.10  # lower than v1 (0.20): more aggressive on predicted pos -> flip to neg
    fn_th = 0.90  # higher than v1 (0.80): more aggressive on predicted neg -> flip to pos

    flipped_fp = (y_orig == 1) & (consensus < fp_th)
    flipped_fn = (y_orig == 0) & (consensus > fn_th)

    y_clean = y_orig.copy()
    y_clean[flipped_fp] = 0
    y_clean[flipped_fn] = 1

    print(f'Original pos/neg: {y_orig.sum()}/{len(y_orig)-y_orig.sum()}')
    print(f'Cleaned pos/neg: {y_clean.sum()}/{len(y_clean)-y_clean.sum()}')
    print(f'Flipped pos->neg: {flipped_fp.sum()}')
    print(f'Flipped neg->pos: {flipped_fn.sum()}')
    print(f'Agreement: {(y_orig==y_clean).mean():.4f}')

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'cleaned_labels_v2.npz'),
        y_clean=y_clean,
        y_orig=y_orig,
        consensus=consensus,
        flipped_fp=flipped_fp,
        flipped_fn=flipped_fn,
        fp_th=fp_th,
        fn_th=fn_th,
    )
    print(f'Saved to {os.path.join(OUTPUT_DIR, "cleaned_labels_v2.npz")}')


if __name__ == '__main__':
    main()
