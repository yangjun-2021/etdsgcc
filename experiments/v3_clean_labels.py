"""V3 label cleaning: cross-validated consensus over UNTAINTED original-label voters.

Voters are the v3voter_* OOFs: models trained on original SGCC labels with NO
cleaned-label-derived prior features (see experiments/v3_*.py). This breaks the
circular cleaning chain of v1/v2 (whose voters were trained on cleaned labels).

Flips (v1-compatible thresholds):
- label=1 with vote consensus < fp_th  -> 0   (suspected false positive labels)
- label=0 with vote consensus > fn_th  -> 1   (suspected false negative labels)

Outputs:
- output/cleaned_labels_v3.npz  (y_clean, y_orig, consensus, flip masks, voter names)
- figures/v3_flipped_samples.png (curve grid of flipped samples, for the paper)

Usage:
    conda run -n ml python experiments/v3_clean_labels.py
"""
import os
import sys
import glob
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import OUTPUT_DIR, SEED

FP_TH = 0.3
FN_TH = 0.7
MIN_VOTERS = 3
MIN_VOTER_AUC = 0.80  # voters weaker than this vs y_orig are excluded from consensus

# Per-component GBDT OOFs are excluded: their blend (v3voter_gbdt) already
# represents that family; including them would triple-count GBDT in the vote.
EXCLUDE_VOTERS = {'v3voter_gbdt_lgb', 'v3voter_gbdt_xgb', 'v3voter_gbdt_cb'}

# Extra vetted untainted voters outside the v3voter_* family: trained on
# original labels with original-label-derived features only (see
# experiments/v3_audit_provenance.py for the full provenance table).
EXTRA_VOTERS = [
    ('expertB_tcn_leaf', 'sgcc_expert_b.npz', 'oof_proba'),  # TCN + leaf from original Expert A
    ('expertA_gbdt', 'sgcc_expert_a.npz', 'oof_proba'),      # GBDT on stat features, original labels
]


def load_voters(y):
    voters = {}
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, 'v3voter_*_oof.npz'))):
        d = np.load(path)
        for key in d.files:
            if key.startswith('oof_') and d[key].shape == y.shape:
                name = os.path.basename(path).replace('_oof.npz', '')
                if name in EXCLUDE_VOTERS:
                    continue
                voters[name] = d[key].astype(np.float64)
    for name, fname, key in EXTRA_VOTERS:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            d = np.load(path)
            if key in d.files and d[key].shape == y.shape:
                voters[name] = d[key].astype(np.float64)
    return voters


def main():
    cl = np.load(os.path.join(OUTPUT_DIR, 'cleaned_labels_v1.npz'))
    y_orig = cl['y_orig'].astype(int)
    y_clean_v1 = cl['y_clean'].astype(int)

    voters = load_voters(y_orig)
    # Diagnostics: per-voter agreement with y_orig; drop weak voters
    from sklearn.metrics import roc_auc_score
    dropped = []
    for n in list(voters.keys()):
        auc = roc_auc_score(y_orig, voters[n])
        print(f'  {n:28s} AUC_vs_yorig={auc:.4f} mean_p={voters[n].mean():.3f}')
        if auc < MIN_VOTER_AUC:
            dropped.append(n)
            del voters[n]
    if dropped:
        print(f'Dropped weak voters (AUC < {MIN_VOTER_AUC}): {dropped}')
    names = sorted(voters.keys())
    print(f'Voters ({len(names)}): {names}')
    assert len(names) >= MIN_VOTERS, f'need >= {MIN_VOTERS} v3voter OOFs, got {len(names)}'

    P = np.column_stack([voters[n] for n in names])
    consensus = (P > 0.5).mean(axis=1)
    mean_prob = P.mean(axis=1)

    # Flip rule: mean-probability extremes (robust to weak voters: a flip
    # requires ALL voters to be confidently against the label). Vote-fraction
    # consensus is kept for diagnostics/comparability with v1.
    flip_score = mean_prob

    # Threshold sweep for the record (both rules)
    print('\nThreshold sweep (flip counts, mean_prob rule | vote rule):')
    for fp_th in [0.15, 0.2, 0.3]:
        for fn_th in [0.7, 0.8, 0.85]:
            ffp = int(((y_orig == 1) & (flip_score < fp_th)).sum())
            ffn = int(((y_orig == 0) & (flip_score > fn_th)).sum())
            vfp = int(((y_orig == 1) & (consensus < fp_th)).sum())
            vfn = int(((y_orig == 0) & (consensus > fn_th)).sum())
            print(f'  fp_th={fp_th:.2f} fn_th={fn_th:.2f}: mean_prob {ffp:4d}/{ffn:4d} '
                  f'(tot {ffp+ffn:4d}) | votes {vfp:4d}/{vfn:4d} (tot {vfp+vfn:4d})')

    flip_fp = (y_orig == 1) & (flip_score < FP_TH)
    flip_fn = (y_orig == 0) & (flip_score > FN_TH)
    y_clean = y_orig.copy()
    y_clean[flip_fp] = 0
    y_clean[flip_fn] = 1

    print(f'\nV3 cleaning (mean_prob rule, fp_th={FP_TH}, fn_th={FN_TH}): '
          f'flipped {flip_fp.sum()} pos->neg, {flip_fn.sum()} neg->pos, '
          f'total {(flip_fp | flip_fn).sum()} / {len(y_orig)}')
    print(f'Positive rate: original {y_orig.mean()*100:.2f}% -> v3 {y_clean.mean()*100:.2f}% '
          f'(v1 was {y_clean_v1.mean()*100:.2f}%)')
    print(f'V3 vs v1 label diff: {(y_clean != y_clean_v1).sum()} samples')

    save_path = os.path.join(OUTPUT_DIR, 'cleaned_labels_v3.npz')
    np.savez_compressed(
        save_path,
        y_clean=y_clean,
        y_orig=y_orig,
        consensus=consensus,
        mean_prob=mean_prob,
        flipped_fp=flip_fp,
        flipped_fn=flip_fn,
        voter_names=np.array(names),
        fp_th=np.float64(FP_TH),
        fn_th=np.float64(FN_TH),
    )
    print(f'Saved to {save_path}')

    # Conservative variant (0.2/0.8) for the paper's sensitivity analysis
    cons_path = os.path.join(OUTPUT_DIR, 'cleaned_labels_v3_conservative.npz')
    y_cons = y_orig.copy()
    cons_fp = (y_orig == 1) & (flip_score < 0.2)
    cons_fn = (y_orig == 0) & (flip_score > 0.8)
    y_cons[cons_fp] = 0
    y_cons[cons_fn] = 1
    np.savez_compressed(cons_path, y_clean=y_cons, y_orig=y_orig,
                        flipped_fp=cons_fp, flipped_fn=cons_fn,
                        voter_names=np.array(names))
    print(f'Saved conservative variant ({(cons_fp | cons_fn).sum()} flips) to {cons_path}')

    # --- Sanity visualization: curves of flipped samples ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        pre = np.load(os.path.join(OUTPUT_DIR, 'sgcc_preprocessed_raw_3ch.npz'))
        X = pre['X_seq'][:, 0, :]  # value channel

        idx_fp = np.where(flip_fp)[0][:6]   # was theft, relabeled honest
        idx_fn = np.where(flip_fn)[0][:6]   # was honest, relabeled theft
        n_rows = 2
        fig, axes = plt.subplots(n_rows, 6, figsize=(24, 6), sharey=False)
        for j, idx in enumerate(idx_fp):
            axes[0, j].plot(X[idx], lw=0.6)
            axes[0, j].set_title(f'#{idx} pos->neg\ncons={consensus[idx]:.2f}', fontsize=8)
        for j, idx in enumerate(idx_fn):
            axes[1, j].plot(X[idx], lw=0.6, color='r')
            axes[1, j].set_title(f'#{idx} neg->pos\ncons={consensus[idx]:.2f}', fontsize=8)
        axes[0, 0].set_ylabel('pos->neg flips')
        axes[1, 0].set_ylabel('neg->pos flips')
        fig.suptitle('V3 label cleaning: flipped samples (raw consumption)')
        fig.tight_layout()
        fig_path = os.path.join(PROJECT_ROOT, 'figures', 'v3_flipped_samples.png')
        fig.savefig(fig_path, dpi=120)
        print(f'Saved figure to {fig_path}')
    except Exception as e:
        print(f'Figure export failed (non-fatal): {e}')


if __name__ == '__main__':
    main()
