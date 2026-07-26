"""
ETD-SGCC Pipeline: Unified entry point for electricity theft detection.

Usage:
    python run_pipeline.py --dataset sgcc
    python run_pipeline.py --dataset oedi
    python run_pipeline.py --dataset both
    python run_pipeline.py --dataset sgcc --no-advanced
    python run_pipeline.py --dataset sgcc --skip-preprocess
    python run_pipeline.py --dataset sgcc --foundation
    python run_pipeline.py --dataset sgcc --causal
    python run_pipeline.py --dataset sgcc --contrastive
"""
import argparse
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything


def _load_cached_expert_oof(dataset, expert):
    """Load cached OOF probabilities for an expert if available."""
    import numpy as np
    path = os.path.join(OUTPUT_DIR, f'{dataset}_expert_{expert}.npz')
    if os.path.exists(path):
        try:
            data = np.load(path)
            return data['oof_proba']
        except Exception:
            pass
    return None


def _load_cached_informer_oof(dataset):
    """Load cached Informer OOF, preferring the pipeline cache then the legacy file."""
    import numpy as np
    path = os.path.join(OUTPUT_DIR, f'{dataset}_expert_c.npz')
    if os.path.exists(path):
        try:
            data = np.load(path)
            return data['oof_proba']
        except Exception:
            pass
    legacy = os.path.join(OUTPUT_DIR, 'informer_oof.npz')
    if os.path.exists(legacy):
        try:
            data = np.load(legacy)
            return data['oof_informer']
        except Exception:
            pass
    return None


def _load_cached_multiscale_cnn_oof(dataset):
    """Load cached MultiScaleCNN1D Expert C OOF if available."""
    import numpy as np
    path = os.path.join(OUTPUT_DIR, f'{dataset}_expert_c_multiscale.npz')
    if os.path.exists(path):
        try:
            data = np.load(path)
            return data['oof_proba']
        except Exception:
            pass
    return None


def run_sgcc_meta_v2_pipeline():
    """Run SGCC meta-learner v2 using cached OOFs (fast path to F1>0.90)."""
    print("\n" + "=" * 70)
    print("  SGCC META-LEARNER V2 PIPELINE")
    print("=" * 70)
    start_time = time.time()
    import numpy as np

    print("\n[Step 1/2] Loading cached SGCC expert OOFs...")
    oof_proba_a = _load_cached_expert_oof('sgcc', 'a')
    oof_proba_b = _load_cached_expert_oof('sgcc', 'b')
    oof_proba_c = _load_cached_informer_oof('sgcc')

    # Load labels matching existing OOFs
    labels = None
    impute_mask = None
    path_a = os.path.join(OUTPUT_DIR, 'sgcc_expert_a.npz')
    if os.path.exists(path_a):
        data = np.load(path_a)
        labels = data['flags']
    if labels is None:
        raise RuntimeError("Could not load SGCC labels from cached expert_a.npz")

    print(f"  Labels: {labels.shape}, theft rate: {labels.mean():.4f}")

    print("\n[Step 2/2] Training ImprovedMetaLearner v2...")
    from src.training.meta_learner_v2 import ImprovedMetaLearner
    meta_learner = ImprovedMetaLearner(dataset='sgcc')
    results = meta_learner.train(
        stat_features=None,
        labels=labels,
        impute_mask=impute_mask,
        oof_proba_a=oof_proba_a,
        oof_proba_b=oof_proba_b,
        oof_proba_c=oof_proba_c,
        skip_new_experts=True,
    )

    print("\n[Step 3/3] Evaluating SGCC...")
    from src.evaluation.evaluate import evaluate_dataset
    evaluate_dataset('sgcc', results, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n  SGCC Meta-Learner v2 Pipeline completed in {elapsed/60:.1f} minutes")
    return results


def run_sgcc_pipeline(use_advanced_features=True, skip_preprocess=False, use_amst=False,
                       use_informer=True, use_multiscale_cnn=False, use_meta_v2=False):
    """Run the full SGCC pipeline."""
    print("\n" + "=" * 70)
    print("  SGCC DATASET PIPELINE")
    print("=" * 70)
    start_time = time.time()

    print("\n[Step 1/6] Preprocessing SGCC...")
    from src.data.preprocess_sgcc import preprocess_sgcc
    import numpy as np
    cached_path = os.path.join(OUTPUT_DIR, 'sgcc_preprocessed.npz')
    if skip_preprocess and os.path.exists(cached_path):
        print(f"  Loading cached preprocessed data from {cached_path}")
        cached = np.load(cached_path)
        X_seq = cached['X_seq']
        stat_features = cached['stat_features']
        flags = cached['flags']
        impute_mask = cached['impute_mask']
        print(f"  Loaded: X_seq={X_seq.shape}, stat_features={stat_features.shape}")
    else:
        X_seq, stat_features, flags, impute_mask = preprocess_sgcc(
            use_advanced_features=use_advanced_features
        )

    # Fast path: load cached expert OOFs when skipping preprocessing
    oof_proba_a = None
    oof_proba_b = None
    oof_proba_c = None
    leaf_indices = None

    if skip_preprocess:
        print("\n[Step 2-4/6] Attempting to load cached SGCC expert OOFs...")
        oof_proba_a = _load_cached_expert_oof('sgcc', 'a')
        oof_proba_b = _load_cached_expert_oof('sgcc', 'b')
        if use_multiscale_cnn:
            oof_proba_c = _load_cached_multiscale_cnn_oof('sgcc')
        elif use_informer:
            oof_proba_c = _load_cached_informer_oof('sgcc')
        if oof_proba_a is not None:
            print(f"  Loaded Expert A OOF: {oof_proba_a.shape}")
        if oof_proba_b is not None:
            print(f"  Loaded Expert B OOF: {oof_proba_b.shape}")
        if oof_proba_c is not None:
            print(f"  Loaded Expert C OOF: {oof_proba_c.shape}")

    if oof_proba_a is None:
        print("\n[Step 2/6] Training Expert A (GBDT) on SGCC...")
        from src.training.expert_a import ExpertATrainer
        trainer_a = ExpertATrainer(dataset='sgcc')
        oof_proba_a, leaf_indices, _ = trainer_a.train(stat_features, flags, impute_mask=impute_mask)

    if oof_proba_b is None:
        print("\n[Step 3/6] Training Expert B on SGCC...")
        if use_amst:
            print("  (Using AMST-Net: Multi-Scale Mamba-Transformer + HN-SupCon)")
            from src.training.amst_trainer import AMSTTrainer
            trainer_b = AMSTTrainer(dataset='sgcc', use_diffaug=True, use_supcon=True,
                                    use_coteaching=False, epochs=100, batch_size=64,
                                    patience=20, use_prior=True)
            oof_proba_b = trainer_b.train(X_seq, flags, impute_mask=impute_mask,
                                            oof_proba_a=oof_proba_a)
        else:
            print("  (Using TCN+Leaf)")
            from src.training.expert_b import ExpertBTrainer
            trainer_b = ExpertBTrainer(dataset='sgcc')
            oof_proba_b = trainer_b.train(X_seq, stat_features, flags, impute_mask=impute_mask,
                                           oof_proba_a=oof_proba_a, leaf_indices=leaf_indices)

    if use_multiscale_cnn and oof_proba_c is None:
        print("\n[Step 4/6] Training Expert C (MultiScaleCNN1D) on SGCC...")
        from src.training.expert_c_multiscale import ExpertCMultiScaleTrainer
        trainer_c = ExpertCMultiScaleTrainer(dataset='sgcc')
        oof_proba_c = trainer_c.train(X_seq, flags)
    elif use_informer and oof_proba_c is None:
        cached_c = _load_cached_informer_oof('sgcc')
        if cached_c is not None:
            oof_proba_c = cached_c
            print(f"  Loaded Expert C OOF: {oof_proba_c.shape}")
        else:
            print("  No cached Informer OOF found; skipping (CPU training is prohibitively slow).")
            print("  To force training, run src/training/expert_c.py directly.")
    elif not use_informer and not use_multiscale_cnn:
        print("\n[Step 4/6] Training Expert C (Informer) on SGCC...")
        print("  (Skipped)")

    print("\n[Step 5/6] Training Meta-Learner on SGCC...")
    if use_meta_v2:
        from src.training.meta_learner_v2 import ImprovedMetaLearner
        meta_learner = ImprovedMetaLearner(dataset='sgcc')
    else:
        from src.training.meta_learner import MetaLearner
        meta_learner = MetaLearner(dataset='sgcc')
    results = meta_learner.train(stat_features, flags, impute_mask=impute_mask,
                                  oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b,
                                  oof_proba_c=oof_proba_c, skip_new_experts=True)

    print("\n[Step 6/6] Evaluating SGCC...")
    from src.evaluation.evaluate import evaluate_dataset
    evaluate_dataset('sgcc', results, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n  SGCC Pipeline completed in {elapsed/60:.1f} minutes")
    return results

def run_sgcc_foundation_pipeline(use_advanced_features=True):
    """Run SGCC with a masked time-series foundation encoder."""
    print("\n" + "=" * 70)
    print("  SGCC FOUNDATION ENCODER PIPELINE")
    print("=" * 70)
    start_time = time.time()

    print("\n[Step 1/3] Preprocessing SGCC...")
    from src.data.preprocess_sgcc import preprocess_sgcc
    X_seq, stat_features, flags, _ = preprocess_sgcc(use_advanced_features=use_advanced_features)

    print("\n[Step 2/3] Training Foundation Encoder...")
    from src.training.foundation_trainer import FoundationTrainer
    trainer = FoundationTrainer(dataset='sgcc')
    results = trainer.train(X_seq, flags, stat_features=stat_features)

    print("\n[Step 3/3] Evaluating SGCC...")
    from src.evaluation.evaluate import evaluate_dataset
    evaluate_dataset('sgcc', results, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n  SGCC Foundation Pipeline completed in {elapsed/60:.1f} minutes")
    return results

def run_sgcc_causal_pipeline(use_advanced_features=True):
    """Run SGCC with a CATE-TST causal encoder + GBDT hybrid head."""
    print("\n" + "=" * 70)
    print("  SGCC CAUSAL (CATE-TST) PIPELINE")
    print("=" * 70)
    start_time = time.time()

    print("\n[Step 1/3] Preprocessing SGCC...")
    from src.data.preprocess_sgcc import preprocess_sgcc
    X_seq, stat_features, flags, _ = preprocess_sgcc(use_advanced_features=use_advanced_features)

    print("\n[Step 2/3] Training Causal Encoder (CATE-TST)...")
    from src.training.causal_trainer import CausalTrainer
    trainer = CausalTrainer(dataset='sgcc')
    results = trainer.train(X_seq, flags, stat_features=stat_features)

    print("\n[Step 3/3] Evaluating SGCC...")
    from src.evaluation.evaluate import evaluate_dataset
    evaluate_dataset('sgcc', results, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n  SGCC Causal Pipeline completed in {elapsed/60:.1f} minutes")
    return results

def run_sgcc_contrastive_pipeline(use_advanced_features=True):
    """Run SGCC with a contrastive temporal encoder + GBDT hybrid head."""
    print("\n" + "=" * 70)
    print("  SGCC CONTRASTIVE ENCODER PIPELINE")
    print("=" * 70)
    start_time = time.time()

    print("\n[Step 1/3] Preprocessing SGCC...")
    from src.data.preprocess_sgcc import preprocess_sgcc
    X_seq, stat_features, flags, _ = preprocess_sgcc(use_advanced_features=use_advanced_features)

    print("\n[Step 2/3] Training Contrastive Encoder...")
    from src.training.contrastive_trainer import ContrastiveTrainer
    trainer = ContrastiveTrainer(dataset='sgcc')
    results = trainer.train(X_seq, flags, stat_features=stat_features)

    print("\n[Step 3/3] Evaluating SGCC...")
    from src.evaluation.evaluate import evaluate_dataset
    evaluate_dataset('sgcc', results, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n  SGCC Contrastive Pipeline completed in {elapsed/60:.1f} minutes")
    return results

def run_oedi_pipeline(skip_preprocess=False):
    """Run the full OEDI pipeline."""
    print("\n" + "=" * 70)
    print("  OEDI DATASET PIPELINE")
    print("=" * 70)
    start_time = time.time()

    print("\n[Step 1/6] Preprocessing OEDI...")
    from src.data.preprocess_oedi import preprocess_oedi
    X_seq, stat_features, y, fold_assignments = preprocess_oedi()

    print("\n[Step 2/6] Training Expert A (GBDT) on OEDI...")
    from src.training.expert_a import ExpertATrainer
    trainer_a = ExpertATrainer(dataset='oedi')
    oof_proba_a, leaf_indices, _ = trainer_a.train(stat_features, y, fold_assignments=fold_assignments)

    print("\n[Step 3/6] Training Expert B (TCN+Leaf) on OEDI...")
    from src.training.expert_b import ExpertBTrainer
    trainer_b = ExpertBTrainer(dataset='oedi')
    oof_proba_b = trainer_b.train(X_seq, stat_features, y, fold_assignments=fold_assignments,
                                   oof_proba_a=oof_proba_a, leaf_indices=leaf_indices)

    print("\n[Step 4/6] Training Meta-Learner on OEDI...")
    from src.training.meta_learner import MetaLearner
    meta_learner = MetaLearner(dataset='oedi')
    results = meta_learner.train(stat_features, y, fold_assignments=fold_assignments,
                                  oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b,
                                  skip_new_experts=True)

    print("\n[Step 5/6] Evaluating OEDI...")
    from src.evaluation.evaluate import evaluate_dataset
    evaluate_dataset('oedi', results, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n  OEDI Pipeline completed in {elapsed/60:.1f} minutes")
    return results


def main():
    parser = argparse.ArgumentParser(description='ETD-SGCC Pipeline')
    parser.add_argument('--dataset', choices=['sgcc', 'oedi', 'both'], default='sgcc',
                        help='Dataset to run (default: sgcc)')
    parser.add_argument('--no-advanced', action='store_true',
                        help='Disable advanced features (SGCC only)')
    parser.add_argument('--amst', action='store_true',
                        help='Use AMST-Net as Expert B (experimental F1>0.90 path)')
    parser.add_argument('--no-informer', action='store_true',
                        help='Disable Informer Expert C (default: use cached Informer OOF if available)')
    parser.add_argument('--multiscale-cnn', action='store_true',
                        help='Use MultiScaleCNN1D as Expert C instead of Informer')
    parser.add_argument('--meta-v2', action='store_true',
                        help='Use ImprovedMetaLearner v2 (auto-discover OOFs + ensemble selection)')
    parser.add_argument('--foundation', action='store_true',
                        help='Use masked time-series foundation encoder for SGCC')
    parser.add_argument('--causal', action='store_true',
                        help='Use CATE-TST causal encoder for SGCC')
    parser.add_argument('--contrastive', action='store_true',
                        help='Use contrastive temporal encoder for SGCC (recommended)')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip preprocessing if cached')
    args = parser.parse_args()

    seed_everything(SEED)

    use_advanced = not args.no_advanced

    sgcc_results = None
    oedi_results = None

    if args.dataset in ('sgcc', 'both'):
        if args.meta_v2:
            sgcc_results = run_sgcc_meta_v2_pipeline()
        elif args.contrastive:
            sgcc_results = run_sgcc_contrastive_pipeline(use_advanced_features=use_advanced)
        elif args.causal:
            sgcc_results = run_sgcc_causal_pipeline(use_advanced_features=use_advanced)
        elif args.foundation:
            sgcc_results = run_sgcc_foundation_pipeline(use_advanced_features=use_advanced)
        else:
            sgcc_results = run_sgcc_pipeline(use_advanced_features=use_advanced,
                                             skip_preprocess=args.skip_preprocess,
                                             use_amst=args.amst,
                                             use_informer=not args.no_informer,
                                             use_multiscale_cnn=args.multiscale_cnn,
                                             use_meta_v2=args.meta_v2)

    if args.dataset in ('oedi', 'both'):
        oedi_results = run_oedi_pipeline(skip_preprocess=args.skip_preprocess)

    if args.dataset == 'both':
        print("\n" + "=" * 70)
        print("  CROSS-DATASET COMPARISON")
        print("=" * 70)
        sgcc = sgcc_results
        oedi = oedi_results

        print(f"\n  {'Metric':<15s}  {'SGCC':>10s}  {'OEDI':>10s}")
        print(f"  {'-'*40}")

        sgcc_f1 = sgcc.get('best_f1_unconstrained', sgcc.get('best_f1', 0))
        oedi_f1 = oedi.get('best_f1', 0)
        print(f"  {'F1':<15s}  {sgcc_f1:>10.4f}  {oedi_f1:>10.4f}")

        if 'flags' in sgcc:
            from sklearn.metrics import roc_auc_score
            sgcc_auc = roc_auc_score(sgcc['flags'], sgcc['oof_proba_meta'])
            oedi_auc = roc_auc_score(oedi['y'], oedi['oof_proba_meta'])
            print(f"  {'AUC':<15s}  {sgcc_auc:>10.4f}  {oedi_auc:>10.4f}")

        print(f"\n  Pipeline completed successfully!")
        print(f"  Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
