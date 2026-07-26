"""
Legacy pipeline entry point. Use run_pipeline.py for the unified entry point.

This file is kept for backward compatibility.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SEED
from src.utils.utils import seed_everything, evaluate_binary


def run_sgcc_pipeline(use_advanced_features=True):
    print("\n" + "=" * 70)
    print("  SGCC DATASET PIPELINE")
    print("=" * 70)
    start_time = time.time()

    print("\n[Step 1/6] Preprocessing SGCC...")
    from src.data.preprocess_sgcc import preprocess_sgcc
    X_seq, stat_features, flags, impute_mask = preprocess_sgcc(
        use_advanced_features=use_advanced_features
    )

    print("\n[Step 2/6] Training Expert A (GBDT) on SGCC...")
    from src.training.expert_a import ExpertATrainer
    trainer_a = ExpertATrainer(dataset='sgcc')
    oof_proba_a, leaf_indices, _ = trainer_a.train(stat_features, flags, impute_mask=impute_mask)

    print("\n[Step 3/6] Training Expert B (TCN+Leaf) on SGCC...")
    from src.training.expert_b import ExpertBTrainer
    trainer_b = ExpertBTrainer(dataset='sgcc')
    oof_proba_b = trainer_b.train(X_seq, stat_features, flags, impute_mask=impute_mask,
                                   oof_proba_a=oof_proba_a, leaf_indices=leaf_indices)

    print("\n[Step 4/6] Training Meta-Learner on SGCC...")
    from src.training.meta_learner import MetaLearner
    meta_learner = MetaLearner(dataset='sgcc')
    results = meta_learner.train(stat_features, flags, impute_mask=impute_mask,
                                  oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b)

    print("\n[Step 5/6] Evaluating SGCC...")
    from src.evaluation.evaluate import evaluate_dataset
    evaluate_dataset('sgcc', results, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n  SGCC Pipeline completed in {elapsed/60:.1f} minutes")
    return results


def run_oedi_pipeline():
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
                                  oof_proba_a=oof_proba_a, oof_proba_b=oof_proba_b)

    print("\n[Step 5/6] Evaluating OEDI...")
    from src.evaluation.evaluate import evaluate_dataset
    evaluate_dataset('oedi', results, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n  OEDI Pipeline completed in {elapsed/60:.1f} minutes")
    return results


if __name__ == '__main__':
    seed_everything(SEED)

    dataset = sys.argv[1] if len(sys.argv) > 1 else 'both'
    use_advanced = '--no-advanced' not in sys.argv

    if dataset in ('sgcc', 'both'):
        sgcc_results = run_sgcc_pipeline(use_advanced_features=use_advanced)

    if dataset in ('oedi', 'both'):
        oedi_results = run_oedi_pipeline()

    if dataset == 'both':
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
