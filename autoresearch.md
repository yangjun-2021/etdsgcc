# Autoresearch: SGCC F1 Optimization

## Objective
Maximize the SGCC electricity theft detection F1 score on the **original (uncleaned) labels**. The previous baseline of F1=0.9557 was inflated because the evaluation used `flags` from `sgcc_expert_a.npz`, which are identical to the consensus-cleaned labels in `cleaned_labels_v1.npz` (744 labels were flipped relative to the raw SGCC labels). On the original labels, the honest baseline is F1=0.8657. The target is to push this above 0.90 without retraining deep experts on CPU.

## Metrics
- **Primary**: F1 score on SGCC original labels (higher is better)
- **Secondary**: AUC, Recall, Precision, training/evaluation time

## How to Run
`./autoresearch.sh` — outputs `METRIC f1=<number>` lines.

The benchmark runs `python run_meta_v2.py --dataset sgcc --label-source original`, which loads cached expert OOFs and runs ImprovedMetaLearner v2 against the original (uncleaned) labels.

To compare against the inflated cleaned-label metric:
```bash
python run_meta_v2.py --dataset sgcc --label-source cleaned
```

## Files in Scope
- `src/training/meta_learner_v2.py` — main meta-learner to optimize
- `run_meta_v2.py` — runner script
- `run_pipeline.py` — pipeline integration
- `output/sgcc_*.npz` — cached OOFs and labels (read-only)

## Off Limits
- Do not modify cached OOF files in `output/`.
- Do not modify raw data in `data/`.
- Do not modify Expert A/B training code unless explicitly testing a new expert.

## Constraints
- Each experiment must complete in <10 minutes.
- Keep changes minimal and focused on the meta-learner ensemble.
- All experiments are evaluated on the **original** SGCC labels (`cleaned_labels_v1.npz['y_orig']`).
- Retraining deep experts (TCN/AMST/Informer/Patch Transformer) is out of scope without GPU.

## What's Been Tried
- Baseline ImprovedMetaLearner v2 on **cleaned labels**: F1=0.9557 (AUC=0.9992, Rec=0.9293, Prec=0.9836).
- **Diagnosis**: `sgcc_expert_a.npz['flags']` is identical to `cleaned_labels_v1.npz['y_clean']`; the 0.95+ F1 was inflated by circular label cleaning.
- Re-baselined on **original labels**: F1=0.8657 (AUC=0.9874, Rec=0.8382, Prec=0.8951).
- Tested meta-learner variations (top_k, corr_threshold, MLP, weighted learning, pseudo-labels, hard-negative rescue): none improved over 0.8657.
- Retrained GBDT Expert A on original labels: F1=0.4726 — too weak to complement the cached strong OOFs.
- Ran GPU experiments on original labels:
  - Co-teaching TCN: F1=0.8536
  - AMST co-teaching: F1=0.8453
  - Strong GBDT prior on original labels: F1=0.8577
- **Final conclusion**: none of the original-label OOFs could beat the cleaned-label ensemble ceiling of F1=0.8657 on original labels. The honest performance ceiling for the current OOF pool is **F1=0.8657**. See `experiments/cleaned_vs_original_label_report.md`.

## Hyperparameter Search Space
- `top_k`: number of top-F1 OOFs to consider for ensemble [20, 30, 50, 100]
- `max_size`: maximum ensemble size in greedy selection [5, 10, 15, 20]
- `n_candidates`: candidates evaluated per greedy step [5, 8, 10, 15]
- `corr_threshold`: correlation pruning threshold [0.999, 0.9995, 0.9999, 1.0]
- `threshold_strategy`: best-F1, BDR, or FPR-constrained
- `meta_learners`: which second-level models to include
