# Supplementary Materials

Open-access data accompanying the BSC-ETD electricity theft detection framework paper.
All data corresponds to the SGCC dataset (~42k users × 1035 days).

## Directory Structure

```
supplementary/
├── oof_pool/          # Out-of-fold prediction probability feature pool
│   ├── sgcc_expert_a.npz                — Expert A (GBDT ensemble) OOF, original labels
│   ├── sgcc_expert_a_original.npz       — Expert A OOF, trained on original labels only
│   ├── sgcc_expert_b.npz                — Expert B (TCN+Leaf) OOF
│   ├── final_blend_best_oof.npz         — Final ensemble OOF (best F1 on original labels)
│   ├── oof_summary.csv                  — Table S2: 46-channel OOF signal quality ranking (F1, AUC, Recall, Precision, Threshold)
│   ├── bundled_oofs.csv                 — Full OOF pool with ExpertA/B features + 30+ ensemble OOF signals
│   ├── clean_baseline_oofs.csv          — Clean-baseline OOF pool for comparative experiments
│   └── quick_bundled_oof_eval.csv       — Quick evaluation summary of all bundled OOFs
│
├── ablation/          # Ablation experiment data
│   ├── ablation_metrics.csv             — Per-stage ablation metrics (F1, AUC, Recall, Precision, TP/FP/FN)
│   ├── ablation_summary.csv             — Table S3: Ablation summary with F1/AUC means ± std
│   ├── per_fold_metrics.csv             — Table S4: Per-fold metrics for fold-aware validation
│   ├── significance_tests.csv           — Uncorrected paired t-tests between stages
│   ├── significance_tests_corrected.csv — Table S5: Nadeau-Bengio corrected t-tests with Cohen's d
│   ├── ablation_oofs.npz                — Raw OOF predictions for all ablation stages
│   └── spc_stratified.csv               — SPC stratified analysis (Appendix B.3)
│
├── labels/            # Label cleaning data
│   ├── cleaned_labels_v1.npz            — Original labels (y_orig) + self-consensus cleaned (y_clean, 744 flips)
│   └── cleaned_labels_v3.npz            — Committee-cleaned labels (673 flips, untainted original-label voters)
│
└── tables/            # Supplementary tables
    ├── table_main_results.csv            — Table S1: Main results (46 models, F1/AUC/Recall/Precision/Threshold)
    ├── table_label_noise_analysis_latest.csv — Label noise sensitivity analysis
    ├── table_subgroup_performance_latest.csv — Subgroup-stratified performance
    └── recall_f1_frontier_fast.csv       — Table S6: Threshold-precision complete trade-off
```

## Reproduction

Load any .npz file with:
```python
import numpy as np
data = np.load('oof_pool/sgcc_expert_a.npz')
print(data.files)           # ['oof_proba']
oof = data['oof_proba']     # shape: (42372,) — one probability per user
```

Load labels with:
```python
labels = np.load('labels/cleaned_labels_v1.npz')
y_orig  = labels['y_orig']   # original SGCC labels
y_clean = labels['y_clean']  # consensus-cleaned (self-refined, inflated metrics)
```

## Notes

- The paper headline result on original labels is F1=0.8666 (usage-quintile subgroup thresholds, pooled 5-fold OOF); the single-global-threshold counterpart is F1=0.8657. F1≈0.95 on cleaned labels is inflated by circular label refinement.
- All 5-fold CV results use SEED=42 and stratified splits.
- The `ablation_oofs.npz` contains OOF predictions from all 6 ablation stages (A0–A6) for direct comparison.
