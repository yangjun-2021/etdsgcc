# Autoresearch Ideas Backlog

Ideas extracted from the reference folder that are **not yet tried** in this session. Prune and pick items as the loop continues.

## Ensemble & Stacking
- [ ] **Weighted average with learned per-OOF weights** via coordinate ascent on F1 (extend `_optimize_weighted_blend` to more passes/finer grid).
- [ ] **Stacking with a small MLP** (CPU-trainable) on the selected OOF matrix.
- [ ] **Entropy-weighted stacking** — weight each OOF by its calibration quality (ECE).
- [ ] **Diversity-weighted selection** — add OOFs that maximize pairwise disagreement, not just F1.

## Threshold & Decision Rules
- [ ] **Per-fold threshold optimization** instead of a single global threshold.
- [ ] **Cost-sensitive threshold** minimizing `FP + λ·FN` for a chosen λ (literature often treats missed thieves as more costly).
- [ ] **Platt scaling / isotonic regression** on the final blend to improve calibration before thresholding.

## New OOF Signals (CPU-feasible)
- [ ] **Rescale and inject unsupervised anomaly scores** (Isolation Forest, LOF, k-NN) as an additional OOF channel; current `sgcc_expert_d_unsupervised.npz` signals are weak alone but may help a small ensemble.
- [ ] **Rule-based / missing-pattern OOF** from imputation mask and zero-run statistics.
- [ ] **Neighbor-theft-ratio OOF** using distribution-network topology if transformer/user grouping metadata is available.

## Data-Centric Ideas
- [ ] **Synthetic minority oversampling** (CT-GAN / WGAN-GP) to retrain deep experts; requires GPU or very small models.
- [ ] **Hard-negative mining** — identify samples where the best ensemble fails and train a dedicated rescue model.
- [ ] **Label-noise cleaning** — the strong single OOF may already encode a cleaned label set; compare with raw flags.

## Hyper-Parameter Optimization
- [ ] **Coordinate/grid search** over `top_k`, `corr_threshold`, `max_size`, `n_candidates` using a cheaper proxy metric.
- [ ] **LLM-guided hyper-parameter tuning** (reference: "An ensemble framework for low false positive rate electricity theft detection using large language model-guided hyper-parameter tuning").

## Notes
- Current best F1 = 0.9557, already exceeding the 90% target. Further gains likely require **new strong OOF signals** or **more expressive second-level models**, not just meta-learner tuning.
