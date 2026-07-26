# 原始标签 + 标签噪声鲁棒重训方案

## 目标
在 **原始标签 `cleaned_labels_v1.npz['y_orig']`** 上训练一套新的强专家 OOF，使 `ImprovedMetaLearner v2` 在原始标签评估下突破 **F1=0.90**。

## 为什么之前的 OOF 池到不了 0.90？

当前缓存的顶级 OOF（`final-blend-best`、`sgcc-meta-v2-final`、`gated-rescue-blend` 等）都是在 **清洗标签 `y_clean`** 上训练或调优的。这些 OOF 在原始标签上的天花板约为 **F1≈0.866**，因为它们的学习目标本身已经被清洗标签“引导”过，无法再从原始标签中挖掘出被清洗掉的 744 个样本信息。

要在原始标签上达到 0.90，必须让至少一部分基础专家直接在原始标签上训练，并使用 **标签噪声鲁棒机制** 来缓解那 744 个噪声标签的影响。

## 推荐方法：Co-Teaching + Strong GBDT Prior

项目里已经有 `src/training/coteaching.py` 和 `experiments/train_coteaching_raw_3ch_cv.py`。该方法让两个网络互相监督，只保留每个 epoch 中 loss 较小的样本进行反向传播，从而自动过滤噪声标签。结合从原始标签训练的 GBDT prior，可以进一步提升稳定性。

### 方案步骤

1. **准备原始标签数据**
   - 复用已有的 `output/sgcc_preprocessed_raw_3ch.npz`（X_seq 和 impute_mask）。
   - 用 `cleaned_labels_v1.npz['y_orig']` 替换训练用的 `flags`。

2. **训练原始标签上的强 GBDT prior**
   - 复用 `experiments/retrain_expert_a_original.py` 的思路，但打开 advanced features（需要 GPU/AE 训练）或至少调优 GBDT 超参。
   - 保存为 `output/strong_gbdt_prior_original.npz`。
   - 这一步是为了给后续深度专家提供高质量的 prior，而不是为了直接提升 meta-learner。

3. **Co-Teaching TCN/AMST 训练（核心）**
   - 脚本：`experiments/train_coteaching_original_labels.py`。
   - 使用 `train_coteaching_cv(..., forget_rate=0.15, warmup_epochs=10, ...)`。
   - 输入：`X_seq`（raw 3ch）+ `oof_prior_original`。
   - 标签：`y_orig`。
   - 设备：必须 `cuda`；CPU 训练不可行。
   - 输出：`output/coteaching_original_oof.npz`。

4. **Robust AMST 训练**
   - 脚本：`experiments/train_amst_3ch_original_coteaching.py`。
   - 使用 `AMSTTrainer(use_coteaching=True, ...)`。
   - 同样使用 `y_orig` 和原始标签 prior。
   - 输出：`output/amst_3ch_original_oof.npz`。

5. **Robust Informer / Patch Transformer 训练**
   - 参考 `experiments/train_informer_3ch_strong_prior.py` 和 `train_patch_transformer_raw_3ch_oof.py`。
   - 关键改动：把训练标签替换为 `y_orig`，并加入 mixup + label smoothing。
   - 输出：`output/informer_original_oof.npz`、`output/patch_transformer_original_oof.npz`。

6. **生成新的 OOF 池并跑 autoresearch**
   - 新的 OOF 文件会被 `src/training/meta_learner_v2.py` 的 `_discover_all_npz_oofs` 自动发现。
   - 运行 `./autoresearch.sh`（已默认 `--label-source original`）。
   - 如果新 OOF 在原始标签上达到 F1≈0.88+，集成后有望突破 0.90。

## 预期资源与时间

- **GPU**：必需。每个 co-teaching / AMST / Informer 5-fold 训练在单张 RTX 4090/3090 级别显卡上约需 2–6 小时。
- **CPU**：预处理已经完成，不需要额外 CPU 时间。
- **总时间**：完整跑完 3–4 个新专家约需 1–2 天。

## 风险与备选

- 如果 co-teaching 过滤掉太多正样本，可能导致 recall 下降。可调整 `forget_rate`（0.1–0.2）和 `warmup_epochs`。
- 如果原始标签噪声比例被高估（实际上 744/42372≈1.76% 不算高），普通训练可能已经足够，co-teaching 收益有限。
- 备选：直接使用 `y_orig` 的普通训练（无 co-teaching）作为对照实验，比较是否有提升。

## 与本分支当前状态的关系

- 当前分支：`autoresearch/sgcc-f1-20260713`
- 当前最佳：原始标签 F1=0.8657
- 本方案是当前唯一 realistic 的 0.90+ 路径；其余元学习器/阈值/集成技巧均已尝试并 discard。
