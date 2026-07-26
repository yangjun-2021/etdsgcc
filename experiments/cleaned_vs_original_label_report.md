# SGCC 窃电检测：清洗标签 vs 原始标签评估报告

## 1. 背景

在项目早期优化中，元学习器在 SGCC 上取得了 **F1=0.9557** 的极高结果。后续调查发现，该结果使用了 `output/sgcc_expert_a.npz['flags']` 作为标签，而这些标签与 `cleaned_labels_v1.npz['y_clean']` 完全一致，是**通过模型共识修正后的清洗标签**，并非 SGCC 原始标签。

本报告对比两种标签下的模型表现，说明 0.9557 的不可复现性，并给出当前诚实的性能天花板。

## 2. 标签差异

| 指标 | 清洗标签 `y_clean` | 原始标签 `y_orig` |
|------|-------------------|------------------|
| 样本数 | 42,372 | 42,372 |
| 正例数 | 3,803 | 3,615 |
| 正例比例 | 8.98% | 8.53% |
| 与另一方不一致数 | 744 | 744 |

不一致样本中：
- 466 个被清洗标签标记为正、原始标签为负；
- 278 个被清洗标签标记为负、原始标签为正。

这意味着清洗标签把部分模型预测结果“写回”了真实标签，形成循环验证，导致所有在清洗标签上训练的模型指标被系统性抬高。

## 3. 关键模型表现对比

| 模型 / OOF | 清洗标签 F1 | 原始标签 F1 | 下降 |
|-----------|------------|------------|------|
| `sgcc-meta-v2-final` | **0.9557** | 0.8559 | -0.0998 |
| `meta-final-cleaned` | 0.9548 | 0.8517 | -0.1031 |
| `meta-cleaned` | 0.9497 | 0.8489 | -0.1008 |
| Expert A | 0.6696 | 0.6209 | -0.0487 |
| Co-teaching TCN (原始标签训练) | — | 0.8536 | — |
| AMST co-teaching (原始标签训练) | — | 0.8453 | — |
| Strong GBDT prior (原始标签训练) | — | 0.8577 | — |
| **ImprovedMetaLearner v2 最佳** | **0.9557** | **0.8657** | **-0.0900** |

## 4. Autoresearch 原始标签实验摘要

在切换到原始标签后，共进行 11 轮实验（runs 6–16），尝试了：

- 超参搜索（top_k、corr_threshold、max_size、n_candidates）
- MLP meta-learner
- 标签噪声鲁棒 meta-learning（加权、伪标签）
- hard-negative rescue OOF
- GBDT Expert A 原始标签重训
- Co-teaching TCN 原始标签训练
- AMST-Net co-teaching 原始标签训练
- 原始标签强 GBDT prior

**没有任何一轮能够突破 baseline F1=0.8657。**

## 5. 为什么 0.90 在当前设置下不可达？

1. **顶级 OOF 全部来自清洗标签训练**：`final-blend-best`、`sgcc-meta-v2-final` 等最强信号已经“见过”修正后的标签分布，在原始标签上达到约 0.866 已是其泛化上限。
2. **原始标签训练的模型明显更弱**：在原始标签上从零训练的 Co-teaching TCN / AMST / GBDT prior 分别只有 0.854 / 0.845 / 0.858，无法提供互补信息来超越清洗标签集成。
3. **没有独立测试集**：所有评估都是全量 OOF，进一步说明 0.8657 是这个 OOF 池的拟合天花板。

## 6. 建议

### 立即可用的结论
- 将 **原始标签 F1=0.8657** 作为 SGCC 当前最诚实、最可靠的主要指标。
- 若论文/报告需要引用，应明确区分：
  - 清洗标签结果（0.9557）仅用于方法内部消融；
  - 原始标签结果（0.8657）用于与外部文献对比。

### 若仍要冲击 0.90
需要在原始标签上**重新设计并训练一整套更强的基础模型**，而非在现有 OOF 上做集成。可能方向：
- 更大数据量 / 外部数据；
- 更强的时序基础模型（Mamba、Longformer、时序 Foundation Model）；
- 更先进的标签噪声学习（DivideMix、Co-teaching++、PU Learning）；
- 引入用户拓扑、变压器关系等图信息。

## 7. 相关文件

| 文件 | 说明 |
|------|------|
| `autoresearch.jsonl` | 全部实验记录（segment 0：清洗标签；segment 1：原始标签） |
| `autoresearch-dashboard.md` | 原始标签看板 |
| `experiments/worklog.md` | 每轮实验详细日志 |
| `experiments/original_label_retrain_plan.md` | 原始标签重训方案 |
| `experiments/train_coteaching_original_labels.py` | Co-teaching TCN 原始标签脚本 |
| `experiments/train_amst_3ch_original_coteaching.py` | AMST 原始标签脚本 |
| `experiments/train_informer_3ch_original.py` | Informer 原始标签脚本 |
| `experiments/train_patch_transformer_raw_3ch_original.py` | Patch Transformer 原始标签脚本 |
| `experiments/build_stronger_prior_original.py` | 原始标签强 GBDT prior |

## 8. 分支状态

- 分支：`autoresearch/sgcc-f1-20260713`
- 最新 commit：TBD after final commit
- 结论：原始标签 F1 天花板 **0.8657**，0.90 需从零重训更强 pipeline。
