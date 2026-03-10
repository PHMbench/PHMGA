# PHMGA 论文中期分析方法（Interim，3 模型）

本文件定义“当前可复现子集”的论文写作口径，确保结果可引用且不夸大结论。

## 1) 输入与最小产物

必需输入：
- `manifest_dedup.jsonl`
- 成功组合对应的 `metrics.json`
- `dataset_manifest.json`
- `config_resolve.json`
- `explain/operator_importance.json`（若存在）

建议先生成标准产物：
```bash
conda run -n agent python scripts/paper/collect_matrix_results.py \
  --manifest <manifest_dedup.jsonl> \
  --output-dir <out_dir>

conda run -n agent python scripts/paper/generate_analysis_draft.py \
  --manifest <manifest_dedup.jsonl> \
  --output-dir <out_dir>
```

## 2) Interim 写作规范（必须）

1. 先报告覆盖率，再报告性能。  
2. 结论用语必须是 `preliminary` / `interim`。  
3. 失败组合（`status!=ok`）单列，不参与均值比较。  
4. 仅在相同数据切分协议下比较模型。  

推荐覆盖率句式：
- “Current reproducible coverage is `M2: 5/6`, while `M1/M3` remain incomplete due to runtime or access constraints.”

## 3) 主结果组织（跨 LLM / 跨数据 / 跨消融）

主表维度：`dataset × llm × ablation`  
核心指标：`val_acc`, `test_acc`, `val_macro_f1`, `test_macro_f1`

处理规则：
1. 先从 `paper_main_results.csv` 读取成功行。  
2. 对每个 `(dataset, llm)` 保留 `A0/A1/A2` 三列（缺失留空 `not available`）。  
3. 单独追加失败原因统计表。  

## 4) 消融解读规范（A0-A1, A0-A2）

必算差值：
- `Δ_reflect = A0_full - A1_no_reflect`
- `Δ_prior = A0_full - A2_no_prior`

报告方式：
- 每个 `(dataset, llm)` 单独给差值。  
- 汇总“正增益组合占比”。  

中期结论模板：
- “Within the current subset, A0 outperforms A1 on RM101, indicating preliminary gains from the reflection loop.”
- “A0 vs A2 remains mixed and requires full-matrix confirmation.”

## 5) 可解释性与桥接质量

重点证据：
- `operator_importance.json`
- `config_resolve.json.config_source_mode`
- `config_resolve.json.bridge_quality`

桥接质量建议阈值：
- `effective_ops_ratio >= 0.40`：可接受  
- `0.20 ~ 0.40`：需关注 identity 稀释  
- `< 0.20`：建议回退 builder 重建 DAG  

## 6) 失败归因流程

失败分类：
1. `preflight_failed`（配置/路径/provider）  
2. `failed` + `return_code=137`（资源层）  
3. `failed` + `403/401`（权限/联通层）  
4. 产物缺失（训练完成但汇总失败）  

每类必须输出：
- count
- top-3 error message
- fix action

## 7) 可直接粘贴的论文段落模板

### 7.1 主结果段（Interim）
“We report interim results on the current reproducible subset. Among three active LLM backends, only the M2 setting achieved near-complete coverage (`5/6`), while M1/M3 remain incomplete due to runtime termination or model-access constraints. On Ottawa, M2 reaches approximately `0.91` test accuracy. On RM101, performance is lower (`~0.52–0.62` test accuracy), indicating a harder cross-condition diagnosis setting.”

### 7.2 消融段（Interim）
“Within the RM101 subset under M2, A0 improves over A1, suggesting positive contribution from the reflection loop. A0 and A2 show mixed behavior (A2 higher on validation but slightly lower on test), so the prior-initialization effect remains preliminary and requires full-matrix confirmation.”

### 7.3 局限性段（必须）
“These findings are interim because full 18-combo coverage is not yet complete. Therefore, we do not draw final cross-LLM ranking conclusions at this stage.”

## 8) 表格模板（可直接用于论文）

### Table X. Interim Results (Current Reproducible Subset)
| Dataset | LLM | Ablation | Val Acc | Test Acc | Val Macro-F1 | Test Macro-F1 | Coverage Note |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Table Y. Ablation Gains on RM101 (Interim)
| LLM | A0 Test Acc | A1 Test Acc | A2 Test Acc | Δ(A0-A1) | Δ(A0-A2) | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |

### Table Z. Coverage and Failure Summary
| LLM | Success / Total | Main Failure | Status |
| --- | --- | --- | --- |
