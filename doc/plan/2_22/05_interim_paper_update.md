# 2.22 论文中期结果更新指南（恢复后）

## 目标

把恢复后的 active matrix 结果直接整理成论文中期可用材料，保持 `preliminary/interim` 口径。

---

## 1) 更新输入

恢复完成后，使用以下文件：

- `save/paper_matrix/all_llm_manifest.jsonl`（仅 m1/m2/m3_glm47）
- `save/paper_matrix/paper_main_results.csv`
- `save/paper_matrix/analysis_draft.md`

---

## 2) 覆盖率表（必须先写）

| LLM | Success / 6 | Status | Main Failure |
| --- | --- | --- | --- |
| M1 (`m1_gemini25`) |  |  |  |
| M2 (`m2_gemini3`) |  |  |  |
| M3 (`m3_glm47`) |  |  |  |
| **Total** |  / 18 |  |  |

写作要求：
1. 先写覆盖率，再写性能。  
2. 对未完成模型，必须写阻塞原因。  

---

## 3) Ottawa / RM101 指标更新

建议按以下区块更新：

### Ottawa
- `A0/A1/A2`（按模型分行）
- 指标：`val_acc`, `test_acc`, `val_macro_f1`, `test_macro_f1`

### RM101
- `A0/A1/A2`（按模型分行）
- 同上四项指标

> 若某组合无结果，写 `N/A (not completed)`，不得填充估计值。

---

## 4) 失败保留陈述模板（可直接引用）

### 中期主结论模板

“We report interim results on the current recoverable subset of the 3-LLM matrix. Coverage and performance are both reported. Any missing combinations are retained as pending failures and are not imputed.”

### 失败保留模板（权限）

“Some GLM-4.7 combinations remain blocked by model-access constraints (`403`), therefore final cross-model ranking is not claimed at this stage.”

### 失败保留模板（资源）

“A subset of combinations was interrupted by resource-level termination (`rc=137`), and was retried under reduced training profiles before inclusion.”

---

## 5) 可直接提交的中期表格模板

### Table X — Interim Main Results (Recovered Active Matrix)
| Dataset | LLM | Ablation | Val Acc | Test Acc | Val Macro-F1 | Test Macro-F1 | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Table Y — Coverage and Blocking Summary
| LLM | Completed | Failed | Missing | Blocking Type |
| --- | --- | --- | --- | --- |

### Table Z — Ablation Delta on RM101
| LLM | A0 Test Acc | A1 Test Acc | A2 Test Acc | Δ(A0-A1) | Δ(A0-A2) |
| --- | --- | --- | --- | --- | --- |

---

## 6) 输出验收

恢复收口后，论文中期材料至少包含：

1. 覆盖率表（3 LLM）
2. Ottawa/RM101 指标表
3. 失败保留陈述（权限/资源）
4. 明确的 `interim` 语气（不做最终结论）
