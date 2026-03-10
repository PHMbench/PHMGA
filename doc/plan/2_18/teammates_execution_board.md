# PHMGA 2.18 Teammates 执行看板（按 LLM 并行）

本看板是多人并行执行的单一事实来源（Single Source of Truth）。

## 1) 分工总览（3 模型）

| Owner | Scope | Primary Command | Target Output Root |
| --- | --- | --- | --- |
| Teammate-1 | M1 Gemini-2.5 | `scripts/paper/run_m1_gemini25_full_matrix.sh` | `save/paper_matrix/m1_gemini25` |
| Teammate-2 | M2 Gemini-3 | `scripts/paper/run_m2_gemini3_full_matrix.sh` | `save/paper_matrix/m2_gemini3` |
| Teammate-3 | M3 GLM-4.7 | `scripts/paper/run_m3_glm47_full_matrix.sh` | `save/paper_matrix/m3_glm47` |

## 2) 每位执行者固定流程

1. 先做 Gate-A（联通）+ Gate-B（pilot）  
2. 再跑全矩阵（6 组合）  
3. 去重：`manifest_dedup.jsonl`  
4. 失败重跑：`rerun_failed_from_manifest.sh`（最多 2 轮配置调整）  
5. 产出结果表：`collect_matrix_results.py` + `generate_analysis_draft.py`

## 3) 必交付产物（每位 teammate）

- `<output_root>/manifest.jsonl`
- `<output_root>/manifest_dedup.jsonl`
- `<output_root>/_resolved_cases/*.yaml`
- `<output_root>/_logs/*/preflight.log`
- `<output_root>/_logs/*/run.log`
- `<output_root>/paper_main_results.csv`
- `<output_root>/paper_main_results.md`
- `<output_root>/analysis_draft.md`

## 4) 失败交接协议（必须）

失败组合必须附以下字段后再交接：

| Field | Example |
| --- | --- |
| `combo` | `ottawa__m3_glm47__A0_full` |
| `error_class` | `403` / `137` / `preflight_failed` / `runtime_failed` |
| `next_action` | `switch to fast profile and rerun` |
| `evidence` | `<output_root>/_logs/<combo>/run.log` |

## 5) 当前基线状态（latest dedup）

| LLM | Success/Total | Blocking |
| --- | --- | --- |
| M1 | `0/1` | `rc=137` |
| M2 | `5/6` | `ottawa A2 rc=137` |
| M3 | `0/6` | `403 model access` |

## 6) 人力不足时的串行优先级

按顺序执行：`M2 -> M1 -> M3`。

## 7) 负责人填写区（复制后逐项填写）

| Owner | Gate-A | Gate-B | Full Matrix | Dedup | Rerun | Final Status |
| --- | --- | --- | --- | --- | --- | --- |
| Teammate-1 |  |  |  |  |  |  |
| Teammate-2 |  |  |  |  |  |  |
| Teammate-3 |  |  |  |  |  |  |
