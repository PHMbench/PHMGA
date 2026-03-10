# 2.22 并行执行看板（3 人）

## 固定分工

| Owner | Scope | Primary Command | Expected Artifact Root |
| --- | --- | --- | --- |
| Teammate-1 | M1 补齐 | `scripts/paper/run_m1_gemini25_full_matrix.sh --env agent` | `save/paper_matrix/m1_gemini25` |
| Teammate-2 | M2 失败恢复 | `run_combo.sh`（`ottawa__A2_no_prior`） | `save/paper_matrix/m2_gemini3` |
| Teammate-3 | M3 激活 | Gate-A1/A2 + `run_m3_glm47_full_matrix.sh --env agent` | `save/paper_matrix/m3_glm47` |

## 任务记录字段（固定）

| Owner | Scope | Command | Artifact | ErrorClass | NextAction | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Teammate-1 |  |  |  |  |  | TODO |
| Teammate-2 |  |  |  |  |  | TODO |
| Teammate-3 |  |  |  |  |  | TODO |

## ErrorClass 取值（统一）

- `ok`
- `rc137_resource`
- `403_permission`
- `shape_divisibility`
- `preflight_failed`
- `blocked`

## 每人最小交付

1. `manifest.jsonl`
2. `manifest_dedup.jsonl`（若目录已执行）
3. 至少 1 份 `run.log`
4. 失败时必须附：
   - `error_class`
   - `evidence_path`
   - `next_action`
