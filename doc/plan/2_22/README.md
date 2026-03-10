# PHMGA 2.22 失败实验恢复作战包

## 目标

本目录用于收口历史失败实验，给外部 Agent 一套可直接执行的恢复流程，重点覆盖：

1. 失败基线固化（含证据路径）
2. 分流恢复策略（`rc=137` / `403` / 结构错误）
3. 3 人并行执行分工与验收
4. 论文中期结果更新口径（`preliminary/interim`）

> 范围：优先不改业务代码，先用现有脚本与配置完成最大恢复。

## 文档索引

1. `01_failure_baseline.md`：最新失败快照与证据定位
2. `02_recovery_strategy.md`：三类故障固定恢复策略
3. `03_execution_runbook.md`：逐步可复制执行命令
4. `04_teammate_board.md`：3 人并行看板
5. `05_interim_paper_update.md`：恢复后论文中期结果更新模板
6. `legacy_glm45_archive.md`：GLM-4.5 与历史目录归档证据（仅归档）
7. `../2_23/README.md`：S3 最小代码修复分支（已触发）

## 执行顺序（必须）

1. 先读 `01_failure_baseline.md` 锁定当前状态
2. 按 `03_execution_runbook.md` 跑 Gate 与恢复主线
3. 失败时按 `02_recovery_strategy.md` 分流
4. 并行执行按 `04_teammate_board.md`
5. 产物更新按 `05_interim_paper_update.md`

## Definition of Done

1. 3 个 active 目录均有 `manifest_dedup.jsonl`
2. 至少完成：
   - `m2` 失败组合重跑 1 次
   - `m1` 全矩阵补跑 1 次
   - `m3_glm47` 完成 Gate-A1/A2 + pilot
3. `save/paper_matrix/all_llm_manifest.jsonl` 仅包含 `m1/m2/m3_glm47`
4. 输出更新后的 `paper_main_results.csv` 与 `analysis_draft.md`
5. legacy（GLM-4.5 与历史目录）仅保留归档证据，不混入 active 执行
