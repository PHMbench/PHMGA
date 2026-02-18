# PHMGA 2.16 收口审计索引

本目录用于记录 `2.16` 批次修复的“现状证据”，不再作为待办清单。

## 当前状态（2026-02-17）
- 主链路状态：`Closed (Critical path)`
- 剩余风险：可选依赖告警（`graphviz`, `nolds`），不阻塞主流程
- 默认环境：`conda run -n agent` / `conda activate agent`
- guidebook 快照：已与 `doc/plan/2_10/guidebook.md` 完全对齐（仅保留快照头部差异）

## 文档索引
- `doc/plan/2_16/summary.md`：必要性判定与最终结论
- `doc/plan/2_16/detailed_fixes.md`：逐项状态（Closed/Needed/Deprecated）与代码锚点
- `doc/plan/2_16/statistics.md`：历史统计 + 当前闭环统计
- `doc/plan/2_16/guidebook_snapshot.md`：`doc/plan/2_10/guidebook.md` 同步快照
- `doc/plan/2_16/sync_log.md`：本次同步记录
- `doc/plan/2_16/rm101_run_evidence_2026-02-17.md`：RM101 实跑验收证据
