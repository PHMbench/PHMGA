# 2_10 -> 2_16 同步日志

## 2026-02-17
- 同步来源：`doc/plan/2_10/guidebook.md`
- 同步目标：`doc/plan/2_16/guidebook_snapshot.md`
- 同步类型：文档快照 + 状态矩阵收口

### 差异摘要
1. 将 `2_16` 从“待执行清单”切换为“审计状态视图”。
2. 记录了本轮新增三项代码补修：
   - `src/agents/inquirer_agent.py`（Pearson/cosine 边界）
   - `src/phm_outer_graph.py`（`train_backend` 白名单）
   - `src/agents/dataset_preparer_agent.py`（父链循环保护）
3. 固定运行命令为 `conda run -n agent` 语义。
4. 保留可选依赖 warning（`nolds`, `graphviz`）为非阻塞项。

## 2026-02-17（校准补录）
- 同步来源：`doc/plan/2_10/guidebook.md`
- 同步目标：`doc/plan/2_16/guidebook_snapshot.md`
- 同步类型：快照一致性校准（2_16 与 2_10 对齐）

### 差异摘要
1. 补齐 `B5.1 RM101 验收运行示例` 段落。
2. 补齐 `B6` 产物映射中的 `model_config.resolved.yaml` 与 `preflight_report.json`。
3. 保持正文内容与 `doc/plan/2_10/guidebook.md` 一致，仅保留快照头部元信息差异。

## 2026-02-17（论文实验代码补录）
- 同步来源：`doc/plan/2_10/guidebook.md`
- 同步目标：`doc/plan/2_16/guidebook_snapshot.md`
- 同步类型：实验执行代码与缺口清单补录

### 差异摘要
1. 新增 `D. 论文实验一键代码`（矩阵运行与结果汇总脚本）。
2. 新增 `E. 你还需要补齐的内容`（Ottawa 元数据标识、在线网络、Baseline 对照、统计检验）。
