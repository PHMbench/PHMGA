# States Structure

## 职责
- 定义 `PHMState`、DAG 节点和 builder/executor/train 共享状态合同。
- 维持 root 数据、labels、运行时配置与 agent 产物之间的一致性。

## 为什么需要这一层
- state 是外层 graph 的共享内存。没有独立状态层，任何 agent 都可能私自改字段名和语义。
- 这次 `ref/tst -> train/val/test` 迁移，本质上就是状态合同重构，不是简单重命名。

## 正式入口
- `src/states/phm_states.py`
- `src/states/base.py`
- `src/states/research_state.py`

## 输入/输出边界
- 输入：agent 更新、state load/save、DAG 执行结果。
- 输出：规范化 state、节点对象和兼容期字段映射。
- 不负责：graph 调度、prompt 渲染、训练算法。

## 当前实现状态
- 已实现：root `results` 和 label 合同已规范到 `train/val/test`；legacy `labels_ref/labels_tst` 仅作兼容别名保留。
- 正在收口：`reference_signal/test_signal` 仍保留为历史 root anchor，避免本轮扩大状态改动面。
- `v1.0` 目标但未全落地：更细的 builder/executor/train state 拆分和更轻的公共基类。

## 冗余与历史包袱
- 旧状态把 split 语义写成 `ref/tst`，并且不同 agent 对含义解释不同，导致训练、报告和 dataset 构造容易错位。

## 按 v1.0 的下一步
- 继续把 phase-specific state 从 `PHMState` 中拆分出去。
- 逐步减少兼容别名的可见性，直到可以删除。
