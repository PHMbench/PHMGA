# Tools Structure

## 职责
- 提供算子 schema、注册表和图外可复用 helper。
- 作为 DAG 节点执行时引用的稳定能力层。

## 为什么需要这一层
- 算子实现必须独立于 agent，否则 planner/executor 一改，算子定义就会被一起牵动。
- 只有把 tools 独立出来，closed-world contract 和 operator inventory 才能持续维护。

## 正式入口
- `src/tools/*`
- `OP_REGISTRY` 及其相关 schema/registry 文件

## 输入/输出边界
- 输入：算子参数和上游节点结果。
- 输出：变换后的结果、特征或聚合产物。
- 不负责：graph 调度、LLM 调用、case 运行时管理。

## 当前实现状态
- 已实现：tools 仍是算子定义和图外 helper 的主要归宿。
- 正在收口：部分历史 helper 仍需和 agent/model 边界再切一刀。
- `v1.0` 目标但未全落地：更系统的 operator contracts、inventory 和多输入复杂算子分层。

## 冗余与历史包袱
- 历史上复杂算子支持程度不一致，导致 bridge 需要 proxy/drop/fallback 策略，这部分需要通过 contract 文档持续显式管理。

## 按 v1.0 的下一步
- 继续把 closed-world operator contract 和算子 backlog 从实现细节提升到正式维护对象。
