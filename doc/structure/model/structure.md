# Model Structure

## 职责
- 承载 TSPN、桥接、浅层模型和训练相关实现。
- 处理 DAG 到模型的映射、训练输入输出、解释性产物。

## 为什么需要这一层
- 模型层的真相应该来自 `state.model_config_path` 和结构化 `model.*`，而不是从 `data_cfg` 反向推导。
- 把模型层和 LLM/provider/config glue 分开后，才能判断问题到底出在桥接、训练还是配置。

## 正式入口
- `src/model/`
- `src/model/explainable/*`
- `src/agents/deep_model_train_agent.py`（训练入口代理）

## 输入/输出边界
- 输入：canonical state、resolved model config、dataset 或 root split 数据。
- 输出：训练结果、桥接质量、解释性工件、模型配置快照。
- 不负责：LLM provider 选择、case 运行目录规划、Hydra compose。

## 当前实现状态
- 已实现：`src/model.py` 已删除，`src/model/` 成为唯一归宿；训练入口已优先读取 `state.model_config_path` 和 `state.model_cfg`。
- 正在收口：`data_cfg` 中仍保留少量只读 fallback，以便兼容旧状态，但已退出主链真源。
- `v1.0` 目标但未全落地：更细的 `layers/*` 配置、更多终端模型分层、TSPN v2 级别的多输入复杂算子架构。

## 冗余与历史包袱
- 旧 `case1 -> data_cfg -> trainer` 重复传播 `model_profile`、`autofit_*`、`model_config_path`，已经被判定为结构冗余。
- TSPN 与 operator library 之间仍有阶段性 contract/profile 管理需求，不应继续藏在数据层字段里。

## 按 v1.0 的下一步
- 进一步削减 `data_cfg` fallback。
- 把模型变体、训练 profile、bridge contract 的配置表达继续结构化。
