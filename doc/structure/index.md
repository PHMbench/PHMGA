# PHMGA Structure Index

## 模块地图

| 模块 | 负责什么 | 为什么必须独立成层 | 与相邻层的边界 | 文档 |
|---|---|---|---|---|
| `config` | 组合 Hydra 配置、规范化运行时配置、落盘 resolved config | 不把配置逻辑散落到 case、agent、脚本里 | 向 `cases` / `graph` / `llm` / `model` 提供已规范化配置；不直接执行业务 | [config/structure.md](config/structure.md) |
| `data` | 约束数据来源、metadata 校验、split 规范 | 数据选择和训练逻辑要分离，否则 `ref/test` 旧语义会反复回流 | 给 `cases` / `preflight` / `trainer` 提供 canonical `data.selection`；不承载模型配置 | [data/structure.md](data/structure.md) |
| `graph` | 注册并选择 builder/executor graph | workflow 选择不应散在 `main.py` 和 case 文件中 | 对上给 `cases` 一个 graph 名称接口；对下调用 agents | [graph/structure.md](graph/structure.md) |
| `llm` | 统一 OpenRouter 配置、校验和 client 构造 | 避免 provider 判断继续留在 `model`、`configuration.py`、脚本环境变量拼接里 | 对上暴露已解析 LLM 配置/客户端；不承载训练逻辑 | [llm/structure.md](llm/structure.md) |
| `agents` | 工作流节点逻辑：plan/execute/reflect/report/train 等 | agent 是业务动作层，不能和 graph、case、tool 混写 | 由 `graph` 调度，读写 `states`，调用 `tools` / `model` | [agents/structure.md](agents/structure.md) |
| `cases` | case runner、运行目录、实验级装配 | 运行入口需要和 agent/tool 逻辑解耦，避免 `main.py` 直接 import case | 从 `config` 取配置，选 `graph`，准备 runtime artifacts | [cases/structure.md](cases/structure.md) |
| `model` | TSPN、桥接、训练、浅层模型终端 | 模型实现和 LLM/provider、case glue 分离后才可维护 | 接收已规范化 state/config；不负责配置来源和 LLM 选择 | [model/structure.md](model/structure.md) |
| `tools` | 算子 schema、注册表、图外 helper | 算子定义需要稳定边界，不能让 agent 直接内嵌算子实现 | 被 `agents` 和 `model` 调用；不持有 workflow 状态 | [tools/structure.md](tools/structure.md) |
| `schemas` | 结构化协议对象和边界校验 | 没有 schema，配置/state/report 合同会漂移 | 被 `config` / `states` / `model` / `tools` 共享 | [schemas/structure.md](schemas/structure.md) |
| `prompts` | prompt 模板和渲染 helper | prompt 不是业务流程，不应继续散在 agent 实现细节里 | 被 `agents` / `llm` 调用；不读 env，不直接读 yaml | [prompts/structure.md](prompts/structure.md) |
| `states` | PHMState、DAG 节点、split/label 合同 | 没有统一状态层，builder/executor/train 的数据契约会失控 | 被 `agents` / `model` / `utils` 共享；不负责执行 | [states/structure.md](states/structure.md) |
| `utils` | 无副作用通用 helper、序列化、路径、preflight | 防止 `utils` 回潮成大杂烩，所以必须限制职责 | 只能承载通用 helper；业务逻辑应迁出 | [utils/structure.md](utils/structure.md) |
| `docs` | 结构说明、删减清单、目标态差距记录 | README 只讲怎么用，不足以支撑重构和删减决策 | 为所有模块提供“为什么”和“何时可删”的解释层 | [docs/structure.md](docs/structure.md) |
| `del` | 冗余和删除执行清单 | 没有清单就无法判定哪些兼容层还能留、哪些必须删 | 连接结构文档、测试和删除动作 | [del/00_redundancy_inventory.md](del/00_redundancy_inventory.md) |

## 当前重点边界
- `config -> cases`: 只传已规范化配置，尤其是 canonical `data.selection` 和 `model.*`。
- `cases -> graph`: 只选择 graph，不内嵌 graph 实现。
- `agents -> model`: 只调训练/桥接接口，不再从 `data_cfg` 反向拼模型配置。
- `states -> trainer`: fixed-ids 主链统一使用 `train/val/test`，`ref/test` 只允许停留在 resolver 兼容层。

## 现状摘要
- 已实现：Hydra compose、OpenRouter 单入口、registry 主链、canonical split 合同、部分冗余删除。
- 正在收口：`case1` 偏重、部分 README/注释还残留旧 split 命名、`src/phm_outer_graph.py` 仍需最终删除前的瘦身。
- `v1.0` 目标但未全落地：`llm/roles`、`llm/routing`、`llm/structured_outputs`、`data/metadata_schema`、`data/selection_presets`、`graphs/executor_ml`、`graphs/executor_automl`、`graphs/report_only`、`layers/*`、`experiments/*`、`profiles/*`。
