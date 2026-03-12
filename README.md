# PHMGA

PHMGA 是一个面向论文复现与方法验证的工业时间序列研究仓库，不再承担旧版“通用多智能体平台”的定位。当前仓库只围绕一条清晰主链组织：

`problem -> protocol -> workflow -> dag -> operators -> bridge -> model -> training -> evaluation -> report`

## 当前定位

- 目标：用最小研究闭环支撑论文版 PHMGA，而不是维护历史平台兼容层。
- 前端：`states + prompts + agents` 生成结构先验 DAG。
- 中间：`dag + operators + bridge` 将结构先验固化为 validated DAG JSON，并编译到不同 graph path。
- 后端：`model + training + evaluation` 输出 graph-dependent artifacts 和最终报告。

## 三条 graph path

- `dag_only`
  - 生成 DAG JSON、结构图、节点边清单、方法说明和 `final_report.md`。
- `ml`
  - 将 DAG 编译成特征流水线，运行轻量 ML 基线，输出 `feature_pipeline.json`、`metrics.json`、`predictions.json`、`importance.json`。
- `torch`
  - 将 DAG 编译成可训练构建计划，运行最小训练后端，输出 `model_build_plan.json`、`training_curves.json`、`checkpoint.json`、`importance.json`、`metrics.json`。
  - 当前实现为 NumPy fallback 训练器，用于离线 smoke 和结构合同验证；它不是完整 PyTorch 训练栈。

## 当前仓库骨架

- `doc/structure/`
  - 论文版结构文档与删除账本，是当前设计权威说明。
- `config/`
  - 基础配置 `config/config.yaml`，再按 `data/`、`experiment/`、`model/` 分组；`config/runs/` 负责把某个数据配置与某条 graph path 组合成可直接运行的 config 文件。
- `scripts/preflight.py`
  - 环境、配置、协议和 graph path 预检入口。
- `scripts/run_case.py`
  - 单一运行入口，负责把同一份 validated DAG JSON 编译并执行到某条 graph path。
- `src/`
  - 最小研究核心，不保留旧 `main.py`、`src/tools`、`src/cases`、`src/graph` 等平台式入口。
- `tests/unit/` 与 `tests/smoke/`
  - 分别覆盖合同和最小端到端闭环。

## 配置组织

正式运行入口是一个完整配置文件。当前推荐直接使用 `config/runs/*.yaml`。

配置组织分为两层：

- `config/config.yaml`
  - 基础项目、LLM、artifact、模型参数。
- `config/data/*.yaml`
  - 数据协议与数据来源。
- `config/experiment/*.yaml`
  - graph path 选择。
- `config/runs/*.yaml`
  - 只负责选择一个 `data` 配置和一个 `graph_path`。

当前最重要的几个字段：

- `defaults.dataset`
  - 选择一个数据配置，例如 `rm101`、`ottawa`、`rm101_synth`。
- `defaults.graph_path`
  - 选择 `dag_only`、`ml`、`torch`。
- `llm`
  - 当前默认 `offline_stub`，用于无外部依赖的最小闭环。
- `model.ml`
  - 轻量 ML 路径的训练参数。
- `model.torch`
  - 当前最小 trainable path 的训练参数。

真实数据配置已经接入两套 PHM-Vibench 数据：

- `config/data/rm101.yaml` -> `RM_101_THU_GEARBOX`
- `config/data/ottawa.yaml` -> `RM_017_Ottawa19`

它们都通过 `metadata_path + h5_path` 直接驱动真实 loader。当前真实 H5 约定如下：

- metadata 主索引列使用真实文件列名，例如 `Sample_lenth`、`Channel`
- H5 key 使用字符串化 `Id`
- H5 样本 shape 支持 `(L, C, 1)` 或 `(L, C)`，内部统一转成 `(C, L)`
- H5 实测 shape 优先于 metadata；例如 `RM_101_THU_GEARBOX` 的 metadata 长度是 `768000`，而 H5 实测是 `767999`

synthetic 仍然保留，但只作为 smoke fallback，配置在：

- `config/data/rm101_synth.yaml`
- `config/data/ottawa_synth.yaml`

## 运行方式

```bash
python scripts/preflight.py --config config/runs/rm101_dag.yaml

python scripts/run_case.py \
  --config config/runs/rm101_dag.yaml \
  --output-dir artifacts/rm101_dag

python scripts/run_case.py \
  --config config/runs/ottawa_ml.yaml \
  --output-dir artifacts/ottawa_ml

python scripts/run_case.py \
  --config config/runs/rm101_synth_torch.yaml \
  --output-dir artifacts/rm101_synth_torch
```

## 测试方式

```bash
pytest -q
```

当前测试矩阵覆盖：

- canonical protocol 与 split/window 合同
- DAG JSON 校验与 operator catalog 合同
- bridge 对三条 graph path 的编译产物
- synthetic run configs 在三条 graph path 下的 smoke 闭环
- 真实 `RM_101_THU_GEARBOX` 与 `RM_017_Ottawa19` 的 protocol / `dag_only` 集成检查
- 入口文档与结构文档的基本一致性

## 阅读顺序

建议按以下顺序读仓库：

1. `doc/structure/README.md`
2. `doc/structure/index.md`
3. `doc/structure/00_problem_and_protocol.md`
4. `doc/structure/01_dag_and_operators.md`
5. `doc/structure/02_workflow_and_bridge.md`
6. `doc/structure/03_training_and_evaluation.md`
7. `doc/structure/04_rebuild_checklist.md`

## 给 AI coding agents 的通用约束

这一节是 `AGENTS.md`、`CLAUDE.md`、`GEMINI.md` 统一引用的通用事实源。

- 不要回流旧平台叙事，不要重新引入 `main.py`、`src/tools`、`src/cases`、`src/graph` 等历史主链。
- 通用项目说明以本文件为准；工具专属约束只写在各自的适配文档里。
- 任何新增目录、类型、脚本和 artifact，都必须能回指到论文主链，而不是为了“未来平台扩展”预留壳。
- 代码注释采用“稀疏高值”原则：模块级说明解释存在意义，关键注释解释边界、数据流或非显然实现，不写逐行翻译式注释。
- 文档语言以中文主导，命令、路径、类型名和接口名保留英文。
