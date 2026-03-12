# 00 Problem And Protocol

## 论文任务与核心 claim

论文版 PHMGA 研究的不是通用多智能体平台，而是一个可复现实验系统：用 agentic workflow 生成结构先验 DAG，再把结构先验编译到不同 graph path，验证其在工业时间序列诊断中的解释性和可训练性。

## 为什么需要 LLM workflow 生成结构先验

- 将传统信号处理知识写成结构化 DAG，而不是写成不可追溯提示片段。
- 使方法图、JSON、算子清单和最终报告能够共享同一份结构证据。
- 为 `dag_only`、`ml`、`torch` 三条后端路径提供统一入口。

## Canonical data catalog

- `PHM-Vibench metadata` 是正式数据目录协议。
- `RM101` 与 `Ottawa` 必须被归一化到同一 metadata schema。
- 每个样本最少包含：`sample_id`、`dataset`、`label`、`sampling_rate`、`length`、`channels`、`operating_condition`、`source_h5`。

## 当前真实数据映射

- `config/data/rm101.yaml` 对应 `RM_101_THU_GEARBOX`
- `config/data/ottawa.yaml` 对应 `RM_017_Ottawa19`
- 运行时通过完整配置文件选择数据集，例如 `config/runs/rm101_dag.yaml`、`config/runs/ottawa_ml.yaml`

## 当前实现态

当前 `config/` 和 `src/data/protocol.py` 已经把下面这些字段落成正式输入合同：

- `defaults.dataset`
- `defaults.graph_path`
- `data.dataset_name`
- `data.catalog`
- `data.metadata_schema_version`
- `data.metadata_path`
- `data.h5_path`
- `data.selection.*`
- `data.split.*`
- `data.window.*`
- `data.selected_channels`

## 正式 split 语义

- 唯一正式切分对象：`train_ids`、`val_ids`、`test_ids`。
- 不再保留 `ref/test` 等旧别名。
- 标签对齐、窗口切片、指标汇总都只能围绕 `train/val/test`。

## Window protocol

- `window_size`
- `stride`
- `slice_mode`
- `drop_last_window`

窗口协议必须随 resolved config 和最终产物一起落盘，以保证复现性。

## 数据泄漏边界

- 切分先于窗口切片。
- 不允许跨 split 共享样本或派生窗口。
- `Ottawa` 的原始格式差异只能在 `src/data` 内部适配，不得渗透到 workflow、bridge、training。

## 关于 synthetic materialization

- 当前 `src/data/protocol.py` 已支持两种来源：
  - 配置里存在 `metadata_path + h5_path` 时，走真实 PHM-Vibench loader
  - 配置里没有真实路径时，走 synthetic fallback
- synthetic 只用于 smoke 和无数据环境下的最小闭环，不是正式论文数据来源声明。

## metadata 与 H5 的真实文件事实

- `metadata.xlsx` 和 `gear_metadata.xlsx` 的真实列名是 `Sample_lenth`、`Channel`，不是 README 风格的说明列名。
- 真实 H5 样本当前观测到的 shape 是 `(L, C, 1)`。
- loader 在内部会把 H5 数据统一转成 `(C, L)`，并以 H5 实测长度/通道数作为窗口化与张量处理的权威来源。
- `RM_101_THU_GEARBOX` 存在 metadata `768000` 与 H5 `767999` 的长度偏差，当前实现按 H5 实测值处理。

## 不同 graph 路径的数据使用方式

- `dag_only`: 使用 metadata 和协议上下文生成结构先验，不产生训练指标。
- `ml`: 从 DAG 编译出的特征流水线读取 train/val/test 数据并输出轻量基线指标。
- `torch`: 从 DAG 编译出的模型构建计划读取 train/val/test 数据并输出训练曲线、checkpoint 和解释性结果。
