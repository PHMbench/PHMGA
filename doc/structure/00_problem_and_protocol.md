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

## 不同 graph 路径的数据使用方式

- `dag_only`: 使用 metadata 和协议上下文生成结构先验，不产生训练指标。
- `ml`: 从 DAG 编译出的特征流水线读取 train/val/test 数据并输出轻量基线指标。
- `torch`: 从 DAG 编译出的模型构建计划读取 train/val/test 数据并输出训练曲线、checkpoint 和解释性结果。
