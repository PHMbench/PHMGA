# GEMINI.md

通用项目概述、运行方法、配置方式和论文版主链统一见 [README.md](README.md)。

本文件只保留 Gemini 代理需要知道的仓库适配信息。

## 仓库摘要

- 这是论文导向的 PHM 研究仓库，不是旧版通用 agent 平台。
- 三条 graph path 为 `dag_only`、`ml`、`torch`。
- 前后端法定边界是 `validated DAG JSON`，bridge 负责把它编译到不同 path。
- 运行对象由 Hydra root config 与 `config/runs/*.yaml` preset 决定，而不是通过 CLI 临时改写数据集名。

## Gemini 代理的工作边界

- 通用说明引用 `README.md`，不要在本文件重复维护完整项目手册。
- 结构合同以 `doc/structure/` 为准，尤其是 `00` 到 `04` 五份核心文档。
- 不要生成或引用历史平台入口、历史目录或旧 split 术语。

## 当前实现事实

- 正式人类入口是根目录 `main.py`；`scripts/preflight.py` 和 `scripts/run_case.py` 只作为兼容入口保留。
- `config/data/rm101.yaml` 对应 `RM_101_THU_GEARBOX`，`config/data/ottawa.yaml` 对应 `RM_017_Ottawa19`。
- `torch` path 当前是最小 trainable surrogate，不是完整 PyTorch 训练框架。
