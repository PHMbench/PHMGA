# CLAUDE.md

通用项目说明、仓库结构、运行命令、测试方式和论文版边界统一见 [README.md](README.md)。

本文件只说明 Claude Code 在本仓库内工作的最小适配信息。
永远保持第一性原理，永远保持第一性原理，永远保持第一性原理。
## Claude Code 应关注的事实

- 正式运行入口：
  - `python main.py runtime.action=preflight +runs=rm101_dag`
  - `python main.py +runs=rm101_dag runtime.output_dir=artifacts/rm101_dag`
- 正式结构说明在 `doc/structure/`，不是历史平台文档。
- 当前仓库是论文版最小研究闭环，不存在历史平台式工作流。

## 修改时的边界

- 通用项目介绍不要复制到本文件，直接引用 `README.md`。
- 如果修改 graph path、artifact、状态对象或 bridge 合同，必须同步更新 `doc/structure/`。
- 如果修改运行入口或命令示例，优先更新 `README.md`，再视需要更新本文件。

## 当前实现注意点

- `torch` path 当前是 NumPy fallback 训练器，用于离线 smoke 和合同验证。
- `config/data/rm101.yaml` 对应 `RM_101_THU_GEARBOX`，`config/data/ottawa.yaml` 对应 `RM_017_Ottawa19`。
- 当前 smoke 数据来自 `config/data/*_synth.yaml` 对应的 synthetic materialization。
