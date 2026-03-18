# AGENTS.md

通用项目说明、运行方式、配置入口、测试方式和论文主链约束统一见 [README.md](README.md)。

本文件只保留对 Codex / 通用 coding agent 有效的仓库协作规则。
永远保持第一性原理，永远保持第一性原理，永远保持第一性原理。
## 工作优先级

1. 先服从 `README.md` 中定义的论文版主链。
2. 再服从 `doc/structure/` 中的结构边界与删除账本。
3. 任何实现、重构或补文档，都不得恢复旧平台主链或旧 split 术语。

## 仓库协作规则

- 正式人类入口是根目录 `main.py`。
- `scripts/preflight.py` 和 `scripts/run_case.py` 只作为兼容入口与底层实现保留。
- 正式配置入口是 Hydra root `config/config.yaml` 及其 config groups / `config/runs/*.yaml` preset。
- 前端 workflow 只能围绕 `WorkflowState -> DAGTracker -> validated DAG JSON`。
- bridge 是前后端唯一法定接口层；不要绕过 `compile_dag_for_path()` 直接把 workflow 状态塞进训练端。

## 文档与注释策略

- 通用说明不要重复写到本文件，统一引用 `README.md`。
- 修改接口、路径、artifact 或 graph path 行为时，必须同步更新 `doc/structure/`。
- 注释采用“稀疏高值”策略：解释模块角色、数据流边界和非显然逻辑，不写逐行翻译。

## 明确禁止

- 不要重新引入历史平台入口、历史目录、旧 facade 或多入口运行方式。
- 不要重新引入 `ref/test`、`labels_ref/tst` 等旧 split 语义。
- 不要把当前 `torch` path 的 NumPy fallback 描述成完整 PyTorch 训练器，除非真实实现已替换。
