# Redundancy Inventory And Removal Ledger

| 模块 | 冗余项 | 为什么冗余 | 替代入口 | 当前状态 | 绑定测试 | removal_stage |
|---|---|---|---|---|---|---|
| model | `src/model.py` | 顶层单文件与 `src/model/` 双入口并存 | `src/model/` | removed | `tests/structure/test_docs.py` | done |
| utils | `src/utils.py` | 与 `src/utils/__init__.py` 双实现，行为容易漂移 | `src/utils/` package | removed | `tests/structure/test_docs.py` | done |
| prompts | `src/prompts/reflector_prompt.py` | 与正式 prompt 入口重复 | `src/prompts/*` | removed | `tests/structure/test_docs.py` | done |
| agents | `src/agents/prompt_research.py` / `src/agents/prompts_research.py` | 重复 research prompt 入口 | `src/prompts/research_prompts.py` | removed | `tests/structure/test_docs.py` | done |
| agents | `src/agents/builder/plan.py` / `executor/execute.py` / `report/final_report.py` / `train/deep_model.py` | 仅做 re-export 的兼容壳，不应继续承载新逻辑 | 平铺正式入口 `src/agents/*.py` | facade_only | `tests/structure/test_agents_structure_contracts.py` | phase3_delete |
| agents | `src/agents/shared/compat.py` | 纯透传兼容 helper，没有长期抽象价值 | 直接调用正式 agent 入口 | compat_only | `tests/structure/test_agents_structure_contracts.py` | phase3_delete |
| agents | `src/agents/**/__pycache__` | 构建产物，不属于仓库结构 | 无 | removed | `tests/structure/test_agents_structure_contracts.py` | done |
| cases | `src/cases/*.ipynb` | 历史实验入口不适合作为正式 runner | `doc/` / script archive | removed | `tests/structure/test_docs.py` | done |
| graph | `src/phm_outer_graph.py` | 历史 graph 主入口与新 registry 并存 | `src/graph/registry.py` | facade_only | `tests/structure/test_graph_agents_cases.py::test_phm_outer_graph_is_thin_facade` | phase3_delete |
| cases | `case1.py` 内联过多 runtime glue | case 文件承担了过多 runner/config/data/model glue | `src/cases/base_runner.py` + helpers | shrinking | `tests/structure/test_graph_agents_cases.py` | phase3_split |
| data/model | `case1 -> data_cfg` 重复写入 `model_profile` / `autofit_*` / `model_config_path` | 数据层和模型层同时持有模型真相 | `state.model_config_path` + `state.model_cfg` + `model.*` | mainline_removed | `tests/test_rm101_training_contract.py`, `tests/test_preflight_and_config_resolve.py` | phase2_closed |
| data/state/trainer | fixed-ids 旧 `ref/test` 兼容链 | split 语义和训练语义不一致，导致状态/训练/报告混用 | canonical `data.selection` + `results.train/val/test` + `labels_train/val/test` | mainline_removed_keep_ingress_compat | `tests/test_state_io_integrity.py`, `tests/test_dataset_preparer_agent.py`, `tests/test_tspn_bootstrap_agent.py`, `tests/test_rm101_training_contract.py` | phase2_closed |
| tests/docs | `utils/agents/tests` 中旧 split 命名 | 测试继续使用旧命名会把实现拖回去 | canonical split tests | in_progress | `tests/test_*`, `tests/structure/test_docs.py` | phase2_cleanup |
| config | `src/configuration.py` 旧配置壳 | 历史 provider/config 入口，不应回到主链 | `src/config/*` + Hydra | compat_only | `tests/structure/test_config_data_llm.py` | phase3_delete |
