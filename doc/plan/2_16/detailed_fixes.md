# PHMGA 2.16 详细修复状态矩阵

说明：状态定义为 `Closed / Needed / Deprecated`。本表以当前仓库实现为准。

| 项目 | 状态 | 证据锚点 |
|---|---|---|
| Builder 迭代上限（图层） | Closed | `src/phm_outer_graph.py:98` |
| Reflect 迭代计数更新 | Closed | `src/agents/reflect_agent.py:161` |
| `PHMState.max_builder_iterations` | Closed | `src/states/phm_states.py:389` |
| Plan payload 鲁棒解析（dict/list/fenced/bad-json） | Closed | `src/agents/plan_agent.py:44` |
| `inquirer` Pearson/zero-vector 稳定性 | Closed | `src/agents/inquirer_agent.py:7` |
| `train_backend` 白名单 fail-fast | Closed | `src/phm_outer_graph.py:142` |
| `dataset_preparer` 父链循环保护 | Closed | `src/agents/dataset_preparer_agent.py:11` |
| DistanceOp cosine 安全除法 | Closed | `src/tools/multi_schemas.py:71` |
| Hjorth 零除保护 | Closed | `src/tools/aggregate_schemas.py:304` |
| Cepstrum eps 稳定化 | Closed | `src/tools/transform_schemas.py:69` |
| 小波重构长度对齐 | Closed | `src/tools/transform_schemas.py:166` |
| `_softplus_inv` 数值稳定 | Closed | `src/model/explainable/tspn.py:246` |
| 梯度裁剪 `grad_clip_norm` | Closed | `src/agents/deep_model_train_agent.py:827` |
| Provider 环境一致性校验 | Closed | `src/configuration.py:113` |
| preflight provider_checks 输出 | Closed | `src/utils/preflight.py:114` |
| State checksum（sha256 + strict gate） | Closed | `src/utils/__init__.py:590`, `src/utils.py:384` |
| DAG PNG 导出 DOT 兜底 | Closed | `src/states/phm_states.py:278` |
| “111 bugs 全量修复”叙事作为执行目标 | Deprecated | 用 `summary.md` + 本矩阵替代 |

## 本轮新增测试证据
- `tests/test_executor_backend_validation.py`：非法 backend fail-fast
- `tests/test_inquirer_agent.py`：Pearson/cosine 边界
- `tests/test_dataset_preparer_agent.py`：parent cycle 防护
