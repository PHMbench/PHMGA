# PHMGA Bug 状态矩阵（基于 consolidated 报告）

**日期:** 2026-02-15  
**来源:** `doc/plan/2_10/bug_report_consolidated.md`  
**范围:** Critical C-1 ~ C-18 的当前代码状态核查

## 判定标准
- `Closed`: 代码中已落地且行为符合修复目标
- `Partial`: 部分修复，仍存在关键残留风险
- `Open`: 关键问题仍未修复

## Critical 状态矩阵

| ID | 标题 | 状态 | 代码锚点 | 说明 |
| --- | --- | --- | --- | --- |
| C-1 | 缺失 `get_llm` 导致 ImportError | Closed | `src/model/__init__.py:71` | 已存在 `get_llm`，并具备 provider/model 一致性校验。 |
| C-2 | Builder 无限循环风险 | **Closed (本批修复)** | `src/cases/case1.py` | 新增 `builder.max_iterations` 安全上限（默认 50）。 |
| C-3 | tracker 缓存失效导致状态不一致 | **Closed (本批修复)** | `src/states/phm_states.py` | `PHMState.tracker()` 改为每次返回新 `DAGTracker`，移除缓存依赖。 |
| C-4 | `processed_data` / `results` 字段不一致 | **Closed (本批修复)** | `src/states/phm_states.py`, `src/tools/comparator_tool.py` | 节点数据读取统一走 `results`。 |
| C-5 | 方法缩进错误（类外定义） | **Closed (本批修复)** | `src/states/phm_states.py` | `transfer_to_langgraph/save/load` 已移回 `DAGTracker` 类内。 |
| C-6 | 缺少环路检测 | **Closed (本批修复)** | `src/states/phm_states.py` | `add_node` 新增 `is_directed_acyclic_graph` 检查并拒绝成环边。 |
| C-7 | legacy 配置硬编码路径 | **Closed (本批修复)** | `config/case1.yaml` 等 | `/home/lq/...` 已改为当前仓库路径。 |
| C-8 | legacy 配置缺少 `run_executor` | **Closed (本批修复)** | `config/case1.yaml` 等 | 已补 `run_executor: true`。 |
| C-9 | `_softplus_inv` 数值不稳定 | **Closed (本批修复)** | `src/model/explainable/tspn.py` | 调整为 `clamp(1e-6) -> log(expm1)` 并新增输出区间裁剪 `[-20,20]`。 |
| C-10 | 缺少梯度裁剪 | **Closed (本批修复)** | `src/agents/deep_model_train_agent.py`, `src/model/explainable/config_schema.py` | 新增 `train.grad_clip_norm`，训练循环在 backward 后执行 `clip_grad_norm_`。 |
| C-11 | Transform 算子参数验证不足 | **Closed (本批修复)** | `src/tools/transform_schemas.py`, `src/tools/expand_schemas.py`, `src/tools/multi_schemas.py` | 补齐 Filter/Resample/Savgol/Patch/Cosine 路径校验与安全处理。 |
| C-12 | Hjorth 除零风险 | **Closed (本批修复)** | `src/tools/aggregate_schemas.py` | activity/var_dx/mobility 全链路加入 dtype 自适应 `eps`。 |
| C-13 | API Key fail-fast 缺失 | **Closed (本批修复)** | `src/model/__init__.py`, `src/configuration.py`, `src/utils/preflight.py` | 新增 `Configuration.validate_provider_env()`，preflight 引入 provider_checks，Gemini 缺 key 直接失败。 |
| C-14 | `parents` 类型不一致 | **Closed (本批修复)** | `src/states/phm_states.py` | 新增 `field_validator`，统一规范为 `List[str]`。 |
| C-15 | leaves 更新逻辑不完整 | **Closed (本批修复)** | `src/states/phm_states.py` | 改为按 `out_degree==0` 重算 leaves。 |
| C-16 | `export_json` 访问非通用字段 | **Closed (本批修复)** | `src/states/phm_states.py` | 仅导出通用字段，可选字段通过 `getattr` 条件添加。 |
| C-17 | 小波去噪长度不匹配 | **Closed (本批修复)** | `src/tools/transform_schemas.py` | `waverec` 后新增显式裁剪/零填充以对齐原始长度。 |
| C-18 | Cepstrum log(0) 稳定性 | **Closed (本批修复)** | `src/tools/transform_schemas.py` | 固定 `1e-9` 改为 dtype 自适应 `eps`。 |

## 本批落地内容（RM101 上线并行）
1. 新增 `RM_101_THU_GEARBOX` 运行配置：`config/case_exp_gearbox_rm101.yaml`。  
2. 更新 CLAUDE 本地记忆：`.claude/settings.local.json` 增加默认 `agent` 环境提示。  
3. 新增 builder 全局迭代上限（防止卡死）：`builder.max_iterations`。  
4. 收口 C3/C4/C5/C6/C7/C8/C14/C15/C16。
5. 修复 vibench 过滤元数据缓存写权限问题：`src/utils/data_factory_wrapper.py` 新增 `data.cache_dir` 与只读目录 fallback。  
6. 修复 vibench fallback reader 的相对导入问题：`src/utils/data_factory_wrapper.py` 为 reader 注入 package context。  

## RM101 实跑验证（2026-02-15）
- 预检查通过：`python main.py preflight --config config/case_exp_gearbox_rm101.yaml` 返回 `OK: True`。  
- 链路跑通：`FAKE_LLM=true PHM_REPORT_MODE=template python main.py case1 --config config/case_exp_gearbox_rm101.yaml` 运行完成。  
- 核心产物存在：`metrics.json`、`dataset_manifest.json`、`config_resolve.json`、`final_report.md`。  
- 当前风险：`graphviz`、`nolds` 缺失仅触发 warning；Builder 仍有 `plan_agent` 的 fake-LLM 解析告警（`'list' object has no attribute 'get'`）。

## 下一批建议（按风险排序）
1. 补齐 `src/utils.py` 与 `src/utils/__init__.py` 的双实现治理（收敛到单一实现并加兼容层）。  
2. 处理 Pydantic v2 deprecation 警告（`ConfigDict` / `json_schema_extra`）。  
3. 增补高成本集成回归（RM101 在线 LLM + vibench 全链路 nightly）。  
