# 欠缺环节：LLM 生成 DAG → TSPN 的“真正闭环”还缺什么

本文件只关注“从 **LLM 生成 DAG** 到 **DAG→TSPN→训练→反思→再改结构** 的闭环”，列出目前还缺的关键环节（按优先级）。

---

## Gap 1（最高优先级）：Reflection 还没真正“看训练证据”做结构改写
现状：
- `reflect_agent` 主要审查 DAG 本身（拓扑/问题总结），输出 `needs_revision`
- 训练完成后虽有 `train_history: List[TrainReport]`，但 **反思并未使用** `TrainReport.metrics/explain_summary/error_modes` 来指导下一轮结构调整

缺什么：
- `config_reflect_agent`（或升级 reflect）：
  - 输入：最近一次 `TrainReport`（至少 `metrics + explain_summary + confusion_matrix`）
  - 输出：`ConfigPatch`（对 `model_config.yaml` 的白名单 patch）+ `reason_codes`
- 外环循环把 patch 应用到 `current_model_config`，并触发下一次训练（Immutable Config 模式，禁热更新）

建议落地（最小可跑 P0）：
1. 在 `Reflect` 阶段增加一个分支：如果 `train_history` 非空，则基于 TrainReport 反思并产生 patch
2. 新增 `model_config_agent`（只输出 ConfigPatch JSON，不直接写文件）
3. 增加 “Patch 应用器”：把 patch merge 到 dict，并写入新的 `run_dir/model_config.yaml`

---

## Gap 2：PlanAgent 的 DAG 语义还缺“对 TSPN 可解释初始化友好”的协议
现状：
- `DAG2ConfigAdapter` 能从 `ProcessedData.method/meta.params` 识别少量语义（如 bandpass）
- 但 PlanAgent 生成的工具参数未被强制规范：不同 token 需要不同可解释参数，缺 **统一 contract 表**

缺什么：
- 明确一张“外环算子 → 内环 token/参数”的映射与约束（推荐落到 `SPEC.md` 扩展表）
  - 例如：
    - `transform_schemas.FilterOp(filter_type=band, cutoff=[l,h])` → `WF(fc_hz=(l+h)/2, fb_hz=(h-l)/2)`
    - `hilbert_envelope` → `HT`
    - `fft` → `FFT(align_strategy=interp)`
- ExecuteAgent 在写 `ProcessedData.meta["params"]` 时保证字段齐全（目前已经会写 `params`，但字段命名要收敛）

建议落地：
- 给 `ProcessedData.meta` 增加稳定字段：`token_hint` / `op_uid_source` / `fs_hz`
- 将 band/fft/hilbert 等关键 token 的 I/O contract 与 params schema 显式化（避免 heuristic string match）

---

## Gap 3：DAG→TSPN 配置生成仍较“启发式”，未充分利用 DAG 结构
现状：
- `DAG2ConfigAdapter` 依据 DAG depth 生成层数，并在每层做 token 去重/补 I
- 尚未利用：
  - 多分支 DAG 的“并行结构”决定每层 ops 数量
  - DAG 节点的强类型（目前 method 字符串）

缺什么：
- 更精确的拓扑对齐策略：
  - 多通道/多节点 → 明确映射到 TSPN 的 `in_channels` 与 layer width 分配策略
  - 多个 filter 节点 → 对应多个 `WF` token（而不是“同层去重”）
  - 特征节点（Mean/Std/RMS 等）→ 作为 feature head 的候选集合

建议落地（逐步增强）：
1. 把 “同层 token 去重” 改为 “按 DAG 节点数决定 token 复用次数”，保证结构表达力
2. 在 `init_metadata` 中保存 `op_uid -> source_node_id`（已有 `source_nodes`），并用于报告/可解释追踪

---

## Gap 4：LangGraph 生产化编排未恢复（当前主要靠 fallback）
现状：
- `agent` 环境中 `langgraph==1.0.7` 与 `langchain-core==0.1.53` 不兼容，导致真实 `StateGraph` import 失败
- 目前靠 `src/phm_outer_graph.py` 的 `_FallbackGraph` 跑通链路

缺什么：
- 版本收敛策略（二选一）：
  1) 固定 `langgraph<1.0.0` 以兼容当前 `langchain-core<0.2.0`（最符合当前 requirements.txt 语义）
  2) 升级 langchain stack 到与 langgraph 1.x 兼容的版本（会牵涉更大迁移）

建议落地：
- 在 `conda activate agent` 的依赖里显式 pin（否则 conda/pip 易装到 langgraph 1.x）

---

## Gap 5：诊断精度“可提升/可复现”的实验环仍缺工程化指标与基线
现状：
- 已能产出 `metrics.json/operator_importance.json/wavefilters_params.json`
- 但“高精度”依赖真实数据、超参、训练策略；Dummy_Data 只能用于联通性

缺什么（最小科研闭环）：
- 3 个数据集固定跑法（dataset_name 列表）与统一 split protocol
- Ablation runner（例如：禁用某层 WF/FFT/HT，或减少 ops 数量）
- 结果汇总（表格/曲线）输出到 `doc/paper/` 或 `save/summary/`

---

## Gap 6（可选）：把训练环迁移到 PyTorch Lightning（你已安装）
现状：
- 训练循环目前是纯 torch（足够快、可控）

可选增强：
- 用 Lightning 统一 logger/checkpoint/early stopping（避免手写训练细节）
- 但要注意：外环“可复现”优先，Lightning 配置也要纳入 Immutable Config

