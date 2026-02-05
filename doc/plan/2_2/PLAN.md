## PLAN（阶段任务 / 验收标准）

本文件只写“要做什么 + 如何验收”，不写不可变约定（见 `SPEC.md`）与 I/O schema（见 `AGENT_IO.md`）。

---

### Phase 1（P0）：静态可复现基线（D0~D1）
目标：不引入外环 agent，人工给定一份 `model_config.yaml`，TSPN 能在单一数据集跑通并产出可复现产物。

1) 代码迁移与包内化（`src/model/explainable/`）
   - 修复：complex→real、禁止 `.data`、修复 squeeze、WaveFilters 参数维度一致性
   - 验收：CPU smoke-run 通过；forward shape 恒为 `(B,L,C)`；输出 dtype 为 real
   - 备注：若容器环境未安装 `torch`，可在本机训练环境执行（例如 `conda activate LQ_signal`）以完成训练验收
2) 确定性训练入口
   - `case.yaml + model_config.yaml -> train -> artifacts`
   - 验收：输出 `metrics.json/predictions.csv/operator_importance` 与 checkpoint
3) 最小解释性
   - operator importance top-k（每层）
   - WaveFilters 参数统计（若启用）
   - 验收：报告中能回答“哪层/哪算子最重要”

---

### Phase 2（P1）：解释性→反思→可行动反馈（D2）
目标：训练后输出 agent“看得懂”的结构化反馈，不是只有标量指标。

1) `get_semantic_diagnosis()`（或等价 JSON 摘要）
   - 输入：confusion matrix + operator importance + WF 频带统计
   - 输出：`error_modes + suggested_actions`
   - 验收：能稳定生成 `reason_codes`（见 `AGENT_IO.md`）
2) Reporter 集成
   - 验收：每个 run 自动生成“结果 + 解释性证据”报告

---

### Phase 3（P2）：外环 agent 闭环（D3）
目标：外环 agent 根据 Phase 2 的反馈更新 `model_config.yaml`，跑 Smoke→Train→Explain→Reflect 循环。

1) `model_config_agent`（只输出 `ConfigPatch`）
2) `config_reflect_agent`（决定是否继续、给 reason_codes）
3) `phm_outer_graph` 编排：`Gen Config -> Smoke -> Train -> Explain -> Reflect -> (Loop/Stop)`
验收：
- 任何 invalid config 都能被 fail-fast 拦截并回退
- 闭环最多 N 轮（例如 5），能生成一条可复现的 config 演化链（crud_history）

---

### Phase 4（P3）：多数据集 + Few-shot DG（Task C）
目标：把“可跨数据集”从口号变成工程约束，并完成 Task C（Few-shot DG）。

1) data_factory 接入（reader/task/factory/sampler）
   - 验收：仅通过修改 `case.yaml` 的 `data/task` 就能切换数据集
2) Few-shot DG（episodic）
   - 支持 `task.few_shot.{n_way,k_shot,q_query,n_episodes_*}`
   - 验收：DG 指标/消融表自动生成

---

### 默认优先级（建议）
- P0：先让 TSPN 作为“安静的可复现黑盒/白盒”跑通
- P1：再让解释性成为可行动反馈
- P2：再让 agent 真正闭环优化
- P3：最后才做多数据集与 few-shot（避免同时 debug 三条链路）
