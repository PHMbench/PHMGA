# 2_25 LLM-DAG-TSPN 闭集稳定化计划

## 1. 目标与结论
- 目标：把当前流程切换为“闭集编译系统”，仅允许 TSPN 已实现并签约的算子进入 DAG->TSPN。
- 结论：默认采用 `rm101_closed_v1`，先保证结构可编译、可训练、可审计，再推进覆盖率。

## 2. 实施阶段
### Phase 1: `rm101_closed_v1` 闭集稳定
1. 定义 `operator_contract=rm101_closed_v1`。
2. Planner 仅暴露白名单算子。
3. Execute 对每个 step 执行合同校验，违规即 `contract_violation` 并停止当前轮。
4. Bridge 检测到越界算子直接 fail-fast。
5. 训练门禁：`closed_world_pass=true` 才允许进入训练。

### Phase 2: `rm101_closed_v2` 增强
1. 先实现后放行：`band_power`、`cross_correlation`、`psd`、`spectrogram`、`cepstrum`、`savgol_filter`。
2. 新增算子通过单测与集成验证后并入白名单。

### Phase 3: 全库闭集推进
1. 保持“实现后放行”。
2. 多输入/形变复杂算子纳入 `TSPN v2` 架构任务，单独推进。

## 3. 接口变更
1. `data.operator_contract`，默认 `rm101_closed_v1`。
2. `data.enforce_tspn_closed_world`，默认 `true`。
3. `config_resolve.json` 新增字段：`operator_contract`、`closed_world_pass`、`compile_quality`。
4. 新增工件：
   - `dag_compile_report.json`
   - `contract_violation_report.json`
   - （保留）`compatibility_report.json`

## 4. 验收标准
1. `closed_world_pass` 成功率 >= 95%（重复运行统计）。
2. `proxy_nodes_count=0`。
3. `unsupported_nodes_count=0`。
4. `identity_fallback_nodes_count=0`。
5. `effective_ops_ratio > 0`。
6. `_evidence/*.artifacts.json` 必含 `metrics_path`。

## 5. 默认假设
1. 本轮优先 RM101 闭集稳定，不做全库 1:1 一次性重构。
2. 未实现算子按“禁用”处理，不走代理降级。
3. 性能优化在结构稳定后进行。

