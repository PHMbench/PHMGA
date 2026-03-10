# 2_25 `rm101_closed_v1` 合同定义

## 1. 合同名与默认配置
1. `data.operator_contract=rm101_closed_v1`
2. `data.enforce_tspn_closed_world=true`
3. 适用路径：LLM 规划 -> DAG 执行 -> DAG->TSPN 编译 -> 训练

## 2. 合法算子白名单

### layer (8)
- `detrend`
- `differentiate`
- `fft`
- `filter`
- `hilbert_envelope`
- `integrate`
- `normalize`
- `stft`

### feature (19)
- `abs_mean`
- `clearance_factor`
- `crest_factor`
- `entropy`
- `kurtosis`
- `max`
- `mean`
- `min`
- `peak_to_peak`
- `rms`
- `shape_factor`
- `skew`
- `spectral_centroid`
- `spectral_flatness`
- `spectral_kurtosis`
- `spectral_skewness`
- `std`
- `var`
- `zero_crossing_rate`

## 3. 违规处理规则
1. Planner 阶段：工具清单仅暴露白名单；LLM 输出越界算子在 sanitize 阶段丢弃并记录 `contract_violation`。
2. Execute 阶段：step 命中越界算子时记录 `contract_violation`，立即停止当前构建轮。
3. Bridge 阶段：若 DAG 含越界算子，闭集模式下 fail-fast。
4. Train 阶段：仅当 `closed_world_pass=true` 才允许训练；否则直接返回错误并保留报告工件。

## 4. 编译通过条件（`closed_world_pass=true`）
1. `contract_violations_count=0`
2. `proxy_nodes_count=0`
3. `unsupported_nodes_count=0`
4. `identity_fallback_nodes_count=0`
5. `effective_ops_ratio > 0`

## 5. 标准输出工件
1. `compatibility_report.json`
2. `contract_violation_report.json`
3. `dag_compile_report.json`
4. `config_resolve.json` 中同步记录：
   - `operator_contract`
   - `closed_world_pass`
   - `compile_quality`

