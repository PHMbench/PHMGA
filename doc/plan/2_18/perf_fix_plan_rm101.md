# RM101 性能修复计划（v2.18 收口版）

## 1) 目标与约束

- 主目标：`val_acc >= 0.90`（最多 3 轮恢复）
- 次目标：提升 DAG->TSPN 映射有效性与训练稳定性
- 约束：不引入 baseline，不做统计检验
- Active matrix 口径：`3 LLM × 2 × 3 = 18 combos`（`GLM-4.5` 不再参与）

## 2) 先分层：运行失败不等于性能失败

执行诊断必须按以下层级逐层通过：

1. **联通/权限层**
   - 典型错误：`403 model access`、`401 invalid key`
   - 处理：修 provider/model 权限或 `.env` key/base，不进入性能结论
2. **资源层**
   - 典型错误：`rc=137`（OOM/被系统 kill）
   - 处理：先 `fast profile` 验证链路，再回 `standard/highacc`
3. **模型性能层**
   - 仅当前两层通过后，才讨论 RM101 指标与消融结论

## 3) 当前 RM101 中期事实（M2 子集）

| Ablation | Val Acc | Test Acc | Val Macro-F1 | Test Macro-F1 |
| --- | --- | --- | --- | --- |
| A0 | 0.5417 | 0.6198 | 0.4340 | 0.4584 |
| A1 | 0.4479 | 0.5208 | 0.3325 | 0.3662 |
| A2 | 0.5781 | 0.6042 | 0.4682 | 0.4478 |

解释（仅中期）：
- `A0` 相对 `A1` 有明确提升（反思闭环在当前子集有效）。
- `A2` 在验证集高于 `A0`，但测试集略低于 `A0`，说明先验初始化收益需更完整矩阵确认。

## 4) 修复路径（按优先级）

### P0 依赖闭环（阻断）
1. 按 `env_lock.md` 完成 `librosa/nolds/antropy/graphviz` 安装与验证。
2. case 中启用：
```yaml
preflight:
  strict: true
  block_on_missing_dependencies: [librosa, nolds, antropy]
```
3. 依赖缺失直接阻断训练。

### P1 DAG 质量
- RM101 建议：
  - `data.max_ops_per_iteration: 8`
  - 优先算子：`filter/hilbert_envelope/fft/stft/band_power/cross_correlation/spectral_kurtosis`
- 失败分类必须结构化记录：`missing_dep` / `unsupported_op` / `param_error`。

### P2 DAG->TSPN 桥接质量
- `config_resolve.json.bridge_quality` 必须包含：
  - `effective_ops_ratio`
  - `effective_ops_count`
  - `identity_ops_count`
  - `unsupported_nodes_count`
  - `dropped_nodes_count`
- RM101 推荐：
```yaml
data:
  use_dag_model_config: true
  unsupported_policy: drop
  bridge_min_effective_ops_ratio: 0.4
```

### P3 训练策略（中等预算）
推荐 `highacc`：
```yaml
data:
  epochs: 60
  patience: 15
  lr: 3e-4
  scheduler: cosine
  use_class_weight: true
  use_weighted_sampler: true
  label_smoothing: 0.05
  early_stop_metric: val_macro_f1
```

### P4 易操作性
- 统一入口：
  - `scripts/paper/run_combo.sh --train-profile fast|standard|highacc`
  - `scripts/paper/rerun_failed_from_manifest.sh --manifest <manifest_dedup.jsonl>`
- 失败重跑口径：同一 `combo` 仅认最后一条记录。

## 5) 三轮恢复模板

1. **R1 标准轮**：`standard`，要求 `effective_ops_ratio >= 0.4`
2. **R2 强化轮**：`highacc` + `builder.max_depth=12` + `max_iterations=30`
3. **R3 上限轮**：保持 R2，提升模型容量（`out_channels/scale` 上调）

## 6) 验收标准

1. 功能层：不再出现 `spectral_entropy/librosa/filter_type` 阻塞错误。
2. 结构层：DAG 深度 `>= 6` 且 `config_source_mode=dag_bridge`。
3. 性能层：3 轮内达 `val_acc >= 0.90`；若未达标，提交差距报告（最佳值、差值、主瓶颈、下一步资源需求）。
