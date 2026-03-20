# Enriched Operator Diagnostic Report

## 运行样例

### `dag_only`

- config: `config/runs/rm101_synth_dag.yaml`
- command:
  - `.venv/bin/python scripts/run_case.py --config config/runs/rm101_synth_dag.yaml --output-dir artifacts/diagnostic_rm101_synth_dag_enriched`
- result:
  - status: success
  - rounds: 1
  - reflection decision: `finish`
  - output dir: `artifacts/diagnostic_rm101_synth_dag_enriched`

### `ml`

- config: `config/runs/rm101_synth_ml.yaml`
- command:
  - `.venv/bin/python scripts/run_case.py --config config/runs/rm101_synth_ml.yaml --output-dir artifacts/diagnostic_rm101_synth_ml_enriched`
- result:
  - status: success
  - rounds: 1
  - reflection decision: `finish`
  - output dir: `artifacts/diagnostic_rm101_synth_ml_enriched`

## 实际产物摘要

### enriched `dag_only`

- DAG 节点数：18
- edge 数：20
- operator categories：
  - `TRANSFORM`
  - `EXPAND`
  - `AGGREGATE`
  - `MULTI_VARIABLE`
  - `DECISION`
- 实际方法链包含：
  - `signal.normalize`
  - `signal.filter`
  - `signal.hilbert_envelope`
  - `signal.stft`
  - `signal.patch`
  - `signal.psd`
  - `feature.kurtosis`
  - `feature.spectral_centroid`
  - `feature.band_power`
  - `multi.cross_correlation`
  - `multi.concatenate`
  - `decision.threshold`
- `decision_side_outputs.json` 已生成
- `dag_quality_summary.json`：
  - `depth_ok: true`
  - `execution_gap_count: 0`
  - `recommendation_hint: finish_candidate`

### enriched `ml`

- `feature_pipeline.json` 中已有 5 条可编译 feature specs
- 实际 feature lineage 已覆盖：
  - `normalize -> stft -> spectral_centroid`
  - `normalize -> patch -> kurtosis`
  - `normalize -> psd -> band_power`
  - `normalize -> filter -> hilbert_envelope -> kurtosis`
- `metrics.json`、`importance.json`、`similarity_artifacts.json` 均已生成
- `decision_side_outputs.json` 同样存在，但没有进入 feature lineage

## 当前主链结论

当前仓库已经从“最小 smoke DAG”升级到“有明显 PHM 方法语义的最小论文样例”：

`signal_context -> StepPlan -> execute_agent -> dag_quality_evaluator -> reflect_agent -> validated DAG JSON -> bridge -> graph-dependent artifacts -> report_agent`

这说明以下目标已经落地：

- 五类 operator schema 已进入统一骨架
- `EXPAND` 已 runnable
- `DECISION` 已半执行并进入 report evidence
- `ml` path 已能消费 richer feature pipeline，而不只是一条 `fft -> rms` 单链

## 当前仍然暴露的短板

### 1. planner 仍是 deterministic stub，而不是方法搜索器

当前 enriched DAG 虽然已经更像论文方法，但它仍主要来自离线 stub 的固定策略，而不是：

- 基于真实 operator utility 的多轮搜索
- 基于数据反馈的 branch selection
- 基于 round gain 的结构调整

这意味着当前主链更接近“合同已打通”，还不是“方法搜索已成熟”。

### 2. bridge 仍偏单父链 feature lineage

虽然 DAG 已经有 `cross_correlation` 和 `concatenate`，但当前 `ml` bridge 真正消费的还是 `feature` 节点单链 lineage。也就是说：

- richer DAG 已进入 manifest 和 report
- 但 multi-parent lineage 还没有完整进入 compiled feature plan

### 3. `DECISION` 还只是最小 terminal rule family

`decision.threshold` 已经足以证明：

- node 能进 DAG
- 结果能写 side-output
- report 能引用

但当前还没有：

- richer decision family
- compiled decision side plan
- path-specific decision evidence policy

## 推荐的下一步顺序

### P0：先升级 bridge 的 multi-parent compiled support

优先目标：

- 让 `multi.concatenate` 和 `multi.cross_correlation` 不只是 DAG evidence
- 让它们能更明确地进入 `ml` feature plan

### P1：把 planner 从 deterministic stub 升级成 richer schema-aware planner

优先目标：

- 真正利用 `schema_category / rank_class / input_spec / output_spec`
- 在多轮 `need_patch` 中学会选择不同分支，而不是只生成固定模板

### P2：扩一小批 richer `DECISION` terminal nodes

建议只补 terminal family，不要让 `DECISION` 进入训练内环：

- `decision.threshold`
- `decision.similarity_vote`
- `decision.rule_summary`

## 当前不建议立即做的事

- 不建议先重写完整 torch trainer
- 不建议先恢复旧平台式 extra agents
- 不建议先把 provider-backed planner 变成默认主链

当前最值钱的工作仍然是：

1. 让 richer DAG 真正进入 compiled feature plan
2. 让 planner 学会使用 richer operator schema
