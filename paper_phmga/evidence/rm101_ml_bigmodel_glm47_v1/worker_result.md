# Worker Result

- worker_id: codex-local
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: rm101_ml_bigmodel_glm47_v1
- command: `/mnt/k/2_work/lqql_os/lqql_06_工作与项目/03_论文流水线/p02_agent_langraph/.venv/bin/python main.py +runs=rm101_ml_bigmodel_glm47_v1`
- start_time: 2026-05-04T12:53:00+08:00
- end_time: 2026-05-04T12:55:00+08:00
- provider/model: bigmodel / glm-4.7-flash
- output_dir: `artifacts/paper/rm101_ml_bigmodel_glm47_v1`
- artifact_contract_pass: fail
- feature_separability_pass: fail
- status: reject
- ledger_updated: yes

## Artifact Checklist

- validated_dag.json: no
- compiled_dag_manifest.json: no
- feature_pipeline.json: no
- feature_list.json: no
- feature_separability_summary.json: no
- artifact_index.json: no
- metrics.json: no
- final_report.md: no
- planner_normalization_trace.json: yes
- planner_raw_response.txt: yes

## Failure Summary

The run reached the ML path but failed before artifact export. The compiled feature matrix had zero columns for RM101, and `LogisticRegression` raised:

```text
ValueError: Found array with 0 feature(s) (shape=(62084, 0)) while a minimum of 1 is required by LogisticRegression.
```

This is a DAG/feature pipeline quality failure, not an API credential failure. No metrics or feature separability artifact were emitted, so the row is not selection-eligible and must not enter main tables.

## Follow-up Attempts

- `rm101_ml_bigmodel_glm47_v1_qualityfix5` used the same BigModel free model with prompt/parser/reflection quality fixes and provider backoff enabled.
- The run emitted only `planner_normalization_trace.json`, `planner_raw_response.txt`, and `planner_transport_trace.json`; no validated DAG, artifact contract, metrics, or feature separability summary were produced.
- The retry was interrupted by BigModel HTTP 429 (`1305`, model access volume too high), so it remains noncanonical evidence and does not change the rejected formal row.
- `rm101_ml_bigmodel_glm47_v1_qualityfix7` added deterministic provider fallbacks for planner/report and max-iteration reject-bundle export. It completed with a full artifact bundle at `artifacts/paper/rm101_ml_bigmodel_glm47_v1_qualityfix7`.
- qualityfix7 artifact contract: pass.
- qualityfix7 feature separability: pass (`feature_count=45`, train/val rank corr `0.7078858949864945`).
- qualityfix7 metrics: test accuracy `0.2429144385026738`, test macro_f1 `0.18934628733653974`.
- qualityfix7 remains `reject` because `workflow_state.json:path_artifacts.workflow_exit` records `max_iterations=4 reached before finish`, `compiled_for_rejection_evidence=true`, and final reflection decision `need_patch`; the report was rendered by deterministic provider fallback after a provider error.
