# Worker Result

- worker_id: backend-comparison-owner
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: rm101_ml_codex_v1
- command: `timeout 180 ./scripts/sh/ablation/provider/rm101_ml_codex.sh`
- start_time: 2026-03-19T16:40:00+08:00
- end_time: 2026-03-19T16:43:00+08:00
- provider/model: codex_cli / gpt-5.3-codex
- output_dir: `artifacts/paper/rm101_ml_codex_v1`
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

## Required Evidence

### feature_list

Not generated. Planner transport did not complete.

### feature_separability_summary

Not generated. No runtime-native ML evidence was emitted.

### progress_record

- Wrapper started with `codex_cli / gpt-5.3-codex`.
- RM101 real-data Stage B run entered `plan_agent`.
- Run was bounded with `timeout 180` to avoid indefinite blocking.
- `codex exec` did not return a planner result within the timeout window.
- No runtime artifacts were created.

## Metrics Summary

Unavailable. The run did not reach compile or ML evaluation.

## Failure Summary

The active Codex tuple did not complete planner generation within a bounded Stage B window on RM101.

## Notes

This row is rejected for the current Stage B round. A future rerun is only justified after a transport-level latency fix or an active-set change.
