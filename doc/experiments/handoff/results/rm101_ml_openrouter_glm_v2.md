# Worker Result

- worker_id: backend-comparison-owner
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: rm101_ml_openrouter_glm_v2
- command: `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy .venv/bin/python main.py +runs=rm101_ml_openrouter_glm_v2`
- start_time: 2026-03-21T17:08:20+08:00
- end_time: 2026-03-21T17:08:20+08:00
- provider/model: openrouter / z-ai/glm-4.5-air:free
- output_dir: `artifacts/paper/rm101_ml_openrouter_glm_v2`
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

Not generated. The run failed before planner output materialized.

### feature_separability_summary

Not generated. No runtime-native ML evidence was emitted.

### progress_record

- Preflight passed for `RM101 / ml / openrouter`.
- The first planner request failed at provider transport with HTTP 429.
- The error message explicitly states that `z-ai/glm-4.5-air:free` was temporarily rate-limited upstream by Z.AI.
- Only `planner_transport_trace.json` was written.

## Metrics Summary

Unavailable. The run did not reach compile or ML evaluation.

## Failure Summary

The RM101 OpenRouter formal row failed at the provider boundary with `429 Too Many Requests`. This is an upstream rate-limit failure, not a local proxy refusal and not a schema-normalization failure.

## Notes

This row is rejected for the current Stage B round. Any future rerun must account for upstream rate limits on `z-ai/glm-4.5-air:free`.
