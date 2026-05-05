# Worker Result

- worker_id: codex-local
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: ottawa_ml_openrouter_glm_v2
- command: `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy /mnt/k/2_work/lqql_os/lqql_06_工作与项目/03_论文流水线/p02_agent_langraph/.venv/bin/python main.py +runs=ottawa_ml_openrouter_glm_v2`
- start_time: 2026-05-04T12:37:00+08:00
- end_time: 2026-05-04T12:39:00+08:00
- provider/model: openrouter / z-ai/glm-4.5-air:free
- output_dir: `artifacts/paper/ottawa_ml_openrouter_glm_v2`
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
- planner_transport_trace.json: yes

## Required Evidence

### feature_list

Not generated. The run failed in planner normalization before compile or feature materialization.

### feature_separability_summary

Not generated. No ML artifact bundle was emitted.

### progress_record

- Preflight passed for `Ottawa / ml / openrouter` on `/mnt/k/D01_vibench`.
- Initial planner response did not contain strict StepPlan JSON and was recorded in `planner_normalization_trace.json`.
- Repair produced a normalized two-step plan.
- The subsequent planner call failed with HTTP 429 from OpenRouter upstream for `z-ai/glm-4.5-air:free`.
- No validated DAG, compiled artifacts, metrics, or final report were emitted.

## Metrics Summary

Unavailable. The run never reached compile or ML evaluation.

## Failure Summary

`z-ai/glm-4.5-air:free` is currently rate-limited upstream through OpenRouter free routing. This is a provider availability/rate-limit failure, not evidence that the API key is missing. The row remains rejected for the current attempt and must not enter main tables or backend selection.

## Notes

The provider trace redacts provider user identifiers and does not contain API keys.
