# Worker Result

- worker_id: backend-comparison-owner
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: ottawa_ml_openrouter_glm_v2
- command: `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy .venv/bin/python main.py +runs=ottawa_ml_openrouter_glm_v2`
- start_time: 2026-03-21T17:05:07+08:00
- end_time: 2026-03-21T17:07:44+08:00
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

## Required Evidence

### feature_list

Not generated. The run failed in planner normalization before compile or feature materialization.

### feature_separability_summary

Not generated. No ML artifact bundle was emitted.

### progress_record

- Preflight passed for `Ottawa / ml / openrouter`.
- Initial planner call normalized a single-step response: `hilbert_envelope` on `ch1`.
- A later planner round failed to satisfy the strict `StepPlan` contract after reflection asked for additional depth and symmetry.
- Repair also returned prose instead of strict JSON.
- The run terminated in `plan_agent`; no validated DAG or downstream artifacts were emitted.

## Metrics Summary

Unavailable. The run never reached compile or ML evaluation.

## Failure Summary

`z-ai/glm-4.5-air:free` failed the planner normalization contract on a later rich-lane iteration. The first round produced a usable one-step plan, but the backend then answered with prose describing the DAG state instead of returning strict `{"plan":[...]}` JSON, and the repair pass also failed.

## Notes

This row is rejected for the current Stage B round. The failure mode is schema/repair instability under iterative rich-lane planning, not transport refusal.
