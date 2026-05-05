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

## 2026-05-05 Local Offline Probe

- command: `python main.py runtime.action=preflight +runs=rm101_ml_openrouter_glm_v2`
- result: pass
- evidence: preflight reported `dataset_name=RM_101_THU_GEARBOX`, `sample_count=240`, `source_mode=real`, `provider=openrouter`, `model=z-ai/glm-4.5-air:free`, and `credential_found=true`.
- provider run: not executed in this session because sending real-data-derived workflow context to OpenRouter requires explicit external-disclosure approval.
- local probe command: `python main.py +runs=rm101_ml_openrouter_glm_v2 llm.mode=offline_stub runtime.output_dir=artifacts/paper/rm101_ml_openrouter_glm_v2_offline_probe`
- local probe result: no artifact bundle was accepted.
- local code issue found: the first probe crashed when `execute_agent.py` tried to coerce a terminal decision side-output dict into a numeric parent array.
- local code issue disposition: `src/agents/execute_agent.py` now records an `ExecutionGap` for non-numeric side-output dict parents instead of raising `TypeError`; `tests/unit/test_execute_agent.py` includes a regression case; `python -m pytest tests/unit/test_execute_agent.py` passed with `6 passed`.
- rerun disposition: after the executor fix, the offline probe exceeded the local runtime window with repeated sklearn convergence warnings and no accepted `artifacts/paper/rm101_ml_openrouter_glm_v2_offline_probe` directory, so it was stopped and remains non-selection evidence.
- bounded rerun: `timeout 300s python main.py +runs=rm101_ml_openrouter_glm_v2 llm.mode=offline_stub runtime.max_iterations=1 runtime.output_dir=artifacts/paper/rm101_ml_openrouter_glm_v2_offline_probe_iter1` exited `124` after timeout; no accepted `artifacts/paper/rm101_ml_openrouter_glm_v2_offline_probe_iter1` directory was produced.
