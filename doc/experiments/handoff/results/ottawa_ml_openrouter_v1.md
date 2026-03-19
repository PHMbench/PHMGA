# Worker Result

- worker_id: backend-comparison-owner
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: ottawa_ml_openrouter_v1
- command: `./scripts/sh/ablation/provider/ottawa_ml_openrouter.sh`
- start_time: 2026-03-19T16:35:00+08:00
- end_time: 2026-03-19T16:36:00+08:00
- provider/model: openrouter / stepfun/step-3.5-flash:free
- output_dir: `artifacts/paper/ottawa_ml_openrouter_v1`
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

Not generated. Planner failed before compile.

### feature_separability_summary

Not generated. No feature pipeline was materialized.

### progress_record

- Wrapper started with `openrouter / stepfun/step-3.5-flash:free`.
- Ottawa real-data Stage B run entered `plan_agent`.
- Provider returned explanatory text instead of a normalizable `StepPlan`.
- Repair pass also returned non-normalizable text.
- Run failed before compile and emitted no runtime artifacts.

## Metrics Summary

Unavailable. The run failed in planning.

## Failure Summary

`stepfun/step-3.5-flash:free` did not produce a planner response that could be normalized into `StepPlan` on Ottawa, even after the existing repair pass.

## Notes

This is a direct Stage B rejection signal for the active OpenRouter tuple on Ottawa.
