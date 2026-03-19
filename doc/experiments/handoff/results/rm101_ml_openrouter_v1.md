# Worker Result

- worker_id: backend-comparison-owner
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: rm101_ml_openrouter_v1
- command: `./scripts/sh/ablation/provider/rm101_ml_openrouter.sh`
- start_time: 2026-03-19T16:39:00+08:00
- end_time: 2026-03-19T16:40:00+08:00
- provider/model: openrouter / stepfun/step-3.5-flash:free
- output_dir: `artifacts/paper/rm101_ml_openrouter_v1`
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
- RM101 real-data Stage B run entered `plan_agent`.
- Provider again returned explanatory text instead of a normalizable `StepPlan`.
- Repair pass also failed.
- Run exited before compile and emitted no runtime artifacts.

## Metrics Summary

Unavailable. The run failed in planning.

## Failure Summary

`stepfun/step-3.5-flash:free` failed the planner normalization contract on RM101 as well, matching the Ottawa failure mode.

## Notes

This second real-data failure is sufficient to keep the current active OpenRouter tuple out of backend selection.
