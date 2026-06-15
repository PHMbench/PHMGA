# Worker Result

- worker_id: backend-comparison-owner
- ticket_id: doc/experiments/handoff/06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: ottawa_ml_codex_v3
- command: `python main.py +runs=ottawa_ml_codex_v3`
- start_time: 2026-05-30T17:53Z (approximate, from `planner_transport_trace.json`)
- end_time: 2026-05-30T17:57:33Z
- provider/model: `codex_cli / gpt-5.3-codex`
- provider_note: configured provider path fell back during plan and reflection after `LLMProviderError`
- dataset: RM_017_Ottawa19
- workflow_mode: rich
- output_dir: artifacts/paper/ottawa_ml_codex_v3
- dag_depth: 4
- artifact_contract_pass: pass
- feature_separability_pass: pass
- status: needs_rerun
- ledger_updated: yes

## Artifact Checklist

- validated_dag.json: present
- compiled_dag_manifest.json: present
- feature_pipeline.json: present
- feature_list.json: present
- feature_separability_summary.json: present
- artifact_index.json: present
- metrics.json: present
- final_report.md: present

## Required Evidence

### feature_list

See `artifacts/paper/ottawa_ml_codex_v3/feature_list.json`.

### feature_separability_summary

`feature_separability_summary.json` reports:

- artifact_contract_pass: true
- feature_count: 10
- non_empty_feature_count: 10
- constant_feature_count: 0
- class_count: 3
- split_stability.train_val_rank_corr: 0.7696969696969697
- decision: pass

### progress_record

The run emitted a complete artifact bundle and reached a deterministic `finish` decision. However, it is not clean Codex backend evidence:

- `planner_transport_trace.json` records `planner_smoke` status `returncode_error`.
- The error preview reports `failed to initialize in-process app-server client: Read-only file system (os error 30)`.
- `workflow_state.json` records `provider_plan_fallback: LLMProviderError; generated deterministic normalize/fft/feature plan`.
- `workflow_state.json` and `final_report.md` record deterministic quality/report fallback after provider error.

## Metrics Summary

- train accuracy: 0.8258196721311475
- train macro-F1: 0.8248225746457779
- val accuracy: 0.8852459016393442
- val macro-F1: 0.8844618900893694
- test accuracy: 0.8770491803278688
- test macro-F1: 0.8774661249538376

## Failure Summary

This row should remain `needs_rerun` for Stage B selection. The artifact and feature gates pass, but the configured Codex provider did not cleanly produce the plan/reflection path in the sandboxed run. The row must not be used for `selected_global_best_backend` until a clean Codex provider run succeeds and is reviewed by the harness engineer.

## Notes

An unrestricted retry was requested with a separate output directory to avoid overwriting this evidence bundle, but the request was rejected because it would execute local code that may contact an external Codex provider and write new artifacts without explicit user authorization.
