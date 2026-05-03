---
experiment_id: rm101_ml_openrouter_v1
dataset: RM101
graph_path: ml
phase: compiled
run_type: backend_comparison
provider: openrouter
model: stepfun/step-3.5-flash:free
status: reject
artifact_contract_pass: fail
feature_separability_pass: fail
selection_eligible: 'no'
output_dir: artifacts/paper/rm101_ml_openrouter_v1
bundle_evidence_dir: evidence/rm101_ml_openrouter_v1
---

# Summary

`rm101_ml_openrouter_v1` is a `backend_comparison` run on `RM101` / `ml` with backend `openrouter / stepfun/step-3.5-flash:free`. Canonical status is `reject` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `compiled`
- artifact_contract_pass: `fail`
- feature_separability_pass: `fail`
- selection_eligible: `no`
- worker_result_present: `yes`

# Metrics

- accuracy: unavailable
- macro_f1: unavailable

# Artifacts

- source_output_dir: `artifacts/paper/rm101_ml_openrouter_v1`
- copied_artifact_count: `0`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

Not generated. No feature pipeline was materialized.

# Failure Or Pending Notes

`stepfun/step-3.5-flash:free` failed the planner normalization contract on RM101 as well, matching the Ottawa failure mode.

Ledger note: historical comparison failure: openrouter / stepfun/step-3.5-flash:free; planner output was not normalizable into `StepPlan` after repair; not in current active Stage B set
