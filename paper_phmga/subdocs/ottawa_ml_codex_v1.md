---
experiment_id: ottawa_ml_codex_v1
dataset: Ottawa
graph_path: ml
phase: compiled
run_type: backend_comparison
provider: codex_cli
model: gpt-5.3-codex
status: reject
artifact_contract_pass: fail
feature_separability_pass: fail
selection_eligible: 'no'
output_dir: artifacts/paper/ottawa_ml_codex_v1
bundle_evidence_dir: evidence/ottawa_ml_codex_v1
---

# Summary

`ottawa_ml_codex_v1` is a `backend_comparison` run on `Ottawa` / `ml` with backend `codex_cli / gpt-5.3-codex`. Canonical status is `reject` from the result ledger.

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

- source_output_dir: `artifacts/paper/ottawa_ml_codex_v1`
- copied_artifact_count: `3`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

Not generated. Stage B never reached compile or feature materialization.

# Failure Or Pending Notes

Codex CLI transport did not return a planner result for the Ottawa full Stage B run. This is a runtime/latency blocker, not a bridge or feature-pipeline failure.

Ledger note: backend comparison candidate: codex_cli / gpt-5.3-codex; planner blocked in `codex exec`, no artifacts emitted after extended wait
