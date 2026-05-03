---
experiment_id: rm101_ml_codex_v1
dataset: RM101
graph_path: ml
phase: compiled
run_type: backend_comparison
provider: codex_cli
model: gpt-5.3-codex
status: reject
artifact_contract_pass: fail
feature_separability_pass: fail
selection_eligible: 'no'
output_dir: artifacts/paper/rm101_ml_codex_v1
bundle_evidence_dir: evidence/rm101_ml_codex_v1
---

# Summary

`rm101_ml_codex_v1` is a `backend_comparison` run on `RM101` / `ml` with backend `codex_cli / gpt-5.3-codex`. Canonical status is `reject` from the result ledger.

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

- source_output_dir: `artifacts/paper/rm101_ml_codex_v1`
- copied_artifact_count: `0`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

Not generated. No runtime-native ML evidence was emitted.

# Failure Or Pending Notes

The active Codex tuple did not complete planner generation within a bounded Stage B window on RM101.

Ledger note: backend comparison candidate: codex_cli / gpt-5.3-codex; planner failed to return within a bounded 180s Stage B window
