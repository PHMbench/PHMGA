---
experiment_id: rm101_ml_codex_v3
dataset: RM101
graph_path: ml
phase: null
run_type: backend_comparison
provider: codex_cli
model: gpt-5.3-codex
status: pending
artifact_contract_pass: pending
feature_separability_pass: pending
selection_eligible: 'no'
output_dir: null
bundle_evidence_dir: evidence/rm101_ml_codex_v3
---

# Summary

`rm101_ml_codex_v3` is a `backend_comparison` run on `RM101` / `ml` with backend `codex_cli / gpt-5.3-codex`. Canonical status is `pending` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `None`
- artifact_contract_pass: `pending`
- feature_separability_pass: `pending`
- selection_eligible: `no`
- worker_result_present: `no`

# Metrics

- accuracy: unavailable
- macro_f1: unavailable

# Artifacts

- source_output_dir: `artifacts/paper/rm101_ml_codex_v3`
- copied_artifact_count: `0`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`

# Feature / Diagnosis Evidence

No runtime-native feature separability evidence is available in this bundle.

# Failure Or Pending Notes

Ledger note: backend comparison candidate: `codex_cli / gpt-5.3-codex`; active v3 comparison row with DAG depth target `3-8
