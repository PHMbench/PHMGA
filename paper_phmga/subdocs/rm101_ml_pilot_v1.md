---
experiment_id: rm101_ml_pilot_v1
dataset: RM101
graph_path: ml
phase: compiled
run_type: pilot
provider: offline_stub
model: n/a
status: accept
artifact_contract_pass: n/a
feature_separability_pass: n/a
selection_eligible: n/a
output_dir: artifacts/paper/rm101_ml_pilot_v1
bundle_evidence_dir: evidence/rm101_ml_pilot_v1
---

# Summary

`rm101_ml_pilot_v1` is a `pilot` run on `RM101` / `ml` with backend `offline_stub / n/a`. Canonical status is `accept` from the result ledger.

# Status

- stage: `stage_a_pilot`
- phase: `compiled`
- artifact_contract_pass: `n/a`
- feature_separability_pass: `n/a`
- selection_eligible: `n/a`
- worker_result_present: `no`

# Metrics

- test_accuracy: 0.25
- test_macro_f1: 0.125

# Artifacts

- source_output_dir: `artifacts/paper/rm101_ml_pilot_v1`
- copied_artifact_count: `15`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- validated_dag_alias: `dag.json`

# Feature / Diagnosis Evidence

No runtime-native feature separability evidence is available in this bundle.

# Failure Or Pending Notes

Ledger note: pilot smoke (offline_stub)
