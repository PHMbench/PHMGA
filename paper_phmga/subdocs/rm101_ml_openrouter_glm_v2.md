---
experiment_id: rm101_ml_openrouter_glm_v2
dataset: RM101
graph_path: ml
phase: null
run_type: backend_comparison
provider: openrouter
model: z-ai/glm-4.5-air:free
status: pending
artifact_contract_pass: pending
feature_separability_pass: pending
selection_eligible: 'no'
output_dir: null
bundle_evidence_dir: evidence/rm101_ml_openrouter_glm_v2
---

# Summary

`rm101_ml_openrouter_glm_v2` is a `backend_comparison` run on `RM101` / `ml` with backend `openrouter / z-ai/glm-4.5-air:free`. Canonical status is `pending` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `None`
- artifact_contract_pass: `pending`
- feature_separability_pass: `pending`
- selection_eligible: `no`
- worker_result_present: `yes`

# Metrics

- accuracy: unavailable
- macro_f1: unavailable

# Artifacts

- source_output_dir: `artifacts/paper/rm101_ml_openrouter_glm_v2`
- copied_artifact_count: `0`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

Not generated. No runtime-native ML evidence was emitted.

# Failure Or Pending Notes

The RM101 OpenRouter formal row failed at the provider boundary with `429 Too Many Requests`. This is an upstream rate-limit failure, not a local proxy refusal and not a schema-normalization failure.

Ledger note: backend comparison candidate: `openrouter / z-ai/glm-4.5-air:free`; active free-model comparison row with DAG depth target `3-8
