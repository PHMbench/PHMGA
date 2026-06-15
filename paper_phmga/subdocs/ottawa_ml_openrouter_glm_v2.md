---
experiment_id: ottawa_ml_openrouter_glm_v2
dataset: Ottawa
graph_path: ml
phase: null
run_type: backend_comparison
provider: openrouter
model: z-ai/glm-4.5-air:free
status: reject
artifact_contract_pass: fail
feature_separability_pass: fail
selection_eligible: 'no'
output_dir: null
bundle_evidence_dir: evidence/ottawa_ml_openrouter_glm_v2
---

# Summary

`ottawa_ml_openrouter_glm_v2` is a `backend_comparison` run on `Ottawa` / `ml` with backend `openrouter / z-ai/glm-4.5-air:free`. Canonical status is `reject` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `None`
- artifact_contract_pass: `fail`
- feature_separability_pass: `fail`
- selection_eligible: `no`
- worker_result_present: `yes`

# Metrics

- accuracy: unavailable
- macro_f1: unavailable

# Artifacts

- source_output_dir: `artifacts/paper/ottawa_ml_openrouter_glm_v2`
- copied_artifact_count: `4`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

Not generated. No ML artifact bundle was emitted.

# Failure Or Pending Notes

`z-ai/glm-4.5-air:free` is currently rate-limited upstream through OpenRouter free routing. This is a provider availability/rate-limit failure, not evidence that the API key is missing. The row remains rejected for the current attempt and must not enter main tables or backend selection.

Ledger note: 2026-05-04 rerun rejected: preflight passed, planner repair normalized a two-step plan, then OpenRouter free upstream returned HTTP 429 for `z-ai/glm-4.5-air:free`; no validated DAG or ML artifacts
