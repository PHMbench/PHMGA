---
experiment_id: ottawa_ml_bigmodel_glm47_v1
dataset: Ottawa
graph_path: ml
phase: null
run_type: backend_comparison
provider: bigmodel
model: glm-4.7-flash
status: accept
artifact_contract_pass: pass
feature_separability_pass: pass
selection_eligible: 'yes'
output_dir: null
bundle_evidence_dir: evidence/ottawa_ml_bigmodel_glm47_v1
---

# Summary

`ottawa_ml_bigmodel_glm47_v1` is a `backend_comparison` run on `Ottawa` / `ml` with backend `bigmodel / glm-4.7-flash`. Canonical status is `accept` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `None`
- artifact_contract_pass: `pass`
- feature_separability_pass: `pass`
- selection_eligible: `yes`
- worker_result_present: `yes`

# Metrics

- test_accuracy: 0.7254098360655737
- test_macro_f1: 0.7226535613558728

# Artifacts

- source_output_dir: `artifacts/paper/ottawa_ml_bigmodel_glm47_v1_qualityfix2`
- copied_artifact_count: `22`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

- feature_summary_decision: `pass`
- mean_fisher_score: 1340.7417654646351
- median_fisher_score: 650.8907810987932
- top5_mean_score: 2373.175763466761
- train_val_rank_corr: 0.9491525423728814
- top_features: std_03_normalize_01_ch1 (4023.785510615924), rms_01_normalize_01_ch1 (4023.7855106158795), kurtosis_02_ch1 (1583.708507501605), kurtosis_04_normalize_01_ch1 (1583.708507501604), rms_01_ch1 (650.8907810987932)

# Failure Or Pending Notes

Ledger note: 2026-05-04 qualityfix2 emitted complete artifact bundle with `glm-4.7-flash`; DAG quality finish_candidate, feature separability pass, and test macro_f1 0.7226535613558728; final report used deterministic provider fallback after report-stage provider error
