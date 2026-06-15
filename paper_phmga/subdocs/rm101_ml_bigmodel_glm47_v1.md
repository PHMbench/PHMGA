---
experiment_id: rm101_ml_bigmodel_glm47_v1
dataset: RM101
graph_path: ml
phase: null
run_type: backend_comparison
provider: bigmodel
model: glm-4.7-flash
status: reject
artifact_contract_pass: pass
feature_separability_pass: pass
selection_eligible: 'no'
output_dir: null
bundle_evidence_dir: evidence/rm101_ml_bigmodel_glm47_v1
---

# Summary

`rm101_ml_bigmodel_glm47_v1` is a `backend_comparison` run on `RM101` / `ml` with backend `bigmodel / glm-4.7-flash`. Canonical status is `reject` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `None`
- artifact_contract_pass: `pass`
- feature_separability_pass: `pass`
- selection_eligible: `no`
- worker_result_present: `yes`

# Metrics

- test_accuracy: 0.2429144385026738
- test_macro_f1: 0.18934628733653974

# Artifacts

- source_output_dir: `artifacts/paper/rm101_ml_bigmodel_glm47_v1_qualityfix7`
- copied_artifact_count: `22`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

- feature_summary_decision: `pass`
- mean_fisher_score: 90.87403970660726
- median_fisher_score: 67.19634247917953
- top5_mean_score: 263.10232971512136
- train_val_rank_corr: 0.7078858949864945
- top_features: mean_39_fft_37_normalize_36_ch6 (348.34263975267424), rms_10_fft_09_normalize_08_ch2 (308.8796148002076), kurtosis_20_fft_16_normalize_15_ch3 (235.91929560861186), crest_factor_21_fft_16_normalize_15_ch3 (219.70048769281192), std_40_fft_37_normalize_36_ch6 (202.66961072130127)

# Failure Or Pending Notes

The run reached the ML path but failed before artifact export. The compiled feature matrix had zero columns for RM101, and `LogisticRegression` raised:

```text
ValueError: Found array with 0 feature(s) (shape=(62084, 0)) while a minimum of 1 is required by LogisticRegression.
```

This is a DAG/feature pipeline quality failure, not an API credential failure. No metrics or feature separability artifact were emitted, so the row is not selection-eligible and must not enter main tables.

Ledger note: 2026-05-04 qualityfix7 emitted a complete reject-evidence bundle with deterministic provider fallbacks; workflow_exit shows max_iterations reached before finish with last_reflection_decision=need_patch, so it is not selection-eligible despite artifact/feature gates passing
