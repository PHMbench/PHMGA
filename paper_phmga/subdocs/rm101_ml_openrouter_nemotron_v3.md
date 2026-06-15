---
experiment_id: rm101_ml_openrouter_nemotron_v3
dataset: RM101
graph_path: ml
phase: null
run_type: backend_comparison
provider: openrouter
model: nvidia/nemotron-3-super-120b-a12b:free
status: reject
artifact_contract_pass: pass
feature_separability_pass: pass
selection_eligible: 'no'
output_dir: null
bundle_evidence_dir: evidence/rm101_ml_openrouter_nemotron_v3
---

# Summary

`rm101_ml_openrouter_nemotron_v3` is a `backend_comparison` run on `RM101` / `ml` with backend `openrouter / nvidia/nemotron-3-super-120b-a12b:free`. Canonical status is `reject` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `None`
- artifact_contract_pass: `pass`
- feature_separability_pass: `pass`
- selection_eligible: `no`
- worker_result_present: `yes`

# Metrics

- test_accuracy: 0.2371657754010695
- test_macro_f1: 0.18337824193501234

# Artifacts

- source_output_dir: `artifacts/paper/rm101_ml_openrouter_nemotron_v3_qualityfix1`
- copied_artifact_count: `20`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

- feature_summary_decision: `pass`
- mean_fisher_score: 99.45506402105737
- median_fisher_score: 91.16249560361113
- top5_mean_score: 263.10232971512136
- train_val_rank_corr: 0.6148083623693381
- top_features: mean_39_fft_37_normalize_36_ch6 (348.34263975267424), rms_10_fft_09_normalize_08_ch2 (308.8796148002076), kurtosis_20_fft_16_normalize_15_ch3 (235.91929560861186), crest_factor_21_fft_16_normalize_15_ch3 (219.70048769281192), std_40_fft_37_normalize_36_ch6 (202.66961072130127)

# Failure Or Pending Notes

Ledger note: 2026-05-04 qualityfix1 emitted a complete reject-evidence bundle with `nvidia/nemotron-3-super-120b-a12b:free`; artifact/feature gates passed but workflow_exit shows max_iterations reached before finish with last_reflection_decision=need_replan, so it is not selection-eligible
