---
experiment_id: ottawa_ml_openrouter_nemotron_v3
dataset: Ottawa
graph_path: ml
phase: null
run_type: backend_comparison
provider: openrouter
model: nvidia/nemotron-3-super-120b-a12b:free
status: accept
artifact_contract_pass: pass
feature_separability_pass: pass
selection_eligible: 'yes'
output_dir: null
bundle_evidence_dir: evidence/ottawa_ml_openrouter_nemotron_v3
---

# Summary

`ottawa_ml_openrouter_nemotron_v3` is a `backend_comparison` run on `Ottawa` / `ml` with backend `openrouter / nvidia/nemotron-3-super-120b-a12b:free`. Canonical status is `accept` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `None`
- artifact_contract_pass: `pass`
- feature_separability_pass: `pass`
- selection_eligible: `yes`
- worker_result_present: `yes`

# Metrics

- test_accuracy: 0.8770491803278688
- test_macro_f1: 0.8774661249538376

# Artifacts

- source_output_dir: `artifacts/paper/ottawa_ml_openrouter_nemotron_v3_qualityfix1`
- copied_artifact_count: `20`
- artifact_manifest: `artifact_manifest.json`
- ledger_row: `ledger_row.json`
- worker_result: `worker_result.md`

# Feature / Diagnosis Evidence

- feature_summary_decision: `pass`
- mean_fisher_score: 766.8593127414424
- median_fisher_score: 333.72211700595693
- top5_mean_score: 1531.2390941084936
- train_val_rank_corr: 0.7696969696969697
- top_features: rms_03_fft_02_normalize_01_ch1 (4087.4461267761344), crest_factor_07_fft_02_normalize_01_ch1 (1040.5174361379484), mean_04_fft_02_normalize_01_ch1 (983.3954601160227), std_05_fft_02_normalize_01_ch1 (882.1549637448604), kurtosis_06_fft_02_normalize_01_ch1 (662.6814837675017)

# Failure Or Pending Notes

Ledger note: 2026-05-04 qualityfix1 emitted a complete artifact bundle with `nvidia/nemotron-3-super-120b-a12b:free`; artifact contract and feature separability passed, final report was provider-authored, and workflow reached finish through deterministic quality fallback after invalid reflection output
