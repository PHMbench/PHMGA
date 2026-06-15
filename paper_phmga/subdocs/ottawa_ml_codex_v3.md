---
experiment_id: ottawa_ml_codex_v3
dataset: Ottawa
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
bundle_evidence_dir: evidence/ottawa_ml_codex_v3
---

# Summary

`ottawa_ml_codex_v3` is a `backend_comparison` run on `Ottawa` / `ml` with backend `codex_cli / gpt-5.3-codex`. Canonical status is `pending` from the result ledger.

# Status

- stage: `stage_b_backend_comparison`
- phase: `None`
- artifact_contract_pass: `pending`
- feature_separability_pass: `pending`
- selection_eligible: `no`
- worker_result_present: `yes`

# Metrics

- test_accuracy: 0.8770491803278688
- test_macro_f1: 0.8774661249538376

# Artifacts

- source_output_dir: `artifacts/paper/ottawa_ml_codex_v3`
- copied_artifact_count: `19`
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

This row should remain `needs_rerun` for Stage B selection. The artifact and feature gates pass, but the configured Codex provider did not cleanly produce the plan/reflection path in the sandboxed run. The row must not be used for `selected_global_best_backend` until a clean Codex provider run succeeds and is reviewed by the harness engineer.

Ledger note: 2026-05-30 sandboxed run emitted complete artifacts and feature separability pass evidence, but `planner_transport_trace.json` and `workflow_state.json` show Codex provider fallback after `LLMProviderError` / read-only filesystem app-server initialization failure; treat as `needs_rerun`, not selection-eligible, until a clean Codex provider run is authorized and reviewed
