# Worker Result

- worker_id: manual-simple-qualification
- ticket_id: n/a (qualification lane)
- run_type: simple_qualification
- experiment_id: ottawa_ml_codex_simple
- command: `.venv/bin/python main.py +runs=ottawa_ml_codex_simple`
- start_time: 2026-03-22T10:02:03.557456+08:00
- end_time: 2026-03-22T10:04:18.556630+08:00
- provider/model: codex_cli / gpt-5.3-codex
- dataset: RM_017_Ottawa19
- workflow_mode: simple_fullchain
- output_dir: `artifacts/simple/ottawa_ml_codex_simple`
- dag_depth: `3`
- artifact_contract_pass: pass
- feature_separability_pass: n/a
- status: accept
- ledger_updated: no

## Artifact Checklist

- validated_dag.json: yes
- compiled_dag_manifest.json: yes
- feature_pipeline.json: yes
- feature_list.json: yes
- feature_separability_summary.json: yes
- artifact_index.json: yes
- metrics.json: yes
- final_report.md: yes

## Required Evidence

### feature_list

`feature_list.json` materialized 16 ML features. Representative nodes include:

- `rms_01_hilbert_envelope_07_ch1`
- `rms_09_normalize_01_ch1`
- `spectral_centroid_03_stft_03_ch1`
- `kurtosis_03_stft_03_ch1`
- `spectral_centroid_04_stft_04_ch2`

### feature_separability_summary

`feature_separability_summary.json` reports:

- `artifact_contract_pass: true`
- `feature_count: 16`
- `non_empty_feature_count: 16`
- `constant_feature_count: 4`
- `decision: pass`
- `top5_mean_score: 159.97564240137947`
- `train_val_rank_corr: 0.7323529411764707`

### progress_record

- Preflight passed for `Ottawa / ml / codex_cli / simple_fullchain`.
- The simple lane completed `plan -> execute -> reflect -> compile -> inquirer -> report`.
- `workflow_state.json` records `status=reported`, `halt_reason=null`, and reflection decisions `need_patch -> need_patch -> finish`.
- No `dag_quality_summary.json` or dataset-level runtime trace was written, which is expected for `simple_fullchain`.
- This run qualifies the current branch as capable of real-data runtime closure on Ottawa without entering formal paper ledger flow.

## Metrics Summary

- train accuracy: `1.0`
- train macro_f1: `1.0`
- val accuracy: `1.0`
- val macro_f1: `1.0`
- test accuracy: `0.6666666666666666`
- test macro_f1: `0.5555555555555555`
- DAG node_count: `26`
- DAG edge_count: `24`
- DAG depth: `3`

## Failure Summary

None. This qualification run produced a complete simple-lane artifact bundle on real Ottawa data.

## Notes

This row is qualification evidence only. It does not update `01_result_ledger.md`, does not enter `02_main_tables.md`, and does not affect `selected_global_best_backend`.

A later live rerun with a fresh output override halted instead of reproducing the same bundle, so runbook status for this preset should be treated as `pass_with_local_incident`, not stable formal pass.
