# Worker Result

- worker_id: backend-comparison-owner
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: ottawa_ml_codex_v2
- command: `.venv/bin/python main.py +runs=ottawa_ml_codex_v2`
- start_time: 2026-03-21T16:33:35+08:00
- end_time: 2026-03-21T16:40:14+08:00
- provider/model: codex_cli / gpt-5.3-codex
- output_dir: `artifacts/paper/ottawa_ml_codex_v2`
- artifact_contract_pass: pass
- feature_separability_pass: pass
- status: accept
- ledger_updated: yes

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

`feature_list.json` contains 12 materialized ML features spanning both channels. Representative feature nodes include:

- `band_power_01_stft_03_ch1`
- `spectral_centroid_02_stft_03_ch1`
- `crest_factor_01_hilbert_envelope_07_ch1`
- `band_power_03_stft_04_ch2`
- `spectral_centroid_04_stft_04_ch2`
- `kurtosis_04_hilbert_envelope_08_ch2`

### feature_separability_summary

`feature_separability_summary.json` reports:

- `artifact_contract_pass: true`
- `feature_count: 12`
- `non_empty_feature_count: 12`
- `constant_feature_count: 1`
- `decision: pass`
- `top5_mean_score: 1424.930492113477`
- `train_val_rank_corr: 0.8391608391608393`

### progress_record

- Preflight passed for `Ottawa / ml / codex_cli`.
- Rich lane completed planner, execute, dag_quality, reflect, compile, and report without halt.
- Artifact contract is complete.
- `validated_dag.json` depth is 3, satisfying the `3-8` gate.
- The backend comparison round is still globally incomplete because the RM101 codex row failed and both OpenRouter rows did not clean-pass.

## Metrics Summary

- train accuracy: `0.9760928961748634`
- train macro_f1: `0.976123030065215`
- val accuracy: `0.9808743169398907`
- val macro_f1: `0.9807512441450479`
- test accuracy: `0.9262295081967213`
- test macro_f1: `0.9262164611001821`
- DAG node_count: `22`
- DAG edge_count: `20`
- DAG depth: `3`

## Failure Summary

None. This row produced a complete artifact bundle and passed the local feature separability check.

## Notes

This row is locally acceptable as a Stage B evidence bundle, but it does not unlock global backend selection by itself because the comparison round across both datasets is incomplete.
