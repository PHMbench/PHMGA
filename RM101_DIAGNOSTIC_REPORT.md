# RM101 Diagnostic Report

Source run:

- `openrouter / google/gemini-2.5-pro (simulated, all-domain mixed)`
- Source directory: `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1`

Reference tables:

- `artifacts/paper/rm101_all_domains_mixed_simulated_v1/rm101_final_accuracy_table.md`
- `artifacts/paper/rm101_all_domains_mixed_simulated_v1/dag_summary_table.md`
- `artifacts/paper/rm101_all_domains_mixed_simulated_v1/cross_dag_late_fusion.md`

## Automated Selection

- Selection basis: `val_macro_f1`
- Best overall run: `Gemini 2.5 Pro (simulated, all-domain mixed)`
- Best single leaf: `torque_normalize_24_rms_10_stft_06_ch6`
- Final choice: `best_single_leaf`
- Best single validation macro-F1: `0.817153331109743`
- Best single test accuracy: `0.8865248226950354`
- Best single test macro-F1: `0.8864772952124089`
- Weighted ensemble test accuracy: `0.5818927304964538`
- Weighted ensemble test macro-F1: `0.5542699641108251`

For reference, the strongest real baseline in the same all-domain mixed setting is:

- `BigModel / GLM-4.7-FlashX (real baseline)`
- Best single test accuracy: `0.8140514184397163`
- Best single test macro-F1: `0.8229110215858482`

## DAG Summary

- Depth: `5`
- Node count: `52`
- Edge count: `56`
- Leaf count: `21`
- Unique ops:
  - `coherence`
  - `crest_factor`
  - `fft`
  - `kurtosis`
  - `order_band_energy`
  - `order_track_resample`
  - `patch`
  - `psd`
  - `rms`
  - `sideband_ratio`
  - `spectral_centroid`
  - `stft`
  - `torque_normalize`
  - `tsa_cycle_average`

Best DAG figure:

![RM101 Best DAG](artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1/graphs/dag.png)

DAG artifact paths:

- `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1/graphs/dag.png`
- `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1/graphs/dag.json`
- `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1/dag_summary.json`

## Evaluation Protocol

- Dataset: `RM101`
- Split protocol: `all 12 domains mixed-domain stratified split`
- Split-before-windowing: `true`
- Window configuration: `window_size=4096`, `stride=4096`, `slice_mode=sliding`, `drop_last_window=false`
- Original samples after filtering: `240`
- Train / val / test ids: `146 / 46 / 48`
- Train / val / test windows: `27448 / 8648 / 9024`
- Domain scope: `Domain_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`
- Invalid labels removed: `true`

Per-class test window counts:

- `Label 0`: `564`
- `Label 1`: `1316`
- `Label 2`: `1316`
- `Label 3`: `564`
- `Label 4`: `1316`
- `Label 5`: `1316`
- `Label 6`: `1316`
- `Label 7`: `1316`

## Channel Aliases

- `ch1 = speed_key_phase`
- `ch2 = torque`
- `ch3 = motor_vibration_x`
- `ch4 = motor_vibration_y`
- `ch5 = motor_vibration_z`
- `ch6 = gearbox_vibration_x`
- `ch7 = gearbox_vibration_y`
- `ch8 = gearbox_vibration_z`

Interpretation:

- `speed_key_phase` and `torque` are used as side-inputs for speed-aware and load-aware operators.
- Primary diagnosis still comes from vibration branches, especially gearbox vibration channels `ch6-8`.

## Best Branch Interpretation

Selected best leaf:

- `torque_normalize_24_rms_10_stft_06_ch6`
- Branch summary: `stft -> rms -> torque_normalize`
- Channel focus: `ch6 = gearbox_vibration_x`
- Chosen algorithm: `MLP`
- Feature dimension: `129`

Why this branch is strong:

- `stft` preserves localized time-frequency structure under variable speed.
- `rms` compresses the time-frequency representation into a stable energy descriptor.
- `torque_normalize` reduces load-induced amplitude drift using `ch2 = torque`.
- The final feature remains compact enough for robust shallow ML while preserving operating-condition sensitivity.

## Top Candidate Leaves

| leaf_id | branch_summary | algorithm | feature_dim | val_macro_f1 | test_accuracy | test_macro_f1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `torque_normalize_24_rms_10_stft_06_ch6` | `stft -> rms -> torque_normalize` | `MLP` | 129 | 0.8172 | 0.8865 | 0.8865 |
| `crest_factor_15_patch_03_fft_08_ch8` | `fft -> patch -> crest_factor` | `SVM` | 64 | 0.4619 | 0.4497 | 0.4397 |
| `crest_factor_12_stft_08_ch8` | `stft -> crest_factor` | `SVM` | 129 | 0.4551 | 0.4251 | 0.4441 |
| `kurtosis_11_stft_07_ch7` | `stft -> kurtosis` | `SVM` | 129 | 0.4085 | 0.3473 | 0.3682 |
| `rms_13_patch_01_fft_tsa_06_tsa_06_ch6` | `tsa_cycle_average -> fft -> patch -> rms` | `RandomForest` | 4 | 0.3532 | 0.3703 | 0.3267 |

Observations:

- The top leaf is clearly separated from the rest by validation macro-F1.
- `stft`-based branches dominate the top-ranked leaves.
- `patch` and `tsa_cycle_average` contribute useful alternatives, but they do not surpass the `stft -> rms -> torque_normalize` branch.
- `order_track_resample -> order_band_energy` is informative, but weaker than the strongest `stft` branch in this split.

## Comparison Against Other Runs

Main comparison set:

| run | test_accuracy | test_macro_f1 | final_choice |
| --- | ---: | ---: | --- |
| `BigModel / GLM-4.7-FlashX (real baseline)` | 0.8141 | 0.8229 | `best_single_leaf` |
| `Gemini 2.0 Flash (simulated, all-domain mixed)` | 0.1539 | 0.1305 | `weighted_ensemble` |
| `Gemini 2.5 Flash (simulated, all-domain mixed)` | 0.2585 | 0.2525 | `best_single_leaf` |
| `Gemini 2.5 Pro (simulated, all-domain mixed)` | 0.8865 | 0.8865 | `best_single_leaf` |

Cross-DAG late fusion:

- Selection basis: `val_macro_f1`
- Fusion method: `weighted_vote`
- Test accuracy: `0.8241356382978723`
- Test macro-F1: `0.8298924465603508`

Interpretation:

- `Gemini 2.5 Pro (simulated)` is the strongest single run.
- Cross-DAG late fusion improves over weaker runs, but it does not exceed the best `2.5-pro` single-branch result.
- The real baseline remains strong, but the richer simulated `2.5-pro` DAG yields the highest test-set diagnosis score in this experiment set.

## Final Diagnostic Conclusion

Recommended RM101 diagnosis result for reporting:

- Primary result: `Gemini 2.5 Pro (simulated, all-domain mixed)`
- Recommended prediction source: `best_single_leaf`
- Final reported test accuracy: `0.8865`
- Final reported test macro-F1: `0.8865`
- Recommended branch: `stft -> rms -> torque_normalize` on `gearbox_vibration_x (ch6)` with `torque (ch2)` as side-input

Methodological note:

- This result is based on a `simulated planner variant`, not an online Gemini API run.
- The evaluation protocol is a mixed-domain window-level setting, not a leave-one-domain-out generalization benchmark.

## Artifact References

Primary run artifacts:

- `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1/final_report.md`
- `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1/final_selection.json`
- `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1/node_metrics.csv`
- `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1/protocol_summary.json`

Paper bundle:

- `artifacts/paper/rm101_all_domains_mixed_simulated_v1/rm101_final_accuracy_table.md`
- `artifacts/paper/rm101_all_domains_mixed_simulated_v1/dag_summary_table.md`
- `artifacts/paper/rm101_all_domains_mixed_simulated_v1/node_level_results.md`
- `artifacts/paper/rm101_all_domains_mixed_simulated_v1/cross_dag_late_fusion.md`
- `artifacts/paper/rm101_all_domains_mixed_simulated_v1/analysis.md`
