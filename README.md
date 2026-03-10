# PHMGA

## Environment

- Copy `.env.example` to `.env` and fill the API key for your chosen provider.
- Select provider via `LLM_PROVIDER`:
  - `gemini` (default)
  - `openai_compatible` (OpenAI-compatible gateways)
  - `deepseek`
  - `glm`
- TSPN training requires PyTorch. This container may not have `torch` installed; run training in your local env (e.g. `conda activate LQ_signal`) with PyTorch available.

## built_state save mode

`built_state.pkl` now supports configurable save modes through `data.state_save_mode`:

- `auto` (default): mode is selected by source mode
  - `source_mode=vibench` -> `minimal`
  - `source_mode=fixed_ids` -> `full`
- `full`: persist full in-memory state, including root/processed arrays in node `results`.
- `minimal`: persist topology + metadata only; node arrays are stripped.

`save_state` writes companion files:

- `built_state.pkl.sha256`: checksum sidecar
- `built_state.pkl.meta.json`: save metadata, including `effective_mode`, `source_mode`, `numpy_bytes_before`, `numpy_bytes_after`, reduction ratio, and node/channel counts.

Troubleshooting:

- If you run `fixed_ids` training with a minimal snapshot, training will fail fast with an actionable error.
- Fix by setting `data.state_save_mode=full` and rebuilding (or deleting old `built_state.pkl` so it is regenerated).

## Operator VRAM budget (rm101_closed_v1)

This section provides a practical VRAM budget per operator for TSPN closed-world runs.

Assumptions (fixed baseline):
- dtype: `float32`
- input shape per operator call: `B=64, L=4096, C=8`
- base activation size `A = B * L * C * 4 bytes = 8 MiB`
- numbers below are **forward peak estimate** (engineering budget), not exact profiler peaks
- training recommendation: reserve about `2.5x` of forward peak for autograd + buffers
- current container is CPU-only (`torch.cuda.is_available() = False`), so this is formula-based budgeting

### Layer operators (8 / rm101_closed_v1)

| operator | TSPN token | forward peak est. (MiB) | training reserve est. (MiB) | note |
|---|---|---:|---:|---|
| `normalize` | `NORM` | 8 | 20 | reduction + broadcast |
| `integrate` | `INT` | 8 | 20 | `cumsum` |
| `detrend` | `DT` | 16 | 40 | trend tensor + output |
| `differentiate` | `DIFF` | 16 | 40 | diff tensor + concat |
| `fft` | `FFT` | 20 | 50 | `rfft` complex + interpolation |
| `filter` | `WF` | 24 | 60 | freq-domain filter (`rfft/irfft`) |
| `stft` | `STFT` | 25 | 63 | `stft` complex spectrogram + resize |
| `hilbert_envelope` | `HT` | 40 | 100 | FFT + analytic signal (complex) |

### Feature operators (19 / rm101_closed_v1)

| operator | forward peak est. (MiB) | training reserve est. (MiB) | note |
|---|---:|---:|---|
| `mean` | 8 | 20 | time-domain reduction |
| `std` | 8 | 20 | time-domain reduction |
| `var` | 8 | 20 | time-domain reduction |
| `max` | 8 | 20 | time-domain reduction |
| `min` | 8 | 20 | time-domain reduction |
| `peak_to_peak` | 8 | 20 | max-min |
| `abs_mean` | 16 | 40 | abs temp + reduction |
| `rms` | 16 | 40 | square temp + reduction |
| `entropy` | 16 | 40 | softmax/log_softmax buffers |
| `clearance_factor` | 16 | 40 | peak + abs-mean path |
| `crest_factor` | 16 | 40 | peak + rms path |
| `shape_factor` | 16 | 40 | rms + abs-mean path |
| `zero_crossing_rate` | 16 | 40 | sign + diff |
| `skew` | 24 | 60 | 2nd/3rd-order moments |
| `kurtosis` | 24 | 60 | 2nd/4th-order moments |
| `spectral_centroid` | 20 | 50 | `rfft` magnitude path |
| `spectral_flatness` | 24 | 60 | `rfft` + log/geometric mean |
| `spectral_kurtosis` | 24 | 60 | spectral moments |
| `spectral_skewness` | 24 | 60 | spectral moments |

### Non-contract operators (31): closed-world `N/A`

These operators are not allowed by `operator_contract=rm101_closed_v1` and therefore do not enter TSPN compile/training VRAM budgeting in closed-world mode.

- layer proxy (10): `cepstrum`, `denoise_wavelet`, `mel_spectrogram`, `patch`, `power_to_db`, `psd`, `resample`, `savgol_filter`, `spectrogram`, `wavelet_transform`
- layer unsupported (17): `arithmetic`, `coherence`, `concatenate`, `convolution`, `cross_correlation`, `distance`, `dtw_distance`, `element_wise_product`, `emd`, `pca`, `phase_difference`, `subtract`, `time_delay_embedding`, `transfer_function`, `vmd`, `vqt`, `wigner_ville_distribution`
- feature proxy/unsupported (4): `hjorth_parameters`, `approximate_entropy`, `band_power`, `permutation_entropy`

### How to convert to your run settings

For quick scaling from this table:
- activation scales approximately linearly with `B * L * C`
- `new_forward_est ~= table_forward * (B/64) * (L/4096) * (C/8)`
- for mixed precision, multiply by ~`0.5`
- for training reserve, multiply forward estimate by `2.0 ~ 3.0` depending on optimizer/checkpointing

## Utility scripts (NVTA workflow)

- Export node datasets from a saved state: `python scripts/export_node_datasets.py --state <state.pkl>`
- Train shallow ML from exported datasets: `python scripts/train_shallow_ml_from_npz.py --dataset-dir <dir>`
- Generate final report from state (+ optional ML results): `python scripts/generate_report_from_state.py --state <state.pkl> --ml-results <ml.pkl>`
- Manual validation guide: `doc/plan/2_2/MANUAL_TEST.md`



# TODO
## signal processing discovering
## database
## deep research + plan
## state map 重构
## save log


# cases
## case 1: DAG first

## case 2: DAG with data feature reflection

## case 3: decouple DAG and data

## case 4: Coding augmentation

## case 5: prompt evaluation

## case 6: multi-agent 并发

## 
