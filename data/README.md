# Data Protocol

This repository treats the data unit as a **window-level sample**.
`src/data/protocol.py` owns split assignment and window slicing, and
`src/data/dataset_preparer.py` consumes the already-sliced window samples
directly.

## Window Sample Contract

`materialize_split_signals(protocol)` returns:

- `train`: `List[SignalRecord]`
- `val`: `List[SignalRecord]`
- `test`: `List[SignalRecord]`

Each `SignalRecord` is a single window sample with:

- `source_sample_id`
- `window_index`
- `window_id`
- `split`
- `label`
- `window`

Compatibility notes:

- `record.sample_id` is a backward-compatible alias for `record.window_id`
- `record.windows` is a backward-compatible singleton view: `[record.window]`
- downstream runners should consume `record.window` and `record.window_id`

## Tensor Shape

The tensor shape of one window sample is:

```text
(channels, window_size)
```

The split containers themselves are lists of window samples, not stacked
tensors. If you want a batched tensor, it is created downstream by the
dataset preparer or runner.

## Current Formal Main Dimensions

The current formal main presets use sliding windows and provider-backed LLM
for `main` runs.

### RM101

- Dataset: `RM_101_THU_GEARBOX`
- Window shape per sample: `(8, 4096)`
- `train`: `62,084` window samples
- `val`: `12,716` window samples
- `test`: `14,960` window samples
- Equivalent source-sample counts in the current main split:
  - `train`: `166`
  - `val`: `34`
  - `test`: `40`

### Ottawa

- Dataset: `RM_017_Ottawa19`
- Window shape per sample: `(2, 32768)`
- `train`: `2,928` window samples
- `val`: `732` window samples
- `test`: `732` window samples
- Equivalent source-sample counts in the current main split:
  - `train`: `24`
  - `val`: `6`
  - `test`: `6`

## Pilot Presets

The pilot presets use `slice_mode=centered`, so each source sample materializes
as one window sample. Pilot runs are therefore useful for smoke testing the
full workflow, but they are not a substitute for the sliding-window formal
main runs above.

## Practical Rule

- `protocol` cuts windows
- `dataset_preparer` materializes window samples into rows
- `runner` consumes the already-materialized rows
- the original sample provenance is preserved via `source_sample_id`

