# PHMGA

PHMGA is a paper-oriented research scaffold for industrial time-series analysis. The repository is organized around one main chain:

`problem -> protocol -> workflow -> dag -> operators -> bridge -> model -> training -> evaluation -> report`

The current rebuild targets three graph-dependent paths:

- `dag_only`: generate a structural prior and stop at formal artifacts.
- `ml`: compile the DAG into a feature pipeline and run a lightweight ML baseline.
- `torch`: compile the DAG into a trainable build plan and run a trainable surrogate backend.

## Repository layout

- `doc/structure/`: authoritative design documents for the paper version.
- `config/`: base config plus dataset and experiment groups.
- `scripts/preflight.py`: validate environment, config, and protocol assumptions.
- `scripts/run_case.py`: single executable entrypoint for one dataset/path run.
- `src/`: minimal research core.
- `tests/unit/`, `tests/smoke/`: contract and end-to-end coverage.

## Quick start

```bash
python scripts/preflight.py --config config/config.yaml
python scripts/run_case.py --config config/config.yaml --dataset RM101 --graph-path dag_only --output-dir artifacts/rm101_dag
python scripts/run_case.py --config config/config.yaml --dataset Ottawa --graph-path ml --output-dir artifacts/ottawa_ml
python scripts/run_case.py --config config/config.yaml --dataset RM101 --graph-path torch --output-dir artifacts/rm101_torch
```

## Notes

- `PHM-Vibench metadata` is treated as the canonical data catalog contract.
- `RM101` and `Ottawa` are normalized into the same metadata and split semantics.
- The `torch` path prefers PyTorch when available and falls back to a deterministic NumPy trainer for offline smoke runs.
