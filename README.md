# PHMGA

## Environment

- Copy `.env.example` to `.env` and fill the API key for your chosen provider.
- Select provider via `LLM_PROVIDER`:
  - `gemini` (default)
  - `openai_compatible` (OpenAI-compatible gateways)
  - `deepseek`
  - `glm`
- TSPN training requires PyTorch. This container may not have `torch` installed; run training in your local env (e.g. `conda activate LQ_signal`) with PyTorch available.

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
