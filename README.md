# PHMGA Demo

This project demonstrates a minimal LangGraph-based pipeline for Predictive Health Management (PHM) using self-describing operators. It includes a small demo script and unit tests.

## Quick Start

1. Install dependencies (only NumPy and SciPy are required for the demo):
   ```bash
   pip install numpy scipy
   ```
2. Run the unit tests:
   ```bash
   pytest -q
   ```
3. Execute the demo pipeline:
   ```bash
   python -m src.phm_demo
   ```

The demo generates four random reference signals and a test signal. It then applies a plan composed of two operators (patching and mean reduction) and finally classifies the test signal based on cosine similarity.

## Graph Overview

The demo graph consists of four simple nodes:

```text
START -> planner -> execute -> decide -> report -> END
```

- **planner** – supplies the initial plan and signals.
- **execute** – processes the test and reference signals with the plan.
- **decide** – compares processed signals using a `SimilarityOperator` and chooses the closest label.
- **report** – prints the result.

This mirrors the high-level structure described in the prompt but in a condensed form for demonstration purposes.

