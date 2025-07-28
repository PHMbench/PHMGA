# PHMGA

This repository demonstrates a minimal Prognostics and Health Management (PHM) workflow implemented with [LangGraph](https://github.com/langchain-ai/langgraph).

The code defines a small set of self–describing operators based on Pydantic. Each operator exposes its parameters and an `execute` method so a planner can build processing plans dynamically.

## Demo Pipeline

```
Planner -> Tool Executor -> Decision -> Report
```

The demo shows how to generate random reference signals, run a simple plan (patch and mean) on both the test and reference data, and classify the test signal using cosine similarity.

### Running

```bash
pip install -r requirements.txt
python -m src.phm_demo
```

This will print a predicted label for the test signal.

### Extending

- Implement additional operators in `src/tools/functions.py` and `src/tools/schemas.py`.
- Build more complex graphs using `langgraph` and the provided operators.

