# OpenRouter StepFun Incident Note

**Date**: 2026-03-18
**Provider**: `openrouter`
**Model**: `stepfun/step-3.5-flash:free`

## Status: Fixed In Client, Still Requires Backend-Comparison Validation

The original failure was not a provider outage. The issue was that PHMGA treated
`stepfun/step-3.5-flash:free` as a JSON-mode structured model, while real
OpenRouter payloads could place useful text in `reasoning` or other text fields.
The client has since been updated to route this model through text-mode
structured extraction.

## Symptom Observed Before Fix

Provider runs consistently fail with:

```text
LLMSchemaError: Provider response did not contain a valid JSON object.
```

Typical failure mode:

```
content = null
reasoning = "...{ valid JSON object or JSON-like text }..."
```

## Root Cause

The root cause was a client-side transport assumption:

- structured calls were treated as equivalent to `response_format=json_object`
- `stepfun/step-3.5-flash:free` did not reliably satisfy that assumption
- valid structured text could still exist in `reasoning` / text fallback fields

## Fix Landed In `src/llm/client.py`

The provider client now does three things:

1. capability-aware routing:
   - JSON-friendly models can still use JSON mode
   - `stepfun/step-3.5-flash:free` is treated as `text_mode`
2. text extraction fallback:
   - `content`
   - `reasoning`
   - `text`
   - other supported fallback fields
3. JSON object recovery from extracted text:
   - direct JSON
   - fenced JSON
   - JSON surrounded by prose

This means the current contract is:

- provider remains `openrouter`
- `stepfun/step-3.5-flash:free` is the current active OpenRouter backend comparison candidate
- structured agent calls no longer hard depend on JSON mode
- it is not the formal main default
- it only becomes eligible for `selected_global_best_backend` if it passes Stage B artifact and feature separability gates on both datasets

## Current Limitation

What is still missing is not the parser fix, but real-network confirmation under
backend-comparison conditions. Local tests cover the text-mode parsing path,
but the tuple still needs Stage B end-to-end evidence before it can compete for
global backend selection.

## Conclusion

- Do not switch provider.
- Treat this model as a text-mode structured provider on OpenRouter.
- Do not freeze this model into formal main unless it wins Stage B backend comparison.
- Judge success by whether the full PHMGA chain works, artifacts are complete, and feature separability evidence passes, not by whether JSON mode is used.
