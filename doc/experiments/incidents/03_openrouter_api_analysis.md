# OpenRouter Candidate Note

**Date**: 2026-03-18
**Provider**: `openrouter`
**Historical Active Model**: `z-ai/glm-4.5-air:free`

## Status: StepFun Historical Failure Preserved, GLM Historical Round Retained

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
- `z-ai/glm-4.5-air:free` remains a retained historical OpenRouter comparison row (`v2`)
- `stepfun/step-3.5-flash:free` remains a historical comparison failure and registry candidate
- `nvidia/nemotron-3-super-120b-a12b:free` is the current active OpenRouter backend comparison candidate
- structured agent calls no longer hard depend on JSON mode
- it is not the formal main default
- an OpenRouter tuple only becomes eligible for `selected_global_best_backend` if it passes Stage B artifact and feature separability gates on both datasets

## Current Limitation

What is still missing is a clean formal Stage B comparison round for the current active Nemotron tuple.
The StepFun parser path and the GLM `v2` comparison row remain valuable as historical incident records,
but neither is part of the current selection round.

## Conclusion

- Do not switch provider family to chase the incident.
- Keep StepFun as a historical comparison failure and registry candidate.
- Keep `z-ai/glm-4.5-air:free` as retained historical comparison evidence.
- Treat `nvidia/nemotron-3-super-120b-a12b:free` as the current active OpenRouter comparison candidate.
- Do not freeze any OpenRouter tuple into formal main unless it wins Stage B backend comparison.
- Judge success by whether the full PHMGA chain works, artifacts are complete, and feature separability evidence passes, not by whether JSON mode is used.
