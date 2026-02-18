# BUG Report: plan_agent Module Review

**Agent:** Agent 1 (Bug Review Team)
**Date:** 2026-02-15
**Module:** plan_agent

## 1. Review Scope

### Files Reviewed
| File | Lines of Code | Status |
|------|---------------|--------|
| `src/agents/plan_agent.py` | 289 | Reviewed |
| `src/prompts/plan_prompt.py` | 259 | Reviewed |

**Total Lines Reviewed:** 548

## 2. Bugs Found by Severity

### High Severity Bugs

#### BUG-H-001: Unreachable Code After SystemExit
**File:** `src/agents/plan_agent.py:262-265`
**Severity:** High

**Code Location:**
```python
if __name__ == "__main__":
    raise SystemExit(
        "This module is not intended to be executed as a script. "
        "Use pytest (tests/test_plan_agent.py) or run the workflow via `python main.py case1 --config ...`."
    )

    dag = DAGState(...)  # Lines 267-289
    state = PHMState(...)
    # ... more code
```

**Description:**
After `raise SystemExit()`, there is unreachable code (lines 267-289) that can never be executed. This includes variable definitions (`dag`, `state`) and function calls. This represents dead code that wastes maintenance effort and indicates incomplete refactoring.

**Impact:**
- Code maintenance burden
- Potential confusion for developers
- Suggests incomplete cleanup

**Recommended Fix:**
Remove all unreachable code after the `SystemExit`:
```python
if __name__ == "__main__":
    raise SystemExit(
        "This module is not intended to be executed as a script. "
        "Use pytest (tests/test_plan_agent.py) or run the workflow via `python main.py case1 --config ...`."
    )
```

---

#### BUG-H-002: Exception Handling Silences Errors Without Clear Recovery Path
**File:** `src/agents/plan_agent.py:172-186`
**Severity:** High

**Code Location:**
```python
except Exception as e:
    # 捕获 LLM 调用、解析或验证中可能出现的错误
    detailed_plan = []
    error_logs = state.error_logs + [f"Planner error: {e}"]
    state.error_logs = error_logs
    log_event(
        logger,
        level="ERROR",
        event="plan.error",
        phase="builder",
        node="plan",
        message=f"Planner failed: {e}",
        payload={"error_logs_count": len(error_logs)},
    )
```

**Description:**
The broad `except Exception` clause catches all exceptions without differentiation. The function continues execution returning an empty `detailed_plan`, which may cause downstream issues. The error is logged but the calling code has no way to distinguish between:
1. LLM API failures
2. JSON parsing errors
3. Pydantic validation errors
4. Network issues
5. Malformed responses

**Impact:**
- Loss of error context for debugging
- Potential silent failures in the workflow
- Difficult to diagnose production issues

**Recommended Fix:**
```python
except json.JSONDecodeError as e:
    detailed_plan = []
    error_logs = state.error_logs + [f"Planner JSON parse error: {e}"]
    log_event(logger, level="ERROR", event="plan.json_error", ...)
except ValidationError as e:
    detailed_plan = []
    error_logs = state.error_logs + [f"Planner validation error: {e}"]
    log_event(logger, level="ERROR", event="plan.validation_error", ...)
except Exception as e:
    detailed_plan = []
    error_logs = state.error_logs + [f"Planner unexpected error: {type(e).__name__}: {e}"]
    log_event(logger, level="ERROR", event="plan.error", ...)
```

---

#### BUG-H-003: Unsafe JSON Parsing Without Schema Validation
**File:** `src/agents/plan_agent.py:134-141`
**Severity:** High

**Code Location:**
```python
json_str = resp.content
if "```json" in json_str:
    json_str = json_str.split("```json")[1].strip()
if "```" in json_str:
    json_str = json_str.split("```")[0].strip()

# 2. 使用 json.loads() 解析字符串
plan_dict = json.loads(json_str)
```

**Description:**
The JSON parsing logic assumes specific formatting from LLM responses. If `resp.content` is None, empty, or contains malformed JSON, `json.loads()` will raise a `json.JSONDecodeError`. There's no validation that:
- `json_str` is not empty after splitting
- The extracted content is actually valid JSON
- The JSON contains the expected "plan" key

**Impact:**
- Potential runtime crash if LLM returns unexpected format
- No graceful degradation for malformed responses
- Downstream code assumes `plan_dict` exists

**Recommended Fix:**
```python
json_str = resp.content
if not json_str or not isinstance(json_str, str):
    raise ValueError("LLM returned empty or invalid content")

if "```json" in json_str:
    parts = json_str.split("```json", 1)
    if len(parts) > 1:
        json_str = parts[1].strip()
        if "```" in json_str:
            json_str = json_str.split("```", 1)[0].strip()

if not json_str:
    raise ValueError("No JSON content found in LLM response")

plan_dict = json.loads(json_str)
if "plan" not in plan_dict:
    raise ValueError("LLM response missing 'plan' key")
```

---

### Medium Severity Bugs

#### BUG-M-001: Potential IndexError in String Splitting
**File:** `src/agents/plan_agent.py:135-138`
**Severity:** Medium

**Code Location:**
```python
if "```json" in json_str:
    json_str = json_str.split("```json")[1].strip()
if "```" in json_str:
    json_str = json_str.split("```")[0].strip()
```

**Description:**
Direct index access `[1]` and `[0]` without checking list length can cause `IndexError` if the string contains the delimiter but the split doesn't produce enough parts.

**Impact:**
- Runtime crash on edge cases
- Unhandled exception path

**Recommended Fix:**
```python
if "```json" in json_str:
    parts = json_str.split("```json", 1)
    if len(parts) > 1:
        json_str = parts[1].strip()
if "```" in json_str:
    parts = json_str.split("```", 1)
    if len(parts) > 0:
        json_str = parts[0].strip()
```

---

#### BUG-M-002: Missing Null Check for `reference_signal.meta`
**File:** `src/agents/plan_agent.py:159-161`
**Severity:** Medium

**Code Location:**
```python
fs = getattr(state, "fs", None)
if fs is None:
    fs = getattr(state.reference_signal, "meta", {}).get("fs")
```

**Description:**
If `state.reference_signal` is None, this will raise an `AttributeError`. The code assumes `reference_signal` always exists and has a `meta` attribute.

**Impact:**
- Potential crash if state is improperly initialized
- No defensive programming for edge cases

**Recommended Fix:**
```python
fs = getattr(state, "fs", None)
if fs is None:
    ref_signal = getattr(state, "reference_signal", None)
    if ref_signal is not None:
        meta = getattr(ref_signal, "meta", None) or {}
        fs = meta.get("fs")
```

---

#### BUG-M-003: Test Functions Have No Return Value Assertions
**File:** `src/agents/plan_agent.py:190-226, 228-258`
**Severity:** Medium

**Code Location:**
```python
def run_test_with_fake_llm(state: PHMState):
    """使用 FakeLLM 测试 plan_agent。"""
    # ... test code
    assert "detailed_plan" in result
    # No assertion that plan_agent actually returns the expected result to caller
```

**Description:**
The test functions `run_test_with_fake_llm()` and `run_test_with_real_llm()` don't return anything, making it difficult to use them programmatically or verify results in automated testing frameworks. They only print to stdout and use inline assertions.

**Impact:**
- Poor test integration
- Difficult to use in CI/CD pipelines
- No programmatic way to check test results

**Recommended Fix:**
```python
def run_test_with_fake_llm(state: PHMState) -> bool:
    """使用 FakeLLM 测试 plan_agent。返回测试是否通过。"""
    try:
        # ... existing test code
        print("Fake LLM Plan Agent test passed!")
        return True
    except AssertionError as e:
        print(f"Fake LLM Plan Agent test failed: {e}")
        return False
    except Exception as e:
        print(f"Fake LLM Plan Agent test error: {e}")
        return False
```

---

#### BUG-M-004: Missing Import Statement for FakeListChatModel
**File:** `src/agents/plan_agent.py:199`
**Severity:** Medium

**Code Location:**
```python
from langchain_community.chat_models import FakeListChatModel

model._FAKE_LLM = FakeListChatModel(
    responses=[...]
)
```

**Description:**
The import for `FakeListChatModel` is inside a function (`run_test_with_fake_llm`), which is unusual. This import should be at the module level for clarity and to avoid repeated imports if the function is called multiple times.

**Impact:**
- Code style inconsistency
- Potential performance impact if function called repeatedly
- Unclear dependencies

**Recommended Fix:**
Move import to top of file:
```python
from __future__ import annotations
# ... existing imports
from langchain_community.chat_models import FakeListChatModel  # Add at module level
```

---

#### BUG-M-005: Unsafe Direct State Mutation
**File:** `src/agents/plan_agent.py:175-176`
**Severity:** Medium

**Code Location:**
```python
error_logs = state.error_logs + [f"Planner error: {e}"]
state.error_logs = error_logs
```

**Description:**
Direct mutation of the input state parameter. While this may be intentional in LangGraph patterns, it's not clearly documented and could lead to unexpected behavior if the state is shared or if LangGraph's state management expectations aren't met.

**Impact:**
- Potential side effects
- Unclear ownership semantics
- Difficult to reason about state changes

**Recommended Fix:**
Consider returning the error logs in the function's return dict instead:
```python
return {
    "detailed_plan": detailed_plan,
    "error_logs": error_logs
}
```

---

### Low Severity Bugs

#### BUG-L-001: Inconsistent Logging of Full Prompt
**File:** `src/agents/plan_agent.py:115-120`
**Severity:** Low

**Code Location:**
```python
payload={
    "provider": os.getenv("LLM_PROVIDER"),
    "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
    "prompt": PLANNER_PROMPT,  # Full prompt logged
    "inputs": llm_input,
},
```

**Description:**
The full `PLANNER_PROMPT` (which is very large) is logged to the payload. While `logging_setup.py` has truncation logic for LLM prompts, this should be explicitly handled at the source to avoid unnecessary memory overhead.

**Impact:**
- Memory inefficiency
- Larger log files
- Potential performance impact

**Recommended Fix:**
```python
# Use a truncated or summarized version for logging
prompt_summary = f"PLANNER_PROMPT ({len(PLANNER_PROMPT)} chars)"
payload={
    "provider": os.getenv("LLM_PROVIDER"),
    "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
    "prompt": prompt_summary,
    "inputs": llm_input,
},
```

---

#### BUG-L-002: Unused Variable `node_id`
**File:** `src/agents/plan_agent.py:82`
**Severity:** Low

**Code Location:**
```python
dag_topology = {
    "nodes": [
        {
            "node_id": node.node_id,
            # ...
        }
        for node in state.dag_state.nodes.values()
    ],
}
```

**Description:**
The `node_id` is included in the topology but `dag_topology` is only used for JSON serialization to LLM. The field may not be used by the LLM for planning decisions, adding unnecessary payload size.

**Impact:**
- Larger LLM prompt size
- Increased token usage
- Minor performance impact

**Recommended Fix:**
Consider if `node_id` is actually needed for the LLM's planning decisions. If not, remove it from the topology representation.

---

#### BUG-L-003: Commented-Out Code
**File:** `src/agents/plan_agent.py:69, 91`
**Severity:** Low

**Code Location:**
```python
# f"  params:\n{params_str}" # TODO
# "leaves": state.dag_state.leaves, # Optional: include leaves if needed
```

**Description:**
Commented-out code and TODO comments should be resolved or removed. Technical debt that should be cleaned up.

**Impact:**
- Code clutter
- Unclear intentions
- Maintenance burden

**Recommended Fix:**
Either implement the TODO or remove the comment:
- If params are important for LLM, uncomment and fix the formatting
- If leaves are useful for planning, uncomment and use them
- Otherwise, remove these comments

---

#### BUG-L-004: Inconsistent Parameter Check
**File:** `src/agents/plan_agent.py:151`
**Severity:** Low

**Code Location:**
```python
if step_data.get("params") in ("", None):
    step_data["params"] = {}
```

**Description:**
The check `in ("", None)` is inconsistent - it's a tuple membership test. Should use `in (None, "")` or explicitly check for falsy values. Also, this won't catch other falsy values like `[]` or `0`.

**Impact:**
- Edge case bug
- Inconsistent empty value handling

**Recommended Fix:**
```python
params = step_data.get("params")
if params is None or params == "":
    step_data["params"] = {}
```

---

#### BUG-L-005: Missing Type Hints for Test Functions
**File:** `src/agents/plan_agent.py:190, 228`
**Severity:** Low

**Code Location:**
```python
def run_test_with_fake_llm(state: PHMState):  # No return type
def run_test_with_real_llm(state: PHMState):  # No return type
```

**Description:**
Missing return type hints for test functions. Since these are test functions that don't return values, the return hint should be `-> None`.

**Impact:**
- Reduced type safety
- Poor IDE support
- Code style inconsistency

**Recommended Fix:**
```python
def run_test_with_fake_llm(state: PHMState) -> None:
def run_test_with_real_llm(state: PHMState) -> None:
```

---

#### BUG-L-006: Prompt JSON Example Has Syntax Error
**File:** `src/prompts/plan_prompt.py:81-106`
**Severity:** Low

**Code Location:**
```json
{{
  "plan": [
    {{
      "parent": "patch_01_ch1",
      "op_name": "mean",
      "params": {{}}
    }},
    ...
    }}  // Missing comma between entries
    {{
      "parent": "ch2",
      "op_name": "kurtosis",
      "params": {{}}
    }}
  ]
}}
```

**Description:**
The example JSON in `PLANNER_PROMPT` has missing commas between entries (lines 80-81, 94-95, 99-100). This is invalid JSON syntax and could confuse the LLM.

**Impact:**
- LLM may generate invalid JSON
- Poor example for prompt engineering

**Recommended Fix:**
Add missing commas between JSON objects in the example.

---

## 3. Security Concerns

### SECURITY-001: Sensitive Data in Logs
**File:** `src/agents/plan_agent.py:115-120`

**Description:**
The LLM inputs including `user_instruction` are logged. If the user instruction contains sensitive data (e.g., proprietary information), it will be written to log files.

**Mitigation:**
The `logging_setup.py` module has sanitization functions (`_sanitize`, `_mask_string`) that handle common sensitive patterns. However, user instructions could still contain proprietary information that isn't caught by pattern matching.

**Recommendation:**
Document that log files may contain sensitive user data and should be handled accordingly. Consider adding a configuration option to exclude user input from logs.

---

## 4. Statistics Summary

| Category | Count |
|----------|-------|
| High Severity | 3 |
| Medium Severity | 5 |
| Low Severity | 6 |
| Security Concerns | 1 |
| **TOTAL** | **15** |

### Breakdown by Dimension

| Dimension | Count |
|-----------|-------|
| Error Handling | 3 |
| Input Validation | 4 |
| Null/None Handling | 2 |
| Code Quality | 4 |
| Type Safety | 1 |
| Security | 1 |

---

## 5. Recommendations

1. **Immediate Actions (High Priority):**
   - Fix BUG-H-001: Remove unreachable code
   - Fix BUG-H-002: Implement specific exception handling
   - Fix BUG-H-003: Add proper JSON validation

2. **Short-term (Medium Priority):**
   - Add defensive null checks for state attributes
   - Improve test functions with return values
   - Move imports to module level

3. **Long-term (Low Priority):**
   - Clean up commented code and TODOs
   - Add comprehensive type hints
   - Fix prompt template examples
   - Review logging payload size

4. **Testing Recommendations:**
   - Add unit tests for JSON parsing edge cases
   - Add tests for exception handling paths
   - Add tests with malformed LLM responses
   - Add integration tests with various state configurations

---

**Review Completed By:** Agent 1
**Report Generated:** 2026-02-15
