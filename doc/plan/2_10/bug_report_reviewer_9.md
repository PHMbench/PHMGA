# Test Coverage Analysis Report
**Reviewer:** reviewer-9
**Date:** 2025-02-15
**Project:** PHMGA (Prognostics and Health Management Graph Agent)

---

## Executive Summary

This analysis identifies significant test coverage gaps in the PHMGA project. The codebase has a solid foundation with 28 test files covering basic functionality, but **critical gaps remain in error handling, edge cases, and integration scenarios**.

**Key Findings:**
- **12 high-risk untested code paths** in critical agents
- **8 medium-risk missing test cases** for important utility functions
- **Numerous edge cases** without coverage (empty inputs, malformed data, boundary conditions)
- **Missing tests** for 15+ modules including research agents, deep model training, and data loading
- **No property-based testing** for data validation schemas
- **No chaos engineering tests** for LLM failure scenarios

---

## High-Risk Untested Code (Critical)

### 1. Missing Tests for Research Agents
**Files:** `src/agents/deep_research_agents.py`, `src/agents/prompts_research.py`, `src/agents/prompt_research.py`
**Risk Level:** HIGH
**Description:** These research-related agent modules have **no dedicated test files**.
**Impact:** Unknown behavior for research workflows; potential for unhandled exceptions in research-oriented use cases.

**Suggested Tests:**
- `tests/test_deep_research_agents.py` - Test research agent initialization, execution, and error handling
- `tests/test_prompt_research.py` - Test prompt generation, templating, and validation
- Test research agent state management
- Test research agent timeout and cancellation scenarios

---

### 2. Deep Model Training Agent - Incomplete Coverage
**File:** `src/agents/deep_model_train_agent.py`
**Test File:** Partially covered (no dedicated test file)
**Risk Level:** HIGH

**Untested Functions/Lines:**
- `_train_with_vibench_factory` (lines 261-594) - Complex 300+ line function with no dedicated tests
- `_resolve_tspn_config` (lines 101-152) - Partially tested in `test_preflight_and_config_resolve.py` but missing:
  - Line 123-127: Dimension mismatch error path
  - Line 134-138: num_classes mismatch error path
- `_infer_channels_and_length` (lines 155-170) - Error paths not tested
- `_build_fused_view` (lines 173-215) - Edge cases not tested:
  - Empty intersections between channels and labels
  - Shape mismatches
  - Missing results dictionaries

**Missing Edge Cases:**
- Empty training data (lines 196, 688-689)
- Shape validation failures (lines 167, 204-207)
- Device-specific behaviors (CUDA vs CPU)
- Checkpoint save/load failures

**Suggested Tests:**
```python
# tests/test_deep_model_train_agent_edge_cases.py
def test_train_with_vibench_factory_empty_data():
    # Test handling of empty datasets

def test_train_with_vibench_factory_shape_mismatch():
    # Test handling of incorrect array shapes

def test_resolve_tspn_config_dimension_mismatch():
    # Test ValueError when dimensions don't match

def test_resolve_tspn_config_num_classes_mismatch():
    # Test ValueError when num_classes doesn't match

def test_checkpoint_save_failure():
    # Test handling of filesystem errors during checkpoint save

def test_model_init_weights_from_metadata_failure():
    # Test handling of metadata initialization failures
```

---

### 3. DAG Init Agent - No Tests
**File:** `src/agents/dag_init_agent.py`
**Risk Level:** HIGH
**Description:** This agent initializes minimal processed DAGs using LLM but has **no dedicated test file**.

**Untested Code Paths:**
- `_infer_L_and_fs` (lines 11-32) - All error paths
- `_coerce_chain` (lines 35-40) - Edge cases with None/empty inputs
- Full `dag_init_agent` function (lines 43-144)
- LLM failure fallback behavior
- Invalid method name handling

**Suggested Tests:**
```python
# tests/test_dag_init_agent.py
def test_dag_init_agent_creates_processed_nodes():
    # Test basic node creation

def test_dag_init_agent_with_existing_processed_nodes():
    # Test early return when DAG already has ProcessedData

def test_dag_init_agent_llm_failure_fallback():
    # Test fallback to FFT-only initialization on LLM failure

def test_dag_init_agent_invalid_method_names():
    # Test handling of method names not in allowed_ops

def test_infer_L_and_fs_with_missing_data():
    # Test error handling when results['ref'] is missing

def test_infer_L_and_fs_with_invalid_shapes():
    # Test error handling for non-3D arrays
```

---

### 4. TSPN Bootstrap Agent - Limited Coverage
**File:** `src/agents/tspn_bootstrap_agent.py`
**Test File:** `tests/test_tspn_bootstrap_agent.py` (1 test only)
**Risk Level:** MEDIUM-HIGH

**Untested Code Paths:**
- `_infer_channels_and_length` (lines 20-36) - All error paths
- `_infer_num_classes` (lines 39-49) - Edge cases with single class
- `_build_depth_index` (lines 52-67) - Cycle detection (line 60-61)
- `_map_method_to_token` (lines 70-74) - Unsupported operator handling
- WaveFilter injection logic (lines 154-164)

**Suggested Tests:**
```python
def test_build_depth_index_with_cycle():
    # Test ValueError when DAG contains a cycle

def test_infer_num_classes_with_single_class():
    # Test handling of single-class datasets (should return 2)

def test_infer_num_classes_with_no_labels():
    # Test fallback to root meta when labels_ref is empty

def test_map_method_to_token_unsupported():
    # Test handling of unsupported methods
```

---

### 5. Execute Agent - Missing Error Path Tests
**File:** `src/agents/execute_agent.py`
**Test File:** `tests/test_execute_agent.py` (1 basic test)
**Risk Level:** MEDIUM-HIGH

**Untested Code Paths:**
- `_resolve_params` (lines 20-93) - LLM parameter generation failure path (lines 87-91)
- `_execute_multi_variable_op` (lines 107-159) - Edge cases:
  - Line 117-118: Filter invalid parent results
  - Empty signal_keys
- `_execute_single_variable_op` (lines 162-182) - Similar edge cases
- Execute agent neuro-symbolic mode (lines 189-223)
- Parent validation failure (lines 269-272)
- MAX_STEPS limit handling (line 252)

**Suggested Tests:**
```python
def test_resolve_params_llm_failure():
    # Test ValueError when LLM fails to generate required params

def test_execute_multi_variable_op_missing_results():
    # Test handling of missing result dictionaries

def test_execute_multi_variable_op_shape_mismatch():
    # Test handling of shape mismatches between signals

def test_execute_agent_neuro_symbolic_mode():
    # Test the neuro_symbolic_train execution path

def test_execute_agent_max_steps_limit():
    # Test that execution stops after MAX_STEPS

def test_execute_agent_parent_not_found():
    # Test error handling when parent node doesn't exist
```

---

### 6. Inquirer Agent - Missing Path B Tests
**File:** `src/agents/inquirer_agent.py`
**Test File:** `tests/test_inquirer_agent.py` (only tests Path A)
**Risk Level:** MEDIUM

**Untested Code Paths:**
- **Path B: Paired ref/tst leaves** (lines 77-118) - This entire code path is not tested:
  - Grouping by (channel, method)
  - Creating similarity nodes
  - Shape mismatch handling
- Metric calculation edge cases for cosine (zero division), pearson (NaN values)

**Suggested Tests:**
```python
def test_inquirer_agent_paired_leaves_creates_similarity_nodes():
    # Test Path B with paired ref/tst nodes

def test_inquirer_agent_paired_leaves_shape_mismatch():
    # Test shape mismatch handling in Path B

def test_calc_metric_cosine_zero_division():
    # Test cosine similarity with zero-norm vectors

def test_calc_metric_pearson_nan():
    # Test pearson correlation with NaN inputs
```

---

### 7. Shallow ML Agent - Missing Edge Cases
**File:** `src/agents/shallow_ml_agent.py`
**Test File:** `tests/test_shallow_ml_agent.py` (1 basic test)
**Risk Level:** MEDIUM

**Untested Code Paths:**
- pandas unavailable path (lines 37-42)
- cv_folds validation (lines 63-74) - Edge cases:
  - Single unique class in training data
  - cv_folds > number of samples
- Empty predictions dictionary (line 108)
- Ensemble voting logic with invalid probabilities
- SVM algorithm variant (line 22-23)

**Suggested Tests:**
```python
def test_shallow_ml_agent_without_pandas():
    # Test graceful degradation when pandas is unavailable

def test_shallow_ml_agent_single_class_training_data():
    # Test handling of single-class datasets

def test_shallow_ml_agent_cv_folds_exceeds_samples():
    # Test CV fold clamping

def test_shallow_ml_agent_svm_algorithm():
    # Test SVM-based training

def test_shallow_ml_agent_soft_voting_ensemble():
    # Test soft voting ensemble method
```

---

### 8. Reflect Agent - Missing Validation Tests
**File:** `src/agents/reflect_agent.py`
**Test File:** `tests/test_reflect_agent.py` (1 basic test for "finish" decision)
**Risk Level:** MEDIUM

**Untested Code Paths:**
- Invalid input handling (lines 39-40) - "halt" decision
- Invalid JSON parsing (lines 128-130)
- Invalid decision handling (lines 125-127)
- POST_PLAN stage (only POST_EXECUTE tested)
- Debug mode functionality (lines 19-20, 34-111)

**Suggested Tests:**
```python
def test_reflect_agent_invalid_input():
    # Test "halt" decision for missing inputs

def test_reflect_agent_invalid_json_response():
    # Test "halt" decision for unparseable JSON

def test_reflect_agent_invalid_decision_value():
    # Test "halt" decision for unknown decision types

def test_reflect_agent_post_plan_stage():
    # Test reflection at POST_PLAN stage

def test_reflect_agent_with_debug_enabled():
    # Test debug output with PHM_DEBUG_REFLECT=1
```

---

### 9. Report Agent - Missing Fallback Tests
**File:** `src/agents/report_agent.py`
**Test File:** `tests/test_report_agent.py` (1 basic test)
**Risk Level:** MEDIUM

**Untested Code Paths:**
- `_template_report` function (lines 69-120) - Not directly tested
- Template fallback path (lines 179-188)
- LLM exception fallback (lines 199-209)
- Graph export failure handling (lines 149-158)
- Similarity stats extraction from leaf nodes (lines 162-166)

**Suggested Tests:**
```python
def test_report_agent_template_mode():
    # Test PHM_REPORT_MODE=template uses template directly

def test_report_agent_llm_exception_fallback():
    # Test fallback to template when LLM fails

def test_report_agent_graph_export_failure():
    # Test handling of graph export failure

def test_template_report_with_no_test_metrics():
    # Test report formatting when n_test=0
```

---

### 10. Plan Agent - Missing Error Handling
**File:** `src/agents/plan_agent.py`
**Test File:** `tests/test_plan_agent.py` (1 basic test)
**Risk Level:** MEDIUM

**Untested Code Paths:**
- Exception handling (lines 172-186) - Not tested
- Empty params handling (lines 151-152)
- Backward-compat parent in params (lines 146-149)
- fs injection from reference_signal.meta (lines 159-161)
- JSON parsing with malformed responses

**Suggested Tests:**
```python
def test_plan_agent_llm_exception():
    # Test handling of LLM invocation errors

def test_plan_agent_malformed_json():
    # Test handling of unparseable LLM responses

def test_plan_agent_empty_params():
    # Test handling of empty params in step data

def test_plan_agent_parent_in_params_backward_compat():
    # Test backward compatibility for parent inside params
```

---

## Medium-Risk Missing Tests

### 11. PHM Outer Graph - No Tests
**File:** `src/phm_outer_graph.py`
**Risk Level:** MEDIUM
**Description:** Core orchestration module with **no dedicated tests**.

**Untested Functions:**
- `build_builder_graph` (lines 89-128)
- `build_executor_graph` (lines 131-262)
- `_FallbackGraph` class (lines 28-42)
- `_run_node` with exceptions (lines 76-86)
- `_executor_path` routing logic (lines 186-199)

**Suggested Tests:**
```python
# tests/test_phm_outer_graph.py
def test_build_builder_graph_basic():
    # Test graph construction

def test_build_executor_graph_routing():
    # Test tspn_fast_path vs full_path routing

def test_fallback_graph_stream():
    # Test _FallbackGraph when langgraph unavailable

def test_run_node_exception_handling():
    # Test exception logging in _run_node
```

---

### 12. State Classes - Incomplete Coverage
**File:** `src/states/phm_states.py`
**Risk Level:** MEDIUM
**Description:** Core state classes have minimal direct testing.

**Untested Code Paths:**
- `DAGState.__init__` defaults (lines 127-136)
- `DAGTracker.add_node` with duplicate nodes (lines 163-165)
- `DAGTracker.update` (lines 144-155)
- `DAGTracker.export_json` with max_nodes trimming (lines 189-213)
- `DAGTracker.write_png` failure paths (lines 260-290)
- `DAGTracker._build_dot_source` (lines 236-258)

**Suggested Tests:**
```python
# tests/test_phm_states.py
def test_dag_state_init_creates_root_nodes():
    # Test default initialization behavior

def test_dag_tracker_add_duplicate_node():
    # Test handling of duplicate node additions

def test_dag_tracker_export_json_trimming():
    # Test max_nodes parameter in export_json

def test_dag_tracker_write_png_fallback():
    # Test DOT fallback when PNG export fails
```

---

### 13. Utility Modules - Missing Tests
**Files:**
- `src/utils/preflight.py` (partially tested in `test_preflight_and_config_resolve.py`)
- `src/utils/logging_setup.py` (partially tested)
- `src/utils/visualization.py` - **no tests**
- `src/utils/data_factory_wrapper.py` - **no tests**

**Risk Level:** MEDIUM

**Untested Code:**
- `run_preflight_from_config_path` (preflight.py:173-176)
- `write_preflight_report` (preflight.py:179-182)
- `PHMVibenchDataFactory` - entire class untested
- All visualization functions

**Suggested Tests:**
```python
# tests/test_data_factory_wrapper.py
def test_vibench_data_factory_build():
    # Test Vibench data loading

def test_vibench_data_factory_missing_metadata():
    # Test error handling for missing metadata files
```

---

### 14. Configuration Module - No Tests
**File:** `src/configuration.py`
**Risk Level:** MEDIUM
**Description:** Configuration management has **no dedicated tests**.

**Untested Code Paths:**
- All configuration resolution logic
- Environment variable handling
- Configuration validation

---

## Low-Risk but Important Gaps

### 15. Prompt Templates - No Validation Tests
**Files:** `src/prompts/*.py`
**Risk Level:** LOW-MEDIUM
**Description:** Prompt templates are not validated for:
- Required placeholder presence
- Template syntax correctness
- Overflows with long inputs

### 16. Signal Processing Operators - No Comprehensive Tests
**File:** `src/tools/signal_processing_schemas.py` + operator files
**Risk Level:** LOW-MEDIUM
**Description:** Operator registry has minimal testing. No tests for:
- Registration conflicts
- get_operator with unknown op_name
- Operator execute() with edge case inputs

**Suggested Tests:**
```python
# tests/test_operator_registry.py
def test_register_op_duplicate():
    # Test duplicate registration handling

def test_get_operator_unknown():
    # Test KeyError for unknown operators

def test_operator_execute_edge_cases():
    # Test operators with empty, NaN, and inf inputs
```

---

## Missing Integration Tests

### 17. End-to-End Workflow Tests
**Files:** Multiple
**Risk Level:** MEDIUM
**Current State:** `test_end2end.py` is **skipped** in CI

**Missing Scenarios:**
- Full builder graph execution with errors
- Executor graph with both shallow and TSPN backends
- Multi-iteration plan-execute-reflect loops
- State persistence and restoration

### 18. Graph Routing Tests
**File:** `test_executor_tspn_fast_path.py` exists but only tests routing decision
**Missing:**
- Full path execution (inquire -> prepare -> init_dag -> bootstrap -> train -> report)
- tspn_fast_path execution (init_dag -> bootstrap -> train -> report)

---

## Testing Infrastructure Gaps

### 19. No Property-Based Testing
**Risk Level:** MEDIUM
**Description:** No use of hypothesis or similar for:
- Pydantic schema validation
- Shape transformations
- Numeric stability properties

### 20. No Chaos Engineering
**Risk Level:** MEDIUM
**Description:** No tests for:
- LLM timeout scenarios
- Partial LLM responses
- Malformed LLM JSON
- Network failures

### 21. No Performance/Load Tests
**Risk Level:** LOW
**Description:** No tests for:
- Large DAG handling
- Memory usage with deep graphs
- Timeout enforcement

---

## Summary by Module

| Module | Test Coverage | Risk Level | Priority |
|--------|---------------|------------|----------|
| deep_research_agents.py | 0% | HIGH | 1 |
| deep_model_train_agent.py | ~10% | HIGH | 2 |
| dag_init_agent.py | 0% | HIGH | 3 |
| tspn_bootstrap_agent.py | ~20% | MEDIUM-HIGH | 4 |
| execute_agent.py | ~15% | MEDIUM-HIGH | 5 |
| phm_outer_graph.py | 0% | MEDIUM | 6 |
| inquirer_agent.py | ~50% (Path A only) | MEDIUM | 7 |
| shallow_ml_agent.py | ~20% | MEDIUM | 8 |
| data_factory_wrapper.py | 0% | MEDIUM | 9 |
| configuration.py | 0% | MEDIUM | 10 |
| preflight.py | ~40% | LOW-MEDIUM | 11 |
| states/phm_states.py | ~10% | LOW-MEDIUM | 12 |
| visualization.py | 0% | LOW | 13 |
| prompts/*.py | 0% | LOW-MEDIUM | 14 |

---

## Recommended Action Plan

### Phase 1: Critical Gaps (Week 1-2)
1. Add tests for `dag_init_agent.py` - high impact, low complexity
2. Add error path tests for `deep_model_train_agent.py`
3. Add Path B tests for `inquirer_agent.py`
4. Add neuro-symbolic mode tests for `execute_agent.py`

### Phase 2: Core Infrastructure (Week 3-4)
1. Add tests for `phm_outer_graph.py` routing logic
2. Add state management tests for `phm_states.py`
3. Add configuration resolution tests
4. Enable and fix `test_end2end.py`

### Phase 3: Edge Cases & Robustness (Week 5-6)
1. Add property-based tests for schemas
2. Add chaos engineering tests for LLM failures
3. Add empty/invalid input tests for all agents
4. Add timeout and cancellation tests

### Phase 4: Integration & Performance (Week 7-8)
1. Full workflow integration tests
2. Large DAG performance tests
3. Memory usage tests
4. Concurrent execution tests

---

## Test Metrics

Current test statistics:
- **Total test files:** 28
- **Estimated test functions:** ~60
- **Code coverage estimate:** ~35-40%
- **Critical path coverage:** ~50%

Target metrics:
- **Total test files:** 45+
- **Estimated test functions:** 150+
- **Code coverage target:** 80%+
- **Critical path coverage:** 95%+

---

## Conclusion

The PHMGA project has a solid testing foundation but significant gaps remain. The most critical gaps are in error handling paths, edge cases, and integration scenarios. Prioritizing the high-risk items identified in this report will significantly improve the reliability and maintainability of the codebase.

**Immediate priorities:**
1. Test the `dag_init_agent.py` - creates processed nodes, no tests exist
2. Add error path tests for `deep_model_train_agent.py` - complex, untested error handling
3. Add tests for Path B in `inquirer_agent.py` - entire alternate code path untested
4. Enable and fix `test_end2end.py` - currently skipped, critical for validation

**Note:** Many tests are marked as skipped or require special environment variables (`PHM_ENABLE_TORCH_TESTS`, `PHM_ENABLE_GEMINI_TESTS`, etc.). Consider enabling these by default in CI or providing mock implementations.
