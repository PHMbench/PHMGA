# Bug Mitigation Plan
## Prioritized Fix List with Dependencies

**Date:** 2025-02-15
**Based On:** Cross-Module Bug Report
**Affected Modules:** State Management, Core Orchestration, Graph Implementations, Configuration, Integration, TSPN Model, Signal Processing

---

## Fix Priority Matrix

| Priority | Count | Estimated Complexity | Risk if Unfixed |
|----------|-------|---------------------|-----------------|
| **Critical** | 8 | High | System failure/infinite loops |
| **High** | 12 | Medium-High | Data loss/crashes |
| **Medium** | 10 | Low-Medium | Degraded performance |
| **Low** | 15+ | Low | Code quality/maintainability |

---

## CRITICAL Priority Fixes (Week 1)

### C-1: Add Missing `get_llm` Function to src/model.py
**Cross-Module Reference:** CM-3
**Complexity:** Low (1-2 hours)
**Dependencies:** None (blocking all agents)
**Files:** `src/model.py`

**Action:**
```python
# Add to src/model.py after get_default_llm()
def get_llm(config: Optional[Configuration] = None, **kwargs) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model for agent use."""
    conf = config or Configuration.from_runnable_config(None)
    return ChatGoogleGenerativeAI(
        model=conf.phm_model,
        temperature=0.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
        **kwargs,
    )
```

**Testing:**
- Unit test: Verify function returns correct ChatGoogleGenerativeAI instance
- Integration test: Verify plan_agent, reflect_agent, report_agent can import and use

---

### C-2: Fix State Mutation Inconsistency
**Cross-Module Reference:** CM-1
**Complexity:** High (8-16 hours)
**Dependencies:** None
**Files:**
- `src/states/phm_states.py:411-416`
- `src/agents/execute_agent.py:404-445`
- `src/phm_outer_graph.py:28-42`
- `src/cases/case1.py:237-261`

**Actions:**
1. In `phm_states.py`: Remove caching or invalidate properly:
```python
def tracker(self) -> "DAGTracker":
    # Always create a fresh tracker to ensure consistency
    return DAGTracker(self.dag_state)
```

2. In `execute_agent.py`: Remove in-place mutation at line 445
3. In `phm_outer_graph.py`: Use return-value pattern instead of setattr
4. In `case1.py`: Use same state for streaming and updates

**Testing:**
- Unit test: Verify tracker returns fresh instance
- Integration test: State consistency across graph execution
- Comparison test: LangGraph vs fallback produce same results

---

### C-3: Add Maximum Iteration Limit to Builder Graph
**Cross-Module Reference:** CM-2
**Complexity:** Medium (4-6 hours)
**Dependencies:** None
**Files:**
- `src/phm_outer_graph.py:118-126`
- `src/cases/case1.py:237-303`

**Actions:**
1. Add iteration_count field to PHMState
2. Modify builder conditional edge:
```python
builder.add_conditional_edges(
    "reflect",
    lambda state: END if (
        not state.needs_revision or
        state.iteration_count >= 50  # Safety limit
    ) else "plan",
    {"plan": "plan", END: END},
)
```

3. Add guard in case runner:
```python
MAX_ITERATIONS = 50
while iteration <= MAX_ITERATIONS:
    # ... existing code ...
```

**Testing:**
- Unit test: Verify loop terminates after max iterations
- Integration test: Force needs_revision=True and verify termination

---

### C-4: Fix Attribute Name Inconsistency (processed_data vs results)
**Cross-Module Reference:** CM-11
**Complexity:** Low (1-2 hours)
**Dependencies:** None
**Files:**
- `src/states/phm_states.py:306`
- `src/tools/comparator_tool.py:34, 41`

**Action:**
```python
def get_node_data(state: "PHMState", node_id: str):
    """Utility to fetch raw array data from a node."""
    node = state.dag_state.nodes.get(node_id)
    if isinstance(node, InputData):
        return np.asarray(node.data.get("signal", []))
    if isinstance(node, ProcessedData):
        if isinstance(node.results, dict):
            return node.results
        return np.asarray(node.results) if node.results is not None else None
    return None
```

**Testing:**
- Unit test: Test with ProcessedData nodes

---

### C-5: Fix Orphaned Methods in phm_states.py
**Cross-Module Reference:** CM-12
**Complexity:** Medium (2-4 hours)
**Dependencies:** None
**Files:** `src/states/phm_states.py:309-327`

**Action:**
Move methods inside DAGTracker class and update serialization:
```python
class DAGTracker:
    # ... existing methods ...

    def transfer_to_langgraph(self) -> nx.DiGraph:
        """Transfer DAGState to LangGraph-usable networkx graph."""
        return self.g

    def save(self, path: str) -> None:
        """Save DAG state to specified path."""
        import json
        with open(path, 'w') as f:
            json.dump(self.state.model_dump(), f, indent=4)

    def load(self, path: str) -> None:
        """Load DAG state from specified path."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.state = DAGState(**data)
            self.g = nx.DiGraph()
            for n in self.state.nodes.values():
                self._add_node(n)
            self.state.leaves = list(self.state.channels)
```

**Testing:**
- Unit test: Verify save/load round-trip

---

### C-6: Fix Parent Type Inconsistency
**Cross-Module Reference:** CM-4
**Complexity:** Medium (4-6 hours)
**Dependencies:** None
**Files:**
- `src/states/phm_states.py:49`
- Multiple locations where parents is accessed

**Action:**
Add validator to normalize parents to list:
```python
from pydantic import field_validator

class _NodeBase(BaseModel):
    node_id: str = Field(default_factory=lambda: f"n_{uuid.uuid4().hex[:8]}")
    parents: List[str] = Field(default_factory=list)
    stage: Literal["input", "processed", "similarity", "dataset", "output"] = "input"

    @field_validator('parents', mode='before')
    @classmethod
    def normalize_parents(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [v]
        if v is None:
            return []
        return v
```

**Testing:**
- Unit test: Verify string parent converted to list
- Integration test: Verify graph operations with normalized parents

---

### C-7: Add Cycle Detection to DAG Operations
**Cross-Module Reference:** CM-6
**Complexity:** Medium (4-6 hours)
**Dependencies:** C-6 (parent normalization)
**Files:** `src/states/phm_states.py:169-174`

**Action:**
```python
for p in parents:
    if p and p in self.g:
        self.g.add_edge(p, node.node_id)
        if not nx.is_directed_acyclic_graph(self.g):
            self.g.remove_edge(p, node.node_id)
            raise ValueError(f"Adding edge {p} -> {node.node_id} would create a cycle")
```

**Testing:**
- Unit test: Attempt to add cycle, verify error raised

---

### C-8: Fix Hardcoded User Paths in Configs
**Cross-Module Reference:** CM-8
**Complexity:** Low (2-3 hours)
**Dependencies:** None
**Files:** All config YAML files

**Action:**
Replace hardcoded paths with relative paths or environment variables:
```yaml
# Before
save_dir: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save"

# After
save_dir: "./save/case1"
```

**Testing:**
- Integration test: Run case from different working directory

---

## HIGH Priority Fixes (Week 2-3)

### H-1: Fix Leaves Update Logic
**Cross-Module Reference:** CM-7
**Complexity:** Medium (4-6 hours)
**Dependencies:** None
**Files:** `src/states/phm_states.py:181-184`

**Action:**
Recompute leaves from graph topology:
```python
def add_node(self, node: _NodeBase) -> str:
    # ... existing validation ...
    self.state.leaves = [
        nid for nid in self.g.nodes()
        if self.g.out_degree(nid) == 0
    ]
    return node.node_id
```

**Testing:**
- Unit test: Test with diamond graph pattern

---

### H-2: Add Missing run_executor Flag to Legacy Configs
**Cross-Module Reference:** CM-9
**Complexity:** Low (1 hour)
**Dependencies:** None
**Files:** All legacy config YAMLs

**Action:**
Add `run_executor: true` to all legacy configs

---

### H-3: Add API Key Validation
**Cross-Module Reference:** CM-10
**Complexity:** Medium (4-6 hours)
**Dependencies:** None
**Files:** `src/configuration.py`

**Action:**
```python
def validate_api_keys(self) -> list[str]:
    """Return list of missing API keys."""
    provider = self.llm_provider.lower()
    missing = []
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    # ... other providers
    return missing
```

---

### H-4: Fix Export JSON Field Access
**Cross-Module Reference:** CM-13
**Complexity:** Low (2-3 hours)
**Dependencies:** None
**Files:** `src/states/phm_states.py:198-210`

**Action:**
Use conditional field inclusion based on node type

---

### H-5: Update Pydantic v1 to v2 Syntax
**Cross-Module Reference:** CM-5
**Complexity:** Medium (4-6 hours)
**Dependencies:** None
**Files:** Multiple (grep for `.dict()`)

**Action:**
Replace all `.dict()` with `.model_dump()`
Replace `deep=True` copies with `deep=False` where appropriate

---

### H-6: Fix Duplicate .env Loading
**Cross-Module Reference:** CM-14
**Complexity:** Medium (4-6 hours)
**Dependencies:** None
**Files:** Multiple files loading .env

**Action:**
Create single env loading module, load once at entry point

---

### H-7: Signal Processing Critical Numerical Issues
**From:** Reviewer-5
**Complexity:** High (16-24 hours)
**Dependencies:** None
**Files:** `src/tools/*.py`

**Actions:**
1. Fix SavitzkyGolayFilterOp parameter validation
2. Fix FilterOp cutoff frequency validation
3. Fix DenoiseWaveletOp array length handling
4. Fix CepstrumOp log of zero values
5. Fix ResampleOp target length validation
6. Fix HjorthParametersOp division by zero
7. Fix BandPowerOp frequency band validation
8. Fix PatchOp parameter validation

---

### H-8: TSPN Model Numerical Stability
**From:** Reviewer-6
**Complexity:** Medium (8-12 hours)
**Dependencies:** None
**Files:** `src/model/explainable/tspn.py`, `src/agents/deep_model_train_agent.py`

**Actions:**
1. Fix _softplus_inv numerical precision
2. Add gradient clipping to training loops
3. Fix seed setting order
4. Add input validation to forward pass

---

### H-9: Core Orchestration State Issues
**From:** Reviewer-3
**Complexity:** Medium (8-12 hours)
**Dependencies:** C-2 (state mutation)
**Files:** `src/phm_outer_graph.py`, `src/cases/case1.py`

**Actions:**
1. Add iteration limit enforcement
2. Clarify/remove route node
3. Fix deep copy usage

---

### H-10: Graph Implementation Issues
**From:** Reviewer-7
**Complexity:** Medium (8-12 hours)
**Dependencies:** C-6, C-7
**Files:** `src/states/phm_states.py`, `src/agents/execute_agent.py`, `src/utils/__init__.py`

**Actions:**
1. Fix topological sort on empty graphs
2. Fix parent access in multi-variable ops
3. Fix graph export special character handling
4. Add validation to dag_init_agent

---

### H-11: Configuration Path Resolution
**From:** Reviewer-8, Reviewer-10
**Complexity:** Medium (6-8 hours)
**Dependencies:** None
**Files:** `src/configuration.py`, `src/agents/deep_model_train_agent.py`

**Actions:**
1. Fix environment variable naming inconsistency
2. Add config field validation
3. Fix model config path resolution with absolute paths

---

### H-12: Integration State Update Issues
**From:** Reviewer-10
**Complexity:** Medium (6-8 hours)
**Dependencies:** C-2
**Files:** `src/agents/execute_agent.py`, `src/cases/case1.py`

**Actions:**
1. Fix state update application in case1.py
2. Implement conditional edge handling in fallback graph

---

## MEDIUM Priority Fixes (Week 4-5)

### M-1: Weak Type Annotation for nodes Dictionary
**From:** Reviewer-1, BUG-4
**Complexity:** Low (2-3 hours)
**Files:** `src/states/phm_states.py:121`

### M-2: Unsafe State Initialization
**From:** Reviewer-1, BUG-5
**Complexity:** Low (2-3 hours)
**Files:** `src/states/phm_states.py:127-135`

### M-3: Inconsistent Field Naming
**From:** Reviewer-1, BUG-7
**Complexity:** Low (3-4 hours)
**Files:** `src/states/phm_states.py:102-115`

### M-4: Signal Processing Major Issues
**From:** Reviewer-5
**Complexity:** High (16-20 hours)
**Files:** `src/tools/*.py`
- Fix NormalizeOp, MelSpectrogramOp, SpectralCentroidOp, ZeroCrossingRateOp, etc.

### M-5: TSPN Model Training Issues
**From:** Reviewer-6
**Complexity:** Medium (6-8 hours)
**Files:** `src/agents/deep_model_train_agent.py`
- Fix memory leak in evaluation loop
- Fix dtype consistency check

### M-6: Missing Error Propagation
**From:** Reviewer-10
**Complexity:** Medium (4-6 hours)
**Files:** `src/agents/plan_agent.py`, `src/cases/case1.py`

### M-7: Data Factory Module Collision
**From:** Reviewer-10
**Complexity:** Medium (4-6 hours)
**Files:** `src/utils/data_factory_wrapper.py`

### M-8: Multi-Parent Node Handling
**From:** Reviewer-10
**Complexity:** Medium (6-8 hours)
**Files:** `src/agents/dataset_preparer_agent.py`

### M-9: Model Config Schema Inconsistency
**From:** Reviewer-8
**Complexity:** Low (2-3 hours)
**Files:** `config/model_tspn_basic.yaml`

### M-10: Test Coverage for Critical Modules
**From:** Reviewer-9
**Complexity:** High (24-32 hours)
**Files:** `tests/`
- Add tests for dag_init_agent
- Add tests for deep_model_train_agent error paths
- Add tests for phm_outer_graph

---

## LOW Priority Fixes (Week 6+)

### L-1: Redundant Lambda Wrappers
**From:** Reviewer-3, Bug #7
**Complexity:** Low (2-3 hours)
**Files:** `src/phm_outer_graph.py`

### L-2: Signal Processing Minor Issues
**From:** Reviewer-5
**Complexity:** Medium (8-12 hours)
**Files:** `src/tools/*.py`
- Fix EntropyOp, PowerToDecibelOp, TimeDelayEmbeddingOp, etc.

### L-3: Redundant torch Import
**From:** Reviewer-6, Bug #1
**Complexity:** Low (1 hour)
**Files:** `src/model/explainable/tspn.py`

### L-4: Unused Variables and Code Cleanup
**From:** Reviewer-6, Reviewer-1
**Complexity:** Low (4-6 hours)
**Files:** Multiple

### L-5: Configuration Code Quality
**From:** Reviewer-8, Issues #9-13
**Complexity:** Low (6-8 hours)
**Files:** `config/*.yaml`, `src/cases/case1.py`

### L-6: Logger Cleanup Issues
**From:** Reviewer-10, Issue #9
**Complexity:** Low (2-3 hours)
**Files:** `src/cases/case1.py`

### L-7: Comprehensive Test Coverage
**From:** Reviewer-9
**Complexity:** Very High (40-60 hours)
**Files:** `tests/`

---

## Dependency Graph

```
C-1 (get_llm)
  -> No dependencies, blocks agents

C-2 (State Mutation) [HIGH COMPLEXITY]
  -> No dependencies
  -> H-3, H-9, H-12 depend on this

C-3 (Iteration Limit) [MEDIUM COMPLEXITY]
  -> No dependencies

C-4 (Attribute Name)
  -> No dependencies

C-5 (Orphaned Methods) [MEDIUM COMPLEXITY]
  -> No dependencies

C-6 (Parent Type) [MEDIUM COMPLEXITY]
  -> C-7 depends on this

C-7 (Cycle Detection) [MEDIUM COMPLEXITY]
  -> Requires C-6
  -> H-10 depends on this

C-8 (Hardcoded Paths) [LOW COMPLEXITY]
  -> No dependencies

H-1 (Leaves Update) [MEDIUM COMPLEXITY]
  -> No dependencies

H-2 (run_executor Flag) [LOW COMPLEXITY]
  -> No dependencies

H-3 (API Key Validation) [MEDIUM COMPLEXITY]
  -> No dependencies

H-4 (Export JSON) [LOW COMPLEXITY]
  -> No dependencies

H-5 (Pydantic v2) [MEDIUM COMPLEXITY]
  -> No dependencies

H-6 (.env Loading) [MEDIUM COMPLEXITY]
  -> No dependencies

H-7 (Signal Processing) [HIGH COMPLEXITY]
  -> No dependencies

H-8 (TSPN Model) [MEDIUM COMPLEXITY]
  -> No dependencies

H-9 (Orchestration) [MEDIUM COMPLEXITY]
  -> Requires C-2

H-10 (Graph Implementation) [MEDIUM COMPLEXITY]
  -> Requires C-6, C-7

H-11 (Config Resolution) [MEDIUM COMPLEXITY]
  -> No dependencies

H-12 (Integration) [MEDIUM COMPLEXITY]
  -> Requires C-2
```

---

## Recommended Fix Order

### Sprint 1 (Week 1): Critical System Blockers
1. **C-1** (get_llm) - Unblocks all agents
2. **C-4** (Attribute name) - Quick win, removes crash
3. **C-8** (Hardcoded paths) - Unblocks multi-user testing
4. **C-2** (State mutation) - High complexity but critical

### Sprint 2 (Week 2): Critical Safety
5. **C-3** (Iteration limit) - Prevents infinite loops
6. **C-6** (Parent type) - Foundation for other fixes
7. **C-7** (Cycle detection) - Depends on C-6
8. **C-5** (Orphaned methods) - Code correctness

### Sprint 3 (Week 3): High Priority Data Integrity
9. **H-1** (Leaves update)
10. **H-2** (run_executor flag)
11. **H-3** (API key validation)
12. **H-5** (Pydantic v2)

### Sprint 4 (Week 4): High Priority Module Fixes
13. **H-7** (Signal processing critical)
14. **H-8** (TSPN model)
15. **H-9** (Orchestration)
16. **H-10** (Graph implementation)

### Sprint 5 (Week 5): Medium Priority & Integration
17. **H-11** (Config resolution)
18. **H-12** (Integration issues)
19. **H-6** (.env loading)

### Sprint 6+ (Week 6+): Lower Priority
20. M-1 through M-10 (Medium priority fixes)
21. L-1 through L-7 (Low priority and comprehensive testing)

---

## Testing Recommendations

### Unit Tests Required
- State management: tracker, parent normalization, cycle detection
- Graph operations: leaves update, topological sort
- Configuration: env loading, API key validation
- Signal processing: Each operator with edge cases

### Integration Tests Required
- Full builder graph with iteration limits
- State persistence and loading
- LangGraph vs fallback equivalence
- End-to-end workflow with error recovery

### Property-Based Tests Recommended
- State invariants after mutations
- Graph properties (acyclic, correct leaves)
- Numerical stability properties

---

## End of Mitigation Plan
