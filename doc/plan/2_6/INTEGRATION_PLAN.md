# Integration Plan: src/tools → src/model/explainable

## Context

This integration addresses **Gap 2** and **Gap 3** from `GAPS.md`:

- **Gap 2**: PlanAgent's DAG semantics lack "TSPN-friendly initialization" protocol - need explicit operator→token mapping table
- **Gap 3**: DAG→TSPN configuration is heuristic-based, not fully utilizing DAG structure (multi-channel/multi-nodes, multiple filter nodes, feature nodes)

The goal is to integrate the 58+ operators from `src/tools` into the `src/model/explainable` architecture.

---



### Code Changes After Upgrade

- **`src/phm_outer_graph.py`**: Remove `_FallbackGraph`, use real `StateGraph` from `langgraph`
- Update `langchain_core` imports (API changes between 0.1.x and 1.x)

---

## Main Integration Plan

### Current State

### src/tools (58+ operators)
- 5 operator categories: ExpandOp, TransformOp, AggregateOp, MultiVariableOp, DecisionOp
- Registration via `@register_op` decorator with global `OP_REGISTRY`
- Located in: `aggregate_schemas.py` (22), `transform_schemas.py` (16), `expand_schemas.py` (13), `multi_schemas.py` (10)

### src/model/explainable (4 operators)
- `ops.py`: Identity (I), HilbertEnvelope (HT), FFTMagnitude (FFT), WaveFilters (WF)
- `bridge.py`: Heuristic string matching via `_map_node_to_token()`
- `feature_ops.py`: 13 feature extractors

---

## Implementation Phases

### Phase 1: Create Unified Operator Catalog

**New file**: `src/model/explainable/operator_catalog.py`

Components:
- `TSPNTokenFamily` enum - defines token categories
- `OperatorMapping` dataclass - maps PHMGA op to TSPN token with param translation
- `OPERATOR_CATALOG` - main mapping dictionary
- `FEATURE_OPERATOR_CATALOG` - maps AggregateOp to feature_ops.py

**Key mappings**:
- Existing: fft→FFT, hilbert_envelope→HT, filter→WF, identity→I
- Transform: normalize→I, detrend→I, integrate→I, differentiate→I, psd→FFT
- Expand: stft→STFT, wavelet_transform→CWT, vmd→VMD, emd→EMD
- Aggregate: mean→Mean, std→Std, rms→RMS, etc.

### Phase 2: Extend TSPN Operators

**Modify**: `src/model/explainable/ops.py`

Add new differentiable operators:
1. **Normalize** (NORM) - z-score and min-max
2. **Detrend** (DT) - linear trend removal
3. **Integrate** (INT) - cumulative integral
4. **Differentiate** (DIFF) - discrete derivative
5. **STFT** - Short-Time Fourier Transform

Update `make_op()` factory.

### Phase 3: Enhance Bridge with Precise Mapping

**Modify**: `src/model/explainable/bridge.py`

- Use `operator_catalog` for lookups
- Add `_translate_operator_params()` for parameter translation
- Add `preserve_dag_topology` flag (default=True)
- Add `allow_duplicate_tokens` flag (default=True)
- Implement `_build_ops_from_nodes()` - preserves all DAG nodes
- Keep `_build_ops_deduplicated()` for backward compatibility

### Phase 4: Feature Operator Integration

**Modify**: `src/model/explainable/feature_ops.py`

Add missing features from `aggregate_schemas.py`:
- Var, PeakToPeak, ZeroCrossingRate
- SpectralCentroid, SpectralSkewness, SpectralKurtosis, SpectralFlatness
- HjorthParameters, AbsMean, CrestFactor, etc.

### Phase 5: Update Builder

**Modify**: `src/model/explainable/builder.py`

Handle new operator parameters in `build_tspn_from_config()`.

### Phase 6: Extend Configuration Schema

**Modify**: `src/model/explainable/config_schema.py`

Add to `ModelConfig`:
- `stft_init`, `norm_init` parameter dictionaries
- `preserve_topology`, `allow_duplicate_tokens` flags

---

## File Changes

### New Files
```
src/model/explainable/
└── operator_catalog.py
```

### Modified Files
```
src/model/explainable/
├── ops.py
├── bridge.py
├── builder.py
├── config_schema.py
└── feature_ops.py

src/phm_outer_graph.py          # For LangGraph 1.x migration
requirements.txt                 # Update dependency versions
```

---

## Verification

1. **Unit tests**: `tests/test_operator_catalog.py`
2. **Integration tests**: `tests/test_dag2tspn_extended.py`
3. **End-to-end**: `pytest tests/test_full_agent_flow_vibench_tspn_report.py -v`
4. **Backward compatibility**: `pytest tests/ -q`

---

## Execution Order

1. **First**: Fix dependencies (Gap 4) - run the pip install command above
2. **Second**: Update `src/phm_outer_graph.py` for LangGraph 1.x
3. **Third**: Implement main integration (Phases 1-6)
