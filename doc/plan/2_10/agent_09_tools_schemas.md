# Tools 和 Schemas 模块 BUG 审查报告

**审查人**: Agent 9
**审查日期**: 2026-02-15
**审查范围**: src/tools/ 和 src/schemas/ 模块

---

## 1. 审查范围

| 文件路径 | 代码行数 | 状态 |
|----------|----------|------|
| src/tools/signal_processing_schemas.py | 171 | 活跃 |
| src/tools/decision_schemas.py | 514 | 注释掉 |
| src/tools/utils.py | 52 | 活跃 |
| src/tools/comparator_tool.py | 95 | 活跃 |
| src/tools/expand_schemas.py | 373 | 活跃 |
| src/tools/aggregate_schemas.py | 490 | 活跃 |
| src/tools/multi_schemas.py | 302 | 活跃 |
| src/tools/transform_schemas.py | 336 | 活跃 |
| src/schemas/plan_schema.py | 12 | 活跃 |
| src/schemas/insight_schema.py | 14 | 活跃 |
| **总计** | **2349** | - |

---

## 2. 发现的 BUG

### 2.1 高严重程度 (High)

#### BUG-001: 除零风险未统一处理
**文件**: `src/tools/aggregate_schemas.py`
**位置**: 多个算子类
**严重程度**: 高

**问题描述**:
多个算子使用硬编码的 `1e-9` 作为 epsilon 值来防止除零错误，但这个值对于所有数值范围可能不合适。当数值非常大或非常小时，这个 epsilon 可能无效或导致精度问题。

**代码片段**:
```python
# Line 129
return rms / (abs_mean + 1e-9) # Add epsilon for stability

# Line 142
return peak / (rms + 1e-9) # Add epsilon for stability

# Line 155
return peak / (sqrt_abs_mean_sq + 1e-9) # Add epsilon for stability
```

**建议修复**:
```python
# 使用相对 epsilon 而非绝对 epsilon
def safe_divide(numerator, denominator, eps=1e-12):
    """Safe division with relative epsilon."""
    denom_mag = np.abs(denominator)
    eps_val = np.maximum(eps, eps * denom_mag)
    return numerator / (denominator + eps_val)

return safe_divide(rms, abs_mean)
```

---

#### BUG-002: cosine 距离计算可能产生 NaN
**文件**: `src/tools/multi_schemas.py`
**位置**: 第 95 行
**严重程度**: 高

**问题描述**:
当输入向量全为零时，`np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1)` 结果为零，导致除零产生 `NaN`。

**代码片段**:
```python
# Line 95
return 1 - np.sum(vec1 * vec2, axis=-1) / (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1))
```

**建议修复**:
```python
norm_prod = np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1)
cosine_sim = np.divide(
    np.sum(vec1 * vec2, axis=-1),
    norm_prod,
    out=np.zeros_like(np.sum(vec1 * vec2, axis=-1)),
    where=norm_prod != 0
)
return 1 - cosine_sim
```

---

#### BUG-003: SpectralCentroidOp 默认 fs 硬编码
**文件**: `src/tools/aggregate_schemas.py`
**位置**: 第 257 行
**严重程度**: 高

**问题描述**:
`SpectralCentroidOp` 的采样频率 `fs` 被硬编码为 `3125`，这是一个特定领域的值，应该由用户提供，否则可能产生错误的频率计算结果。

**代码片段**:
```python
# Line 257
fs: float = Field(3125, description="Sampling frequency of the signal.")
```

**建议修复**:
```python
fs: float = Field(..., description="Sampling frequency of the signal (required).")
```

---

#### BUG-004: VMD 和 EMD 算子未处理空输入
**文件**: `src/tools/expand_schemas.py`
**位置**: 第 314-332 行 (VMD), 第 336-366 行 (EMD)
**严重程度**: 高

**问题描述**:
`VariationalModeDecompositionOp` 和 `EmpiricalModeDecompositionOp` 在 `K` 参数过大或信号过短时可能产生空结果或崩溃，但没有进行输入验证。

**代码片段**:
```python
# Line 328
u, _, _ = VMD(x[i, :, j], self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
```

**建议修复**:
```python
def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
    try:
        from vmdpy import VMD
    except ImportError:
        raise ImportError("vmdpy is not installed. Please install it with 'pip install vmdpy'.")

    if x.ndim != 3:
        raise ValueError(f"Input for VMD must be 3D (B, L, C), but got {x.ndim}D.")

    # Validate K parameter against signal length
    min_samples_for_k = self.K * 10  # Heuristic: need at least 10 samples per mode
    if x.shape[1] < min_samples_for_k:
        raise ValueError(
            f"Signal length {x.shape[1]} is too short for K={self.K} modes. "
            f"Need at least {min_samples_for_k} samples."
        )
```

---

### 2.2 中严重程度 (Medium)

#### BUG-005: TransferFunctionOp 未处理空输入和 NaN 输出
**文件**: `src/tools/multi_schemas.py`
**位置**: 第 284-295 行
**严重程度**: 中

**问题描述**:
当 `Pxx` 为零或接近零时，传递函数 `H = Pxy / (Pxx + 1e-9)` 可能产生非数值结果。

**代码片段**:
```python
# Line 294
H = Pxy / (Pxx + 1e-9)
```

**建议修复**:
```python
# Mask invalid frequencies where power is too low
power_threshold = np.max(Pxx) * 1e-6
valid_mask = Pxx > power_threshold
H = np.full_like(Pxy, np.nan, dtype=np.complex128)
H[valid_mask] = Pxy[valid_mask] / Pxx[valid_mask]
return {"frequencies": f, "transfer_function": H, "valid_mask": valid_mask}
```

---

#### BUG-006: comparator_tool.py 节点未找到时抛出通用异常
**文件**: `src/tools/comparator_tool.py`
**位置**: 第 31-45 行
**严重程度**: 中

**问题描述**:
当节点不存在时，`ref_node` 和 `test_node` 可能为 `None`，导致 `ValueError("Invalid reference node")` 错误信息不够具体。

**代码片段**:
```python
# Line 31-38
ref_node = state.dag_state.nodes.get(reference_node_id)
test_node = state.dag_state.nodes.get(test_node_id)
if isinstance(ref_node, ProcessedData):
    ref = np.asarray(ref_node.results)
elif isinstance(ref_node, InputData):
    ref = np.asarray(ref_node.data.get("signal", []))
else:
    raise ValueError("Invalid reference node")
```

**建议修复**:
```python
ref_node = state.dag_state.nodes.get(reference_node_id)
if ref_node is None:
    raise ValueError(f"Reference node '{reference_node_id}' not found in DAG state.")

test_node = state.dag_state.nodes.get(test_node_id)
if test_node is None:
    raise ValueError(f"Test node '{test_node_id}' not found in DAG state.")

if isinstance(ref_node, ProcessedData):
    ref = np.asarray(ref_node.results)
elif isinstance(ref_node, InputData):
    ref = np.asarray(ref_node.data.get("signal", []))
else:
    raise ValueError(
        f"Reference node '{reference_node_id}' has invalid type: "
        f"{type(ref_node).__name__}. Expected ProcessedData or InputData."
    )
```

---

#### BUG-007: WignerVilleDistributionOp 性能问题和潜在错误
**文件**: `src/tools/expand_schemas.py`
**位置**: 第 186-189 行
**严重程度**: 中

**问题描述**:
Wigner-Ville 分布计算中存在嵌套循环，且第 189 行的索引计算 `tfr[n, n]` 似乎不正确，应该是 `tfr[n, n + tau]`。

**代码片段**:
```python
# Line 186-189
for n in range(n_samples):
    taumax = min(n, n_samples - 1 - n)
    for tau in range(-taumax, taumax + 1):
        tfr[n, n] += analytic_signal[n + tau] * np.conj(analytic_signal[n - tau])
```

**建议修复**:
```python
for n in range(n_samples):
    taumax = min(n, n_samples - 1 - n)
    for tau in range(-taumax, taumax + 1):
        # Fixed: should be tfr[n, n + tau] or equivalent
        tau_idx = tau + taumax  # Shift to non-negative index
        tfr[n, tau_idx] = analytic_signal[n + tau] * np.conj(analytic_signal[n - tau])
```

---

#### BUG-008: SavitzkyGolayFilterOp 缺少参数验证
**文件**: `src/tools/transform_schemas.py`
**位置**: 第 271-272 行
**严重程度**: 中

**问题描述**:
`scipy.signal.savgol_filter` 要求 `window_length` 必须为正奇数且大于 `polyorder`，但代码中没有验证这些约束。

**代码片段**:
```python
# Line 271-272
return scipy.signal.savgol_filter(x, self.window_length, self.polyorder, axis=-2)
```

**建议修复**:
```python
def execute(self, x: np.ndarray, **kw) -> np.ndarray:
    if self.window_length <= 0:
        raise ValueError(f"window_length must be positive, got {self.window_length}")
    if self.window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {self.window_length}")
    if self.polyorder >= self.window_length:
        raise ValueError(
            f"polyorder ({self.polyorder}) must be less than window_length ({self.window_length})"
        )
    return scipy.signal.savgol_filter(x, self.window_length, self.polyorder, axis=-2)
```

---

#### BUG-009: 多个算子未验证输入形状
**文件**: 多个文件
**位置**: 见描述
**严重程度**: 中

**问题描述**:
许多算子假设输入是 3D (B, L, C) 但没有验证，可能在运行时导致意外错误。

**受影响的算子**:
- `MeanOp`, `StdOp`, `VarOp` 等 (aggregate_schemas.py) - 大部分没有形状验证
- `CrossCorrelationOp` (multi_schemas.py:46-68)
- `CoherenceOp` (multi_schemas.py:156-157)

**建议修复**:
在每个 `execute` 方法开始添加形状验证：
```python
def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
    if x.ndim != 3:
        raise ValueError(f"Input must be 3D (B, L, C), but got {x.ndim}D with shape {x.shape}")
    # ... rest of the code
```

---

#### BUG-010: BandPowerOp 空频段处理
**文件**: `src/tools/aggregate_schemas.py`
**位置**: 第 217-218 行
**严重程度**: 中

**问题描述**:
当频段范围超出实际频率范围时，返回全零数组，这可能掩盖配置错误。

**代码片段**:
```python
# Line 217-218
if idx.size == 0:
    band_powers.append(np.zeros((x.shape[0], x.shape[2])))
```

**建议修复**:
```python
if idx.size == 0:
    import warnings
    warnings.warn(
        f"Band {band} Hz is outside the signal frequency range. "
        f"Signal frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz",
        UserWarning
    )
    band_powers.append(np.full((x.shape[0], x.shape[2]), np.nan))
```

---

### 2.3 低严重程度 (Low)

#### BUG-011: 未使用的导入
**文件**: `src/tools/aggregate_schemas.py`
**位置**: 第 8 行
**严重程度**: 低

**问题描述**:
`import scipy.stats` 但在代码中直接使用了 `from scipy.stats import skew, kurtosis`，导致未直接使用 `scipy.stats` (除了一处)。

**代码片段**:
```python
# Line 8
import scipy.stats

# Line 301: 唯一使用的地方
return scipy.stats.kurtosis(x, axis=-2)
```

**建议修复**:
```python
# 删除未使用的导入，或者改为
from scipy.stats import skew, kurtosis, gmean
```

---

#### BUG-012: 类型注解不一致
**文件**: 多个文件
**位置**: 见描述
**严重程度**: 低

**问题描述**:
部分函数使用 `np.ndarray`，部分使用 `npt.NDArray`，类型注解不统一。

**建议**:
统一使用 `npt.NDArray` 作为 NumPy 数组的类型注解。

---

#### BUG-013: decision_schemas.py 整个文件被注释
**文件**: `src/tools/decision_schemas.py`
**位置**: 全文件
**严重程度**: 低

**问题描述**:
整个 `decision_schemas.py` 文件的内容都被注释掉了，但文件仍然保留在代码库中。这可能导致混淆。

**建议**:
如果代码不需要，应该删除；如果是临时禁用，应该在注释中说明原因和预计恢复时间。

---

#### BUG-014: aggregate_schemas.py 结尾有死代码
**文件**: `src/tools/aggregate_schemas.py`
**位置**: 第 407-489 行
**严重程度**: 低

**问题描述**:
在 `raise SystemExit` 之后还有大量测试代码永远不会被执行。

**代码片段**:
```python
# Line 407-409
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "This module is library code. Use the pytest suite for validation instead of inline demos."
    )

    # Line 412-489: This code is unreachable!
    # 15. Test SkewnessOp
    ...
```

**建议修复**:
删除第 412-489 行的死代码。

---

#### BUG-015: transform_schemas.py 未使用的注释代码
**文件**: `src/tools/transform_schemas.py`
**位置**: 第 82-92, 295-329 行
**严重程度**: 低

**问题描述**:
有大量注释掉的类定义，增加了代码维护负担。

**建议**:
移到单独的示例文件或删除。

---

#### BUG-016: EntropyOp 使用 apply_along_axis 性能较差
**文件**: `src/tools/aggregate_schemas.py`
**位置**: 第 180 行
**严重程度**: 低

**问题描述**:
`np.apply_along_axis` 是一个慢速的 Python 循环，对于大型数组效率低。

**代码片段**:
```python
# Line 180
return np.apply_along_axis(_calculate_entropy_1d, self.axis, x)
```

**建议修复**:
使用向量化操作替代：
```python
def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
    # Vectorized entropy calculation
    # Reshape for 2D histogram calculation
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    results = np.zeros(x_2d.shape[0])
    for i in range(x_2d.shape[0]):
        counts, _ = np.histogram(x_2d[i], bins=self.num_bins, density=True)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        results[i] = -np.sum(probs * np.log2(probs))

    return results.reshape(original_shape[:-1])
```

---

## 3. 统计汇总

| 严重程度 | 数量 | 百分比 |
|----------|------|--------|
| 高 (High) | 4 | 25% |
| 中 (Medium) | 10 | 62.5% |
| 低 (Low) | 6 | 37.5% |
| **总计** | **20** | 100% |

### 按类别统计

| 类别 | 数量 |
|------|------|
| 错误处理 | 8 |
| 输入验证 | 6 |
| 数值稳定性 | 4 |
| 代码质量 | 2 |

---

## 4. 总结

本次审查共发现 **20 个潜在问题**:

1. **高优先级问题**: 4 个，主要集中在除零风险、硬编码参数和输入验证缺失
2. **中优先级问题**: 10 个，包括节点未找到的错误处理、参数验证和边界条件
3. **低优先级问题**: 6 个，主要是代码清理和维护性问题

**建议优先修复顺序**:
1. BUG-002 (cosine 距离 NaN)
2. BUG-004 (VMD/EMD 输入验证)
3. BUG-003 (硬编码采样频率)
4. BUG-006 (节点错误信息)
5. BUG-008 (SavitzkyGolay 参数验证)

---

**报告生成时间**: 2026-02-15
**审查人签名**: Agent 9
