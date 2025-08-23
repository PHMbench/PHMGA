# PHMGA Signal Processing Tools Documentation

This document provides comprehensive guidance for the PHMGA signal processing operator system.

## Architecture Overview

The signal processing system is built on a modular operator architecture with automatic registration and discovery. All operators inherit from base classes that define their behavior and integration patterns.

### Signal Dimension Conventions

- `B`: 批处理大小 (Batch size)，即样本数量
- `L`: 信号长度 (Length)
- `C`: 通道数 (Channels)
- `F`: 频率轴维度
- `T`: 时间轴维度 (帧数)
- `S`: 小波变换尺度轴维度
- `M`: Mel 频谱带数量
- `N`: Patch 数量
- `P`: Patch 长度
- `C'`: 提取的特征数量

### Operator Categories

- **`EXPAND` (rank ↑)**: 新增轴或将单轴拆分为多轴，增加数据维度
- **`TRANSFORM` (rank =)**: 轴数不变，只改变域或数值，保持维度
- **`AGGREGATE` (rank ↓)**: 去掉轴或汇聚到向量/标量，减少维度
- **`MULTI-VARIABLE`**: 接收多个输入，处理多个节点
- **`DECISION`**: 输出判断，通常是流程的终点

## Operator Registry System

### Registration Mechanism

```python
from src.tools.signal_processing_schemas import register_op, TransformOp

@register_op
class CustomOp(TransformOp):
    """Custom signal processing operator."""

    op_name: ClassVar[str] = "custom"
    description: ClassVar[str] = "Custom signal processing operation"
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, L, C)"

    # Operator parameters
    param1: float = Field(1.0, description="Custom parameter")

    def execute(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Implement custom processing logic."""
        return x * self.param1
```

### Operator Discovery

```python
from src.tools.signal_processing_schemas import OP_REGISTRY, get_operator

# List all available operators
available_ops = list(OP_REGISTRY.keys())
print(f"Available operators: {available_ops}")

# Get specific operator class
fft_op_class = get_operator("fft")

# Instantiate operator
fft_op = fft_op_class()

# Apply to signal
result = fft_op.execute(signal_data)
```

### Base Classes

#### PHMOperator (Abstract Base)

```python
class PHMOperator(BaseModel):
    """Abstract base class for all signal processing operators."""

    # Class-level metadata
    op_name: ClassVar[str]  # Unique operator identifier
    description: ClassVar[str]  # Human-readable description
    input_spec: ClassVar[str]  # Expected input shape specification
    output_spec: ClassVar[str]  # Expected output shape specification
    rank_class: ClassVar[RankClass]  # Operator category

    # Instance parameters
    parent: str = Field("", description="Parent node identifier")

    # Runtime metadata (set during execution)
    in_shape: tuple = Field(default=(), description="Actual input shape")
    out_shape: tuple = Field(default=(), description="Actual output shape")

    def execute(self, x: np.ndarray, **kwargs) -> np.ndarray | dict:
        """Execute the operator on input data."""
        raise NotImplementedError
```

#### Specialized Base Classes

```python
class ExpandOp(PHMOperator):
    """Base class for dimensionality-increasing operators."""
    rank_class: ClassVar[RankClass] = "EXPAND"

class TransformOp(PHMOperator):
    """Base class for dimensionality-preserving operators."""
    rank_class: ClassVar[RankClass] = "TRANSFORM"

class AggregateOp(PHMOperator):
    """Base class for dimensionality-reducing operators."""
    rank_class: ClassVar[RankClass] = "AGGREGATE"

class MultiVariableOp(PHMOperator):
    """Base class for multi-input operators."""
    rank_class: ClassVar[RankClass] = "MultiVariable"

class DecisionOp(PHMOperator):
    """Base class for decision-making operators."""
    rank_class: ClassVar[RankClass] = "DECISION"
```

## Complete Operator Catalog

### EXPAND Operators (rank ↑)

这类操作会增加数据的维度，通常用于将一维时序信号转换为二维的时频表示或图像块。

| 工具 | 描述 | 输入维度 | 输出维度 |
| :--- | :--- | :--- | :--- |
| **Patching** | 将长信号切分成多个重叠或不重叠的短片段 (Patch)。 | `(B, L, C)` | `(B, N, P, C)` |
| **STFT** | 短时傅里叶变换，生成频谱图 (Spectrogram)，将时域信号转换为时频域表示。 | `(B, L, C)` | `(B, F, T, C)` |
| **CWT** | 连续小波变换，生成尺度图 (Scalogram)，提供信号在不同尺度下的时频分析。 | `(B, L, C)` | `(B, S, L, C)` |
| **Mel Spectrogram** | Mel 频谱图，在频率轴上使用 Mel 尺度，更符合人类听觉特性。 | `(B, L, C)` | `(B, M, T, C)` |
| **Spectrogram** | 频谱图，是 STFT 结果的幅值平方。 | `(B, L, C)` | `(B, F, T, C)` |
| **VQT** | 可变Q变换，一种在对数频率尺度上具有恒定分辨率的高级时频分析方法。 | `(B, L, C)` | `(B, Q, T, C)` |
| **Time-Delay Embedding** | 时间延迟嵌入，用于从时间序列重构相空间。 | `(B, L, C)` | `(B, L', D, C)` |
| **EMD** | 经验模态分解，将信号分解为多个本征模态函数（IMF）。 | `(B, L, C)` | `(B, L, I, C)` |
| **VMD** | 变分模态分解，一种比 EMD 更稳健的信号分解方法。 | `(B, L, C)` | `(B, L, K, C)` |
| **Wigner-Ville** | 维格纳-威利分布，一种高分辨率时频分析方法。 | `(B, L, C)` | `(B, L, F, C)` |
| **Scalogram** | 尺度图，是 CWT 结果的幅值或幅值平方。 | `(B, L, C)` | `(B, S, L, C)` |

#### Usage Examples

**STFT (Short-Time Fourier Transform):**
```python
from src.tools.expand_schemas import STFTOp

stft_op = STFTOp(
    fs=1000,        # Sampling frequency
    nperseg=256,    # Window length
    noverlap=128,   # Overlap length
    window='hann'   # Window type
)

spectrogram = stft_op.execute(signal)  # (B, L, C) -> (B, F, T, C)
```

**Time-Delay Embedding:**
```python
from src.tools.expand_schemas import TimeDelayEmbeddingOp

tde_op = TimeDelayEmbeddingOp(
    dimension=3,    # Embedding dimension
    delay=4        # Time delay
)

phase_space = tde_op.execute(signal)  # (B, L, C) -> (B, L', D, C)
```

### TRANSFORM Operators (rank =)

这类操作在不改变数据维度数量的前提下，对信号进行变换或处理。

| 工具 | 描述 | 输入维度 | 输出维度 |
| :--- | :--- | :--- | :--- |
| **Normalize** | 对信号进行归一化处理，如 Z-score 标准化或 Min-Max 缩放。 | `(B, L, C)` | `(B, L, C)` |
| **Detrend** | 移除信号中的趋势项，通常是线性的。 | `(B, L, C)` | `(B, L, C)` |
| **FFT** | 快速傅里叶变换，将时域信号转换为频域表示。 | `(B, L, C)` | `(B, F, C)` |
| **PSD** | 功率谱密度，使用 Welch 方法计算信号的功率谱。 | `(B, L, C)` | `(B, F, C)` |
| **Integrate** | 计算信号的累积积分（例如，从加速度到速度）。 | `(B, L, C)` | `(B, L, C)` |
| **Differentiate** | 计算信号的差分（例如，从速度到加速度）。 | `(B, L, C)` | `(B, L-1, C)` |
| **Power to dB** | 将功率谱或频谱图转换为分贝（dB）单位。 | `(B, F, ...)` | `(B, F, ...)` |
| **PCA** | 主成分分析，用于特征降维。 | `(B, C')` | `(B, n_components)` |
| **Adaptive Filter** | 自适应滤波，用于从信号中去除噪声。 | `Dict['d', 'x']` | `Dict['y', 'e', 'w']` |
| **Savitzky-Golay Filter** | SG 滤波器，一种强大的平滑去噪方法。 | `(B, L, C)` | `(B, L, C)` |
| **Cepstrum** | 倒谱分析，用于检测信号中的谐波成分。 | `(B, L, C)` | `(B, L, C)` |
| **Filter** | 使用滤波器（如低通、高通、带通）去除或保留特定频率成分。 | `(B, L, C)` | `(B, L, C)` |
| **Denoise (Wavelet)** | 使用小波阈值去噪，一种先进的去噪方法。 | `(B, L, C)` | `(B, L, C)` |
| **Hilbert Envelope** | 通过希尔伯特变换计算信号的解析信号，并提取其包络。 | `(B, L, C)` | `(B, L, C)` |
| **Resample** | 将信号重采样到新的长度。 | `(B, L, C)` | `(B, L_new, C)` |

#### Usage Examples

**FFT (Fast Fourier Transform):**
```python
from src.tools.transform_schemas import FFTOp

fft_op = FFTOp()
frequency_spectrum = fft_op.execute(signal)  # (B, L, C) -> (B, F, C)
```

**Digital Filtering:**
```python
from src.tools.transform_schemas import FilterOp

filter_op = FilterOp(
    filter_type="bandpass",
    low_freq=50,
    high_freq=200,
    fs=1000,
    order=4
)

filtered_signal = filter_op.execute(signal)  # (B, L, C) -> (B, L, C)
```

### AGGREGATE Operators (rank ↓)

这类操作会减少数据的维度，通常用于从信号中提取紧凑的特征表示。

| 工具 | 描述 | 输入维度 | 输出维度 |
| :--- | :--- | :--- | :--- |
| **时域统计特征** | 计算信号的各种统计量，如均值、方差、峰度等。 | `(B, L, C)` | `(B, C')` |
| **过零率 (ZCR)** | 计算信号穿过零点的频率。 | `(B, L, C)` | `(B, C')` |
| **峰峰值 (P2P)** | 计算信号的最大值和最小值之差。 | `(B, L, C)` | `(B, C')` |
| **Hjorth 参数** | 计算信号的活动性、移动性和复杂性。 | `(B, L, C)` | `(B, 3, C)` |
| **近似熵 (ApEn)** | 量化信号的规律性和复杂度。 | `(B, L, C)` | `(B, C')` |
| **排列熵 (PermEn)** | 基于排序模式量化信号的复杂度。 | `(B, L, C)` | `(B, C')` |
| **频域统计特征** | 计算功率谱密度(PSD)的统计量。 | `(B, F, C)` | `(B, C')` |
| **谱质心** | 计算频谱的能量中心，是重要的频域特征。 | `(B, F, C)` | `(B, C')` |
| **谱偏度** | 计算频谱的偏度。 | `(B, F, C)` | `(B, C')` |
| **谱峰度** | 计算频谱的峰度。 | `(B, F, C)` | `(B, C')` |
| **谱平坦度** | 衡量频谱的音调特性。 | `(B, F, C)` | `(B, C')` |
| **频带功率 (Band Power)** | 计算特定频带内的平均功率。 | `(B, F, C)` | `(B, C')` |

#### Statistical Features Formulas

以下是常用的时域统计特征，它们将长度为 `L` 的信号段聚合为单个标量值。

| 特征 | 公式 |
| :--- | :--- |
| **均值 (Mean)** | $$\mu = \frac{1}{L}\sum_{i=1}^{L} x_i$$ |
| **标准差 (Std)** | $$\sigma = \sqrt{\frac{1}{L}\sum_{i=1}^{L} (x_i - \mu)^2}$$ |
| **方差 (Var)** | $$\sigma^2 = \frac{1}{L}\sum_{i=1}^{L} (x_i - \mu)^2$$ |
| **熵 (Entropy)** | $$H(x) = -\sum_{i=1}^{N} p(x_i) \log p(x_i)$$ |
| **最大值 (Max)** | $$\max(x) = \max_{i} x_i$$ |
| **最小值 (Min)** | $$\min(x) = \min_{i} x_i$$ |
| **绝对值均值 (AbsMean)** | $$\text{abs\_mean}(x) = \frac{1}{L}\sum_{i=1}^{L} |x_i|$$ |
| **均方根 (RMS)** | $$\text{rms}(x) = \sqrt{\frac{1}{L}\sum_{i=1}^{L} x_i^2}$$ |
| **偏度 (Skewness)** | $$\text{skewness}(x) = \frac{\frac{1}{L}\sum_{i=1}^{L} (x_i - \mu)^3}{\sigma^3}$$ |
| **峰度 (Kurtosis)** | $$\text{kurtosis}(x) = \frac{\frac{1}{L}\sum_{i=1}^{L} (x_i - \mu)^4}{\sigma^4}$$ |
| **波形因子 (Shape Factor)** | $$\text{shape\_factor}(x) = \frac{\text{rms}(x)}{\text{abs\_mean}(x)}$$ |
| **峰值因子 (Crest Factor)** | $$\text{crest\_factor}(x) = \frac{\max_{i} |x_i|}{\text{rms}(x)}$$ |
| **裕度因子 (Clearance Factor)** | $$\text{clearance\_factor}(x) = \frac{\max_{i} |x_i|}{\left(\frac{1}{L}\sum_{i=1}^{L} \sqrt{|x_i|}\right)^2}$$ |
| **峰峰值 (Peak-to-Peak)** | $$\text{p2p}(x) = \max(x) - \min(x)$$ |
| **过零率 (Zero-Crossing Rate)** | $$\text{zcr}(x) = \frac{1}{2(L-1)}\sum_{i=1}^{L-1} |\text{sgn}(x_i) - \text{sgn}(x_{i-1})|$$ |

#### Usage Examples

**Statistical Features:**
```python
from src.tools.aggregate_schemas import MeanOp, StdOp, RMSOp, KurtosisOp

# Extract multiple statistical features
features = {}
features['mean'] = MeanOp().execute(signal)
features['std'] = StdOp().execute(signal)  
features['rms'] = RMSOp().execute(signal)
features['kurtosis'] = KurtosisOp().execute(signal)
```

**Band Power Analysis:**
```python
from src.tools.aggregate_schemas import BandPowerOp

band_power_op = BandPowerOp(
    fs=1000,
    bands=[(0, 50), (50, 150), (150, 300)]  # Frequency bands
)

band_powers = band_power_op.execute(frequency_spectrum)
```

### MULTI-VARIABLE Operators

这类工具用于比较或组合来自不同处理分支的节点。

| 工具 | 描述 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| **Subtract** | 信号相减，计算两个信号的差值（残差）。 | 两个 `(B, L, C)` | `(B, L, C)` |
| **Arithmetic** | 在两个信号之间执行基本的数学运算（+,-,*,/）。 | 两个 `(B, L, C)` | `(B, L, C)` |
| **Element-wise Product** | 信号逐元素相乘，可用于调制或加窗。 | 两个 `(B, L, C)` | `(B, L, C)` |
| **Convolution** | 对信号和核执行一维卷积。 | `(B, L, C)` 和 `(K,)` | `(B, L', C)` |
| **Transfer Function** | 估计输入输出信号间的传递函数。 | `(L,)` 和 `(L,)` | 字典 |
| **Phase Difference** | 计算两个信号在频域的相位差。 | 两个 `(B, F, C)` | `(B, F, C)` |
| **Coherence** | 相干函数，分析两个信号在频域的线性相关性。 | 两个 `(B, L, C)` | `(B, F, C)` |
| **DTW Distance** | 动态时间规整，计算两个不等长序列的相似度。 | `(L1, C)` 和 `(L2, C)` | 标量 |
| **Cross-Correlation** | 计算两个信号的互相关，分析其相似性与延迟。 | 两个 `(B, L, C)` | `(B, L_corr, C)` |
| **Distance** | 计算两个特征向量之间的距离（如欧氏距离）。 | 两个 `(B, C')` | `(B,)` |
| **Concatenate** | 沿指定轴拼接多个特征向量。 | 多个 `(B, C')` | `(B, C_new)` |
| **Compare** | 比较两个或多个特征向量/矩阵，例如计算距离或相似度。 | 多个 `(B, C')` | `(B, C'')` 或字典 |

#### Usage Examples

**Signal Comparison:**
```python
from src.tools.multi_schemas import CompareOp

compare_op = CompareOp(metrics=["euclidean", "cosine"])
comparison = compare_op.execute({"ref": reference_signals, "test": test_signals})
```

**Cross-Correlation:**
```python
from src.tools.multi_schemas import CrossCorrelationOp

cross_corr_op = CrossCorrelationOp(mode="full", normalize=True)
correlation = cross_corr_op.execute({"signal1": sig1, "signal2": sig2})
```

### DECISION Operators

这类工具通常是流程的终点，输出最终的结论。

| 工具 | 描述 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| **Threshold** | 判断输入值是否超过阈值。 | 标量 | 字典 (布尔值) |
| **Find Peaks** | 在信号或频谱中寻找峰值。 | `(L,)` | 字典 |
| **Change Point Detection** | 检测信号统计特性的突变点。 | `(L, C)` | 字典 |
| **Outlier Detection** | 使用孤立森林等算法检测特征集中的异常点。 | `(B, C')` | 字典 |
| **KS-Test** | KS检验，判断两个样本是否来自同一分布。 | `(L1,)` 和 `(L2,)` | 字典 |
| **Harmonic Analysis** | 从频谱中识别基频的谐波系列。 | 字典 | 字典 |
| **Rule-Based Decision** | 基于一组简单的规则做出决策。 | 字典 | 字典 (布尔值) |
| **Similarity Classifier** | 通过计算与参考特征的相似度来进行分类。 | 字典 | 字典 (字符串) |
| **Anomaly Scorer** | 基于与健康状态的距离计算异常分数。 | 字典 | 字典 (浮点数) |
| **Classifier** | 基于提取的特征进行分类，输出故障类型。 | `(B, C')` | 字符串标签 |
| **Scoring** | 对比参考和测试信号的特征，给出一个健康评分或异常分数。 | 多个 `(B, C')` | 浮点数值 |

## Custom Operator Development

### Basic Operator Template

```python
from src.tools.signal_processing_schemas import register_op, TransformOp
import numpy as np
from pydantic import Field
from typing import ClassVar

@register_op
class CustomNormalizeOp(TransformOp):
    """Custom normalization operator with configurable method."""

    op_name: ClassVar[str] = "custom_normalize"
    description: ClassVar[str] = "Custom signal normalization with multiple methods"
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, L, C)"

    # Operator parameters
    method: str = Field("zscore", description="Normalization method: 'zscore', 'minmax', 'robust'")
    axis: int = Field(-2, description="Axis along which to normalize")

    def execute(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Execute custom normalization."""

        if self.method == "zscore":
            mean = np.mean(x, axis=self.axis, keepdims=True)
            std = np.std(x, axis=self.axis, keepdims=True)
            return (x - mean) / (std + 1e-8)

        elif self.method == "minmax":
            min_val = np.min(x, axis=self.axis, keepdims=True)
            max_val = np.max(x, axis=self.axis, keepdims=True)
            return (x - min_val) / (max_val - min_val + 1e-8)

        elif self.method == "robust":
            median = np.median(x, axis=self.axis, keepdims=True)
            mad = np.median(np.abs(x - median), axis=self.axis, keepdims=True)
            return (x - median) / (mad + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

# Usage
custom_op = CustomNormalizeOp(method="robust", axis=-2)
normalized = custom_op.execute(signal_data)
```

### Multi-Variable Operator Template

```python
@register_op
class CrossCorrelationOp(MultiVariableOp):
    """Cross-correlation analysis between multiple signals."""

    op_name: ClassVar[str] = "cross_correlation"
    description: ClassVar[str] = "Compute cross-correlation between signal pairs"
    input_spec: ClassVar[str] = "Dict[str, (B, L, C)]"
    output_spec: ClassVar[str] = "(B, 2*L-1, C)"

    mode: str = Field("full", description="Correlation mode: 'full', 'valid', 'same'")
    normalize: bool = Field(True, description="Normalize correlation coefficients")

    def execute(self, x: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Execute cross-correlation analysis."""

        if len(x) != 2:
            raise ValueError("Cross-correlation requires exactly 2 input signals")

        signals = list(x.values())
        sig1, sig2 = signals[0], signals[1]

        # Implementation logic here
        return correlation_result
```

### Decision Operator Template

```python
@register_op
class ThresholdClassifierOp(DecisionOp):
    """Simple threshold-based classifier."""

    op_name: ClassVar[str] = "threshold_classifier"
    description: ClassVar[str] = "Classify signals based on feature thresholds"
    input_spec: ClassVar[str] = "(B, F)"  # Features
    output_spec: ClassVar[str] = "Dict[str, Any]"

    thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature thresholds for classification"
    )

    def execute(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Execute threshold-based classification."""
        
        # Classification logic here
        return {
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "decision_rule": "threshold_based"
        }
```

## Integration with PHMGA System

### Agent Integration

Operators are automatically discovered and used by agents:

```python
from src.agents.execute_agent import execute_agent
from src.states.phm_states import PHMState

# Operators used through detailed plan
state = PHMState(
    detailed_plan=[
        {"parent": "ch1", "op_name": "fft", "params": {}},
        {"parent": "fft_ch1", "op_name": "mean", "params": {"axis": -2}},
        {"parent": "mean_fft_ch1", "op_name": "compare", "params": {"metrics": ["euclidean"]}}
    ]
)

# Execute agent automatically applies operators
result = execute_agent(state)
```

### Error Handling

Operators include comprehensive error handling:

```python
try:
    result = operator.execute(input_data)
except ValueError as e:
    print(f"Input validation error: {e}")
    state.dag_state.error_log.append(f"Operator {operator.op_name} validation failed: {e}")
except Exception as e:
    print(f"Execution error: {e}")
    state.dag_state.error_log.append(f"Operator {operator.op_name} execution failed: {e}")
```

## Performance Optimization

### Memory Management

```python
def process_large_signals(signals: List[np.ndarray], operators: List[PHMOperator]) -> List[np.ndarray]:
    """Process large signals with memory management."""

    results = []
    
    for signal in signals:
        # Process in chunks if signal is too large
        if signal.nbytes > 100 * 1024 * 1024:  # 100MB threshold
            chunk_size = 1024
            chunks = [signal[:, i:i+chunk_size, :] for i in range(0, signal.shape[1], chunk_size)]
            
            chunk_results = []
            for chunk in chunks:
                for op in operators:
                    chunk = op.execute(chunk)
                chunk_results.append(chunk)
            
            # Concatenate results
            result = np.concatenate(chunk_results, axis=1)
        else:
            # Process normally
            result = signal
            for op in operators:
                result = op.execute(result)
        
        results.append(result)
    
    return results
```

### Efficient Pipeline Construction

```python
def create_processing_pipeline(operator_configs: List[Dict]) -> List[PHMOperator]:
    """Create optimized operator pipeline."""
    
    operators = []
    for config in operator_configs:
        op_class = get_operator(config["op_name"])
        operator = op_class(**config.get("params", {}))
        operators.append(operator)
    
    return operators

# Usage
pipeline_config = [
    {"op_name": "normalize", "params": {"method": "zscore"}},
    {"op_name": "filter", "params": {"filter_type": "bandpass", "low_freq": 10, "high_freq": 100}},
    {"op_name": "fft", "params": {}},
    {"op_name": "mean", "params": {"axis": -2}}
]

pipeline = create_processing_pipeline(pipeline_config)
```

## Testing Patterns

### Unit Testing

```python
def test_custom_operator():
    """Test custom operator implementation."""
    
    # Create test data
    test_signal = np.random.randn(2, 1024, 1)
    
    # Test operator
    op = CustomNormalizeOp(method="zscore")
    result = op.execute(test_signal)
    
    # Validate results
    assert result.shape == test_signal.shape
    assert np.allclose(np.mean(result, axis=-2), 0, atol=1e-6)
    assert np.allclose(np.std(result, axis=-2), 1, atol=1e-6)
    
    print("✅ Custom operator test passed!")
```

### Integration Testing

```python
def test_operator_chain():
    """Test chained operator execution."""
    
    signal = np.random.randn(1, 1024, 1)
    
    # Create operator chain
    ops = [
        get_operator("normalize")(),
        get_operator("fft")(),
        get_operator("mean")(axis=-2)
    ]
    
    # Apply operators sequentially
    result = signal
    for op in ops:
        result = op.execute(result)
    
    # Validate final result
    assert result.shape[0] == 1  # Batch preserved
    assert result.shape[1] == 1  # Aggregated to single value
    
    print("✅ Operator chain test passed!")
```

This comprehensive signal processing toolkit provides the foundation for flexible, powerful, and maintainable signal analysis workflows in the PHMGA system.