# PHMGA Signal Processing Tools

The tools module provides a comprehensive suite of signal processing operators for the PHMGA system. These operators are organized by their effect on data dimensionality and functionality, enabling flexible and powerful signal analysis pipelines.

## Architecture Overview

The signal processing system is built on a modular operator architecture with automatic registration and discovery. All operators inherit from base classes that define their behavior and integration patterns.

### Operator Categories

- **`EXPAND` (Rank ↑)**: Increase data dimensionality (e.g., time-domain to time-frequency domain)
- **`TRANSFORM` (Rank =)**: Preserve dimensionality while changing domain or values (e.g., filtering, normalization)
- **`AGGREGATE` (Rank ↓)**: Reduce dimensionality to extract compact features (e.g., statistical measures)
- **`MULTI-VARIABLE`**: Process multiple input nodes for comparison or combination
- **`DECISION`**: Output classification results or decisions rather than signal data

### Core Design Principles

1. **Automatic Registration**: Operators are automatically discovered and registered using decorators
2. **Type Safety**: Comprehensive type hints and validation using Pydantic
3. **Immutable Operations**: Operators don't modify input data, ensuring pipeline integrity
4. **Metadata Preservation**: Shape and processing information tracked throughout pipeline
5. **Error Handling**: Graceful degradation with comprehensive error reporting

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

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray | dict:
        """Callable interface with lifecycle hooks."""
        self._before_call(x)
        result = self.execute(x, **kwargs)
        self._after_call(result)
        return result
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

<!-- ## API Reference

### Complete Operator Catalog

The following sections provide comprehensive documentation for all available operators, organized by category.

---

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

---

<details>
<summary><strong>EXPAND (rank ↑)</strong>: 新增轴或将单轴拆分为多轴</summary>

这类操作会增加数据的维度，通常用于将一维时序信号转换为二维的时频表示或图像块。

| 工具 | 描述 | 输入维度 | 输出维度 | 计算公式 |
| :--- | :--- | :--- | :--- | :--- |
| **Patching** | 将长信号切分成多个重叠或不重叠的短片段 (Patch)。 | `(B, L, C)` | `(B, N, P, C)` | $x[n] \to x[i, j]$ |
| **STFT** | 短时傅里叶变换，生成频谱图 (Spectrogram)，将时域信号转换为时频域表示。 | `(B, L, C)` | `(B, F, T, C)` | $X(f, t) = \sum_n x[n] w[n-t] e^{-j2\pi fn/N}$ |
| **CWT** | 连续小波变换，生成尺度图 (Scalogram)，提供信号在不同尺度下的时频分析。 | `(B, L, C)` | `(B, S, L, C)` | $C(s, \tau) = \int x(t) \psi^*\left(\frac{t-\tau}{s}\right) dt$ |
| **Mel Spectrogram**| Mel 频谱图，在频率轴上使用 Mel 尺度，更符合人类听觉特性。 | `(B, L, C)` | `(B, M, T, C)` | $M(m, t) = |\text{STFT}(x)|^2 \times \text{Mel}_{\text{filter}}$ |
| **Spectrogram** | 频谱图，是 STFT 结果的幅值平方。 | `(B, L, C)` | `(B, F, T, C)` | $S(f, t) = |\text{STFT}(x(t))|^2$ |
| **VQT** | 可变Q变换，一种在对数频率尺度上具有恒定分辨率的高级时频分析方法。 | `(B, L, C)` | `(B, Q, T, C)` | $V(k, t) = \sum_n x[n] w_k[n-t] e^{-j2\pi f_k n/N}$ |
| **Time-Delay Embedding** | 时间延迟嵌入，用于从时间序列重构相空间。 | `(B, L, C)` | `(B, L', D, C)` | $X(t) = [x(t), x(t-\tau), \ldots, x(t-(d-1)\tau)]$ |
| **EMD** | 经验模态分解，将信号分解为多个本征模态函数（IMF）。 | `(B, L, C)` | `(B, L, I, C)` | $x(t) = \sum_i \text{IMF}_i(t) + r(t)$ |
| **VMD** | 变分模态分解，一种比 EMD 更稳健的信号分解方法。 | `(B, L, C)` | `(B, L, K, C)` | $x(t) = \sum_k u_k(t)$ |
| **Wigner-Ville** | 维格纳-威利分布，一种高分辨率时频分析方法。 | `(B, L, C)` | `(B, L, F, C)` | $W(t, f) = \int x(t+\tau/2) x^*(t-\tau/2) e^{-j2\pi f\tau} d\tau$ |
| **Scalogram** | 尺度图，是 CWT 结果的幅值或幅值平方。 | `(B, L, C)` | `(B, S, L, C)` | $S(s, \tau) = |\text{CWT}(s, \tau)|^2$ |

</details>

---

<details>
<summary><strong>TRANSFORM (rank =)</strong>: 轴数不变，只改变域或数值</summary>

这类操作在不改变数据维度数量的前提下，对信号进行变换或处理。

| 工具 | 描述 | 输入维度 | 输出维度 | 计算公式 |
| :--- | :--- | :--- | :--- | :--- |
| **Normalize** | 对信号进行归一化处理，如 Z-score 标准化或 Min-Max 缩放。 | `(B, L, C)` | `(B, L, C)` | $\frac{x - \mu}{\sigma}$ or $\frac{x - \min}{\max - \min}$ |
| **Detrend** | 移除信号中的趋势项，通常是线性的。 | `(B, L, C)` | `(B, L, C)` | $x'(t) = x(t) - (at + b)$ |
| **FFT** | 快速傅里叶变换，将时域信号转换为频域表示。虽然域改变，但通常保持维度不变。 | `(B, L, C)` | `(B, F, C)` | $X[k] = \sum_n x[n] e^{-j2\pi kn/N}$ |
| **PSD** | 功率谱密度，使用 Welch 方法计算信号的功率谱。 | `(B, L, C)` | `(B, F, C)` | $P(f) = \frac{|X(f)|^2}{N}$ |
| **Integrate** | 计算信号的累积积分（例如，从加速度到速度）。 | `(B, L, C)` | `(B, L, C)` | $y[n] = \sum_{i=0}^n x[i]$ |
| **Differentiate** | 计算信号的差分（例如，从速度到加速度）。 | `(B, L, C)` | `(B, L-1, C)` | $y[n] = x[n] - x[n-1]$ |
| **Power to dB** | 将功率谱或频谱图转换为分贝（dB）单位。 | `(B, F, ...)` | `(B, F, ...)` | $10 \log_{10}(P)$ |
| **PCA** | 主成分分析，用于特征降维。 | `(B, C')` | `(B, n_components)` | $Y = XW$ |
| **Adaptive Filter** | 自适应滤波，用于从信号中去除噪声。 | `Dict['d', 'x']` | `Dict['y', 'e', 'w']` | $y[n] = w^T[n] x[n]$ |
| **Savitzky-Golay Filter** | SG 滤波器，一种强大的平滑去噪方法。 | `(B, L, C)` | `(B, L, C)` | 多项式最小二乘拟合 |
| **Cepstrum** | 倒谱分析，用于检测信号中的谐波成分。 | `(B, L, C)` | `(B, L, C)` | $c[n] = \text{IFFT}(\log|\text{FFT}(x[n])|)$ |
| **Filter** | 使用滤波器（如低通、高通、带通）去除或保留特定频率成分。 | `(B, L, C)` | `(B, L, C)` | $Y(z) = H(z) X(z)$ |
| **Denoise (Wavelet)** | 使用小波阈值去噪，一种先进的去噪方法。 | `(B, L, C)` | `(B, L, C)` | 小波阈值处理 |
| **Hilbert Envelope**| 通过希尔伯特变换计算信号的解析信号，并提取其包络。 | `(B, L, C)` | `(B, L, C)` | $|x(t) + j\mathcal{H}\{x(t)\}|$ |
| **Resample** | 将信号重采样到新的长度。 | `(B, L, C)` | `(B, L_new, C)` | 插值/抽取算法 |

</details>

---

<details>
<summary><strong>AGGREGATE (rank ↓)</strong>: 去掉轴或汇聚到向量/标量</summary>

这类操作会减少数据的维度，通常用于从信号中提取紧凑的特征表示。

| 工具 | 描述 | 输入维度 | 输出维度 | 计算公式 |
| :--- | :--- | :--- | :--- | :--- |
| **时域统计特征** | 计算信号的各种统计量，如均值、方差、峰度等。 | `(B, L, C)` | `(B, C')` | 参见下方详细公式表 |
| **过零率 (ZCR)** | 计算信号穿过零点的频率。 | `(B, L, C)` | `(B, C')` | $\text{ZCR} = \frac{1}{2(L-1)}\sum_{i=1}^{L-1} |\text{sgn}(x_i) - \text{sgn}(x_{i-1})|$ |
| **峰峰值 (P2P)** | 计算信号的最大值和最小值之差。 | `(B, L, C)` | `(B, C')` | $\text{P2P} = \max(x) - \min(x)$ |
| **Hjorth 参数** | 计算信号的活动性、移动性和复杂性。 | `(B, L, C)` | `(B, 3, C)` | Activity, Mobility, Complexity |
| **近似熵 (ApEn)** | 量化信号的规律性和复杂度。 | `(B, L, C)` | `(B, C')` | $-\log(\phi(m+1) / \phi(m))$ |
| **排列熵 (PermEn)** | 基于排序模式量化信号的复杂度。 | `(B, L, C)` | `(B, C')` | $-\sum p_i \log(p_i)$ (排序模式) |
| **频域统计特征** | 计算功率谱密度(PSD)的统计量。 | `(B, F, C)` | `(B, C')` | PSD上的统计量 |
| **谱质心** | 计算频谱的能量中心，是重要的频域特征。 | `(B, F, C)` | `(B, C')` | $\frac{\sum f \times |X(f)|^2}{\sum |X(f)|^2}$ |
| **谱偏度** | 计算频谱的偏度。 | `(B, F, C)` | `(B, C')` | 频谱的三阶矩 |
| **谱峰度** | 计算频谱的峰度。 | `(B, F, C)` | `(B, C')` | 频谱的四阶矩 |
| **谱平坦度** | 衡量频谱的音调特性。 | `(B, F, C)` | `(B, C')` | 几何平均 / 算术平均 |
| **频带功率 (Band Power)** | 计算特定频带内的平均功率。 | `(B, F, C)` | `(B, C')` | $\int_{f_1}^{f_2} P(f) df$ |
| **CNN Pooling** | 卷积神经网络中的池化层（如平均池化、最大池化），用于降低特征图的空间维度。 | `(B, H, W, C)` | `(B, H', W', C)` 或 `(B, C')` | $\max$ or $\text{mean}$ pooling |

#### 特征提取详情

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

</details>
---

<details>
<summary><strong>MULTI-VARIABLE</strong>: 接收多个输入</summary>

这类工具用于比较或组合来自不同处理分支的节点。

| 工具 | 描述 | 输入 | 输出 | 计算公式 |
| :--- | :--- | :--- | :--- | :--- |
| **Subtract** | 信号相减，计算两个信号的差值（残差）。 | 两个 `(B, L, C)` | `(B, L, C)` | $z = x - y$ |
| **Arithmetic** | 在两个信号之间执行基本的数学运算（+,-,*,/）。 | 两个 `(B, L, C)` | `(B, L, C)` | $z = x \oplus y$ where $\oplus \in \{+, -, \times, \div\}$ |
| **Element-wise Product** | 信号逐元素相乘，可用于调制或加窗。 | 两个 `(B, L, C)` | `(B, L, C)` | $z[n] = x[n] \times y[n]$ |
| **Convolution** | 对信号和核执行一维卷积。 | `(B, L, C)` 和 `(K,)` | `(B, L', C)` | $z[n] = \sum_k x[k] h[n-k]$ |
| **Transfer Function** | 估计输入输出信号间的传递函数。 | `(L,)` 和 `(L,)` | 字典 | $H(f) = \frac{Y(f)}{X(f)}$ |
| **Phase Difference** | 计算两个信号在频域的相位差。 | 两个 `(B, F, C)` | `(B, F, C)` | $\Delta\phi = \arg(X) - \arg(Y)$ |
| **Coherence** | 相干函数，分析两个信号在频域的线性相关性。 | 两个 `(B, L, C)` | `(B, F, C)` | $C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f) P_{yy}(f)}$ |
| **DTW Distance** | 动态时间规整，计算两个不等长序列的相似度。 | `(L1, C)` 和 `(L2, C)` | 标量 | 动态规划对齐成本 |
| **Cross-Correlation** | 计算两个信号的互相关，分析其相似性与延迟。 | 两个 `(B, L, C)` | `(B, L_corr, C)` | $R_{xy}[k] = \sum_n x[n] y[n-k]$ |
| **Distance** | 计算两个特征向量之间的距离（如欧氏距离）。 | 两个 `(B, C')` | `(B,)` | $d = \|x - y\|_p$ |
| **Concatenate** | 沿指定轴拼接多个特征向量。 | 多个 `(B, C')` | `(B, C_new)` | $z = [x; y]$ |
| **Compare** | 比较两个或多个特征向量/矩阵，例如计算距离或相似度。 | 多个 `(B, C')` | `(B, C'')` 或字典 | 多种相似度/距离指标 |

</details>

---

<details>
<summary><strong>DECISION</strong>: 输出判断</summary>

这类工具通常是流程的终点，输出最终的结论。

| 工具 | 描述 | 输入 | 输出 | 计算公式 |
| :--- | :--- | :--- | :--- | :--- |
| **Threshold** | 判断输入值是否超过阈值。 | 标量 | 字典 (布尔值) | $y = \begin{cases} 1 & \text{if } x > \theta \\ 0 & \text{otherwise} \end{cases}$ |
| **Find Peaks** | 在信号或频谱中寻找峰值。 | `(L,)` | 字典 | $\arg\max_{\text{local}}(x)$ |
| **Change Point Detection** | 检测信号统计特性的突变点。 | `(L, C)` | 字典 | 统计变化检测 (CUSUM, 等) |
| **Outlier Detection** | 使用孤立森林等算法检测特征集中的异常点。 | `(B, C')` | 字典 | 孤立分数或 LOF |
| **KS-Test** | KS检验，判断两个样本是否来自同一分布。 | `(L1,)` 和 `(L2,)` | 字典 | $D = \max|F_1(x) - F_2(x)|$ |
| **Harmonic Analysis** | 从频谱中识别基频的谐波系列。 | 字典 | 字典 | $f_n = n \times f_0$ |
| **Rule-Based Decision** | 基于一组简单的规则（例如 "rms > 0.5 AND crest < 1.2"）做出决策。 | 字典 | 字典 (布尔值) | 布尔逻辑评估 |
| **Similarity Classifier** | 通过计算与参考特征的相似度来进行分类。 | 字典 | 字典 (字符串) | $\arg\min_c d(x, c)$ |
| **Anomaly Scorer** | 基于与健康状态的距离计算异常分数。 | 字典 | 字典 (浮点数) | $\text{score} = d(x, \mu_{\text{normal}})$ |
| **Classifier** | 基于提取的特征进行分类，输出故障类型。 | `(B, C')` | 字符串标签 | 模型特定 (SVM, RF, 等) |
| **Scoring** | 对比参考和测试信号的特征，给出一个健康评分或异常分数。 | 多个 `(B, C')` | 浮点数值 | 距离/相似度指标 |

</details> -->

## Usage Examples

### Basic Operator Usage

#### Single Operator Application

```python
from src.tools.expand_schemas import STFTOp
import numpy as np

# Create test signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
signal = signal.reshape(1, -1, 1)  # Shape: (batch=1, length=1000, channels=1)

# Create and configure operator
stft_op = STFTOp(
    fs=fs,
    nperseg=256,    # Window length
    noverlap=128,   # Overlap length
    window='hann'   # Window type
)

# Apply operator
spectrogram = stft_op.execute(signal)
print(f"Input shape: {signal.shape}")
print(f"Output shape: {spectrogram.shape}")
print(f"Operator metadata: {stft_op.op_name}, {stft_op.description}")
```

#### Chained Operations

```python
from src.tools.expand_schemas import STFTOp
from src.tools.aggregate_schemas import MeanOp
from src.tools.transform_schemas import FFTOp

# Create processing chain
signal = np.random.randn(1, 1024, 1)

# Step 1: FFT transform
fft_op = FFTOp()
fft_result = fft_op.execute(signal)
print(f"FFT: {signal.shape} → {fft_result.shape}")

# Step 2: Aggregate to features
mean_op = MeanOp(axis=-2)  # Average over frequency axis
features = mean_op.execute(fft_result)
print(f"Mean: {fft_result.shape} → {features.shape}")

# Step 3: STFT for time-frequency analysis
stft_op = STFTOp(fs=1000, nperseg=128)
spectrogram = stft_op.execute(signal)
print(f"STFT: {signal.shape} → {spectrogram.shape}")
```

### Multi-Variable Operations

```python
from src.tools.multi_schemas import ArithmeticOp, CompareOp

# Create two signals for comparison
signal1 = np.random.randn(1, 1024, 1)
signal2 = np.random.randn(1, 1024, 1) * 1.5

# Arithmetic operations
add_op = ArithmeticOp(operation="add")
sum_result = add_op.execute({"signal1": signal1, "signal2": signal2})

subtract_op = ArithmeticOp(operation="subtract")
diff_result = subtract_op.execute({"signal1": signal1, "signal2": signal2})

# Comparison operations
compare_op = CompareOp(metrics=["euclidean", "cosine"])
comparison = compare_op.execute({"ref": signal1, "test": signal2})
print(f"Comparison results: {comparison}")
```

### Feature Extraction Pipeline

```python
from src.tools.aggregate_schemas import (
    MeanOp, StdOp, RMSOp, KurtosisOp, SkewnessOp
)

def extract_time_domain_features(signal: np.ndarray) -> dict:
    """Extract comprehensive time-domain features."""

    features = {}

    # Statistical features
    features['mean'] = MeanOp().execute(signal)
    features['std'] = StdOp().execute(signal)
    features['rms'] = RMSOp().execute(signal)
    features['kurtosis'] = KurtosisOp().execute(signal)
    features['skewness'] = SkewnessOp().execute(signal)

    return features

# Apply to signal
signal = np.random.randn(1, 1024, 1)
features = extract_time_domain_features(signal)

for name, value in features.items():
    print(f"{name}: {value.shape} = {value.flatten()}")
```

### Frequency Domain Analysis

```python
from src.tools.transform_schemas import FFTOp
from src.tools.aggregate_schemas import BandPowerOp
from src.tools.expand_schemas import STFTOp

def frequency_analysis_pipeline(signal: np.ndarray, fs: float) -> dict:
    """Complete frequency domain analysis."""

    results = {}

    # 1. FFT Analysis
    fft_op = FFTOp()
    fft_result = fft_op.execute(signal)
    results['fft_spectrum'] = fft_result

    # 2. Band Power Analysis
    bands = [(0, 50), (50, 150), (150, 300), (300, 500)]  # Frequency bands
    band_power_op = BandPowerOp(fs=fs, bands=bands)
    band_powers = band_power_op.execute(signal)
    results['band_powers'] = band_powers

    # 3. Time-Frequency Analysis
    stft_op = STFTOp(fs=fs, nperseg=256, noverlap=128)
    spectrogram = stft_op.execute(signal)
    results['spectrogram'] = spectrogram

    return results

# Apply pipeline
fs = 1000
signal = np.random.randn(1, 1024, 1)
freq_results = frequency_analysis_pipeline(signal, fs)

for name, result in freq_results.items():
    print(f"{name}: {result.shape}")
```

### Advanced Signal Processing

```python
from src.tools.expand_schemas import EmpiricalModeDecompositionOp, TimeDelayEmbeddingOp
from src.tools.transform_schemas import FilterOp

def advanced_signal_analysis(signal: np.ndarray, fs: float) -> dict:
    """Advanced signal processing techniques."""

    results = {}

    # 1. Empirical Mode Decomposition
    emd_op = EmpiricalModeDecompositionOp()
    imfs = emd_op.execute(signal)
    results['imfs'] = imfs
    print(f"EMD decomposed signal into {imfs.shape[-2]} IMFs")

    # 2. Time-Delay Embedding for phase space reconstruction
    tde_op = TimeDelayEmbeddingOp(dimension=3, delay=4)
    phase_space = tde_op.execute(signal)
    results['phase_space'] = phase_space
    print(f"Phase space reconstruction: {phase_space.shape}")

    # 3. Filtering
    filter_op = FilterOp(
        filter_type="bandpass",
        low_freq=50,
        high_freq=200,
        fs=fs,
        order=4
    )
    filtered_signal = filter_op.execute(signal)
    results['filtered'] = filtered_signal

    return results

# Apply advanced analysis
signal = np.random.randn(1, 2048, 1)  # Longer signal for EMD
advanced_results = advanced_signal_analysis(signal, 1000)
```

## Integration with PHMGA System

### Agent Integration

Operators are seamlessly integrated with the PHMGA agent system:

```python
from src.agents.execute_agent import execute_agent
from src.states.phm_states import PHMState

# Operators are automatically discovered and used by execute_agent
state = PHMState(
    detailed_plan=[
        {"parent": "ch1", "op_name": "fft", "params": {}},
        {"parent": "fft_ch1", "op_name": "mean", "params": {"axis": -2}},
        {"parent": "mean_fft_ch1", "op_name": "compare", "params": {"metrics": ["euclidean"]}}
    ],
    # ... other state fields
)

# Execute agent automatically applies operators
result = execute_agent(state)
```

### DAG Construction

Operators automatically build the computational DAG:

```python
from src.tools.signal_processing_schemas import get_operator

# Get operator class
op_class = get_operator("stft")

# Create operator instance with parameters
op_instance = op_class(fs=1000, nperseg=256, parent="ch1")

# Operator metadata is used for DAG construction
print(f"Operator: {op_instance.op_name}")
print(f"Input spec: {op_instance.input_spec}")
print(f"Output spec: {op_instance.output_spec}")
print(f"Parent node: {op_instance.parent}")
```

### Error Handling

Operators include comprehensive error handling:

```python
try:
    # Operator execution with validation
    result = operator.execute(invalid_input)
except ValueError as e:
    print(f"Input validation error: {e}")
except Exception as e:
    print(f"Execution error: {e}")
    # Error is logged to DAG error log
    state.dag_state.error_log.append(f"Operator {operator.op_name} failed: {e}")
```

## Custom Operator Development

### Creating New Operators

#### Basic Transform Operator

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

#### Advanced Multi-Variable Operator

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

        # Ensure signals have same shape
        if sig1.shape != sig2.shape:
            raise ValueError(f"Signal shapes must match: {sig1.shape} vs {sig2.shape}")

        # Compute cross-correlation for each batch and channel
        batch_size, length, channels = sig1.shape

        if self.mode == "full":
            output_length = 2 * length - 1
        elif self.mode == "same":
            output_length = length
        else:  # valid
            output_length = max(0, length - length + 1)

        result = np.zeros((batch_size, output_length, channels))

        for b in range(batch_size):
            for c in range(channels):
                corr = np.correlate(sig1[b, :, c], sig2[b, :, c], mode=self.mode)

                if self.normalize:
                    # Normalize by signal energies
                    norm_factor = np.sqrt(np.sum(sig1[b, :, c]**2) * np.sum(sig2[b, :, c]**2))
                    corr = corr / (norm_factor + 1e-8)

                result[b, :, c] = corr

        return result

# Usage
cross_corr_op = CrossCorrelationOp(mode="full", normalize=True)
correlation = cross_corr_op.execute({"signal1": sig1, "signal2": sig2})
```

#### Decision Operator

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
    labels: Dict[str, str] = Field(
        default_factory=lambda: {"below": "healthy", "above": "faulty"},
        description="Classification labels"
    )

    def execute(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Execute threshold-based classification."""

        batch_size, n_features = x.shape
        predictions = []
        confidence_scores = []

        for b in range(batch_size):
            features = x[b, :]

            # Simple threshold logic (can be extended)
            if len(self.thresholds) > 0:
                # Use provided thresholds
                above_threshold = any(
                    features[i] > thresh
                    for i, thresh in enumerate(self.thresholds.values())
                    if i < len(features)
                )
            else:
                # Use mean as threshold
                above_threshold = np.mean(features) > 0.5

            prediction = self.labels["above"] if above_threshold else self.labels["below"]
            confidence = float(np.max(features)) if above_threshold else float(1 - np.max(features))

            predictions.append(prediction)
            confidence_scores.append(confidence)

        return {
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "n_samples": batch_size,
            "decision_rule": "threshold_based"
        }

# Usage
classifier_op = ThresholdClassifierOp(
    thresholds={"feature_0": 0.5, "feature_1": 0.3},
    labels={"below": "normal", "above": "anomaly"}
)
classification = classifier_op.execute(feature_matrix)
```

### Testing Custom Operators

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

if __name__ == "__main__":
    test_custom_operator()
```

## Performance Optimization

### Efficient Operator Implementation

```python
@register_op
class OptimizedFFTOp(TransformOp):
    """Performance-optimized FFT operator."""

    op_name: ClassVar[str] = "optimized_fft"
    description: ClassVar[str] = "Memory-efficient FFT with batch processing"
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, F, C)"

    def execute(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Execute optimized FFT."""

        # Use real FFT for real-valued signals
        if np.isrealobj(x):
            result = np.fft.rfft(x, axis=-2)
        else:
            result = np.fft.fft(x, axis=-2)

        # Return magnitude spectrum
        return np.abs(result)
```

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

## Dependencies and Requirements

### Core Dependencies

```python
# Required packages
numpy>=1.21.0          # Numerical computing
scipy>=1.7.0           # Scientific computing
pydantic>=2.0.0        # Data validation
typing-extensions      # Type hints

# Optional dependencies
librosa>=0.9.0         # Audio processing (for some operators)
scikit-learn>=1.0.0    # Machine learning (for some operators)
matplotlib>=3.5.0      # Visualization (for debugging)
```

### Installation

```bash
# Install core dependencies
pip install numpy scipy pydantic typing-extensions

# Install optional dependencies
pip install librosa scikit-learn matplotlib

# Or install all at once
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

#### Operator Not Found
```python
# Problem: KeyError when getting operator
try:
    op_class = get_operator("nonexistent_op")
except KeyError as e:
    print(f"Operator not found: {e}")
    print(f"Available operators: {list(OP_REGISTRY.keys())}")
```

#### Shape Mismatch
```python
# Problem: Input shape doesn't match operator expectations
def validate_input_shape(operator: PHMOperator, input_data: np.ndarray) -> bool:
    """Validate input shape against operator specification."""

    expected_dims = len(operator.input_spec.split(','))
    actual_dims = len(input_data.shape)

    if actual_dims != expected_dims:
        print(f"Shape mismatch: expected {expected_dims}D, got {actual_dims}D")
        return False

    return True
```

#### Parameter Resolution
```python
# Problem: Missing required parameters
def check_required_parameters(op_class: type, params: dict) -> List[str]:
    """Check for missing required parameters."""

    missing = []
    for field_name, field_info in op_class.model_fields.items():
        if field_info.is_required() and field_name not in params:
            missing.append(field_name)

    return missing
```

This comprehensive tools documentation provides everything needed to understand, use, and extend the PHMGA signal processing system.
```

