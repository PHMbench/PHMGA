# 信号处理工具说明

本文档介绍了用于信号处理的各种工具，根据其对输入数据维度（秩）的影响，分为三类：`EXPAND`（升维）、`TRANSFORM`（同维）和 `AGGREGATE`（降维）。

输入信号维度约定：
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

| 工具 | 描述 | 输入维度 | 输出维度 |
| :--- | :--- | :--- | :--- |
| **Patching** | 将长信号切分成多个重叠或不重叠的短片段 (Patch)。 | `(B, L, C)` | `(B, N, P, C)` |
| **STFT** | 短时傅里叶变换，生成频谱图 (Spectrogram)，将时域信号转换为时频域表示。 | `(B, L, C)` | `(B, F, T, C)` |
| **CWT** | 连续小波变换，生成尺度图 (Scalogram)，提供信号在不同尺度下的时频分析。 | `(B, L, C)` | `(B, S, L, C)` |
| **Spectrogram** | 频谱图，是 STFT 结果的幅值平方。 | `(B, L, C)` | `(B, F, T, C)` |
| **Mel Spectrogram**| Mel 频谱图，在频率轴上使用 Mel 尺度，更符合人类听觉特性。 | `(B, L, C)` | `(B, M, T, C)` |
| **Scalogram** | 尺度图，是 CWT 结果的幅值或幅值平方。 | `(B, L, C)` | `(B, S, L, C)` |

</details>

---

<details>
<summary><strong>TRANSFORM (rank =)</strong>: 轴数不变，只改变域或数值</summary>

这类操作在不改变数据维度数量的前提下，对信号进行变换或处理。

| 工具 | 描述 | 输入维度 | 输出维度 |
| :--- | :--- | :--- | :--- |
| **Normalize** | 对信号进行归一化处理，如 Z-score 标准化或 Min-Max 缩放。 | `(B, L, C)` | `(B, L, C)` |
| **Detrend** | 移除信号中的趋势项，通常是线性的。 | `(B, L, C)` | `(B, L, C)` |
| **FFT** | 快速傅里叶变换，将时域信号转换为频域表示。虽然域改变，但通常保持维度不变。 | `(B, L, C)` | `(B, L', C)` (L'≈L/2) |
| **Filtering** | 使用滤波器（如低通、高通、带通）去除或保留特定频率成分。 | `(B, L, C)` | `(B, L, C)` |
| **Hilbert Envelope**| 通过希尔伯特变换计算信号的解析信号，并提取其包络。 | `(B, L, C)` | `(B, L, C)` |

</details>

---

<details>
<summary><strong>AGGREGATE (rank ↓)</strong>: 去掉轴或汇聚到向量/标量</summary>

这类操作会减少数据的维度，通常用于从信号中提取紧凑的特征表示。

| 工具 | 描述 | 输入维度 | 输出维度 |
| :--- | :--- | :--- | :--- |
| **时域统计特征** | 计算信号的各种统计量，如均值、方差、峰度等。 | `(B, L, C)` | `(B, C')` |
| **频域统计特征** | 计算功率谱密度(PSD)的统计量。 | `(B, L, C)` | `(B, C')` |
| **CNN Pooling** | 卷积神经网络中的池化层（如平均池化、最大池化），用于降低特征图的空间维度。 | `(B, H, W, C)` | `(B, H', W', C)` 或 `(B, C')` |

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

</details>