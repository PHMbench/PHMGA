# 信号处理工具说明

本文档介绍了用于信号处理的各种工具，根据其对输入数据维度（秩）的影响，分为以下类别：

- **`EXPAND` (升维)**: 新增轴或将单轴拆分为多轴，如从时域到时频域。
- **`TRANSFORM` (同维)**: 不改变数据维度数量，只改变域或数值，如滤波、归一化。
- **`AGGREGATE` (降维)**: 减少数据维度，通常用于从信号中提取紧凑的特征，如计算统计量。
- **`MULTI-VARIABLE` (多变量)**: 接收多个输入节点进行比较或组合。
- **`DECISION` (决策)**: 不输出信号数据，而是输出判断结果或文本，如分类或评分。

---

### 输入信号维度约定
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
| **Mel Spectrogram**| Mel 频谱图，在频率轴上使用 Mel 尺度，更符合人类听觉特性。 | `(B, L, C)` | `(B, M, T, C)` |
| **Spectrogram** | 频谱图，是 STFT 结果的幅值平方。 | `(B, L, C)` | `(B, F, T, C)` |
| **VQT** | 可变Q变换，一种在对数频率尺度上具有恒定分辨率的高级时频分析方法。 | `(B, L, C)` | `(B, Q, T, C)` |
| **Time-Delay Embedding** | 时间延迟嵌入，用于从时间序列重构相空间。 | `(B, L, C)` | `(B, L', D, C)` |
| **EMD** | 经验模态分解，将信号分解为多个本征模态函数（IMF）。 | `(B, L, C)` | `(B, L, I, C)` |
| **VMD** | 变分模态分解，一种比 EMD 更稳健的信号分解方法。 | `(B, L, C)` | `(B, L, K, C)` |
| **Wigner-Ville** | 维格纳-威利分布，一种高分辨率时频分析方法。 | `(B, L, C)` | `(B, L, F, C)` |
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
| **FFT** | 快速傅里叶变换，将时域信号转换为频域表示。虽然域改变，但通常保持维度不变。 | `(B, L, C)` | `(B, F, C)` |
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
| **Hilbert Envelope**| 通过希尔伯特变换计算信号的解析信号，并提取其包络。 | `(B, L, C)` | `(B, L, C)` |
| **Resample** | 将信号重采样到新的长度。 | `(B, L, C)` | `(B, L_new, C)` |

</details>

---

<details>
<summary><strong>AGGREGATE (rank ↓)</strong>: 去掉轴或汇聚到向量/标量</summary>

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
| **峰峰值 (Peak-to-Peak)** | $$\text{p2p}(x) = \max(x) - \min(x)$$ |
| **过零率 (Zero-Crossing Rate)** | $$\text{zcr}(x) = \frac{1}{2(L-1)}\sum_{i=1}^{L-1} |\text{sgn}(x_i) - \text{sgn}(x_{i-1})|$$ |

</details>

---

<details>
<summary><strong>MULTI-VARIABLE</strong>: 接收多个输入</summary>

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

</details>

---

<details>
<summary><strong>DECISION</strong>: 输出判断</summary>

这类工具通常是流程的终点，输出最终的结论。

| 工具 | 描述 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| **Threshold** | 判断输入值是否超过阈值。 | 标量 | 字典 (布尔值) |
| **Find Peaks** | 在信号或频谱中寻找峰值。 | `(L,)` | 字典 |
| **Change Point Detection** | 检测信号统计特性的突变点。 | `(L, C)` | 字典 |
| **Outlier Detection** | 使用孤立森林等算法检测特征集中的异常点。 | `(B, C')` | 字典 |
| **KS-Test** | KS检验，判断两个样本是否来自同一分布。 | `(L1,)` 和 `(L2,)` | 字典 |
| **Harmonic Analysis** | 从频谱中识别基频的谐波系列。 | 字典 | 字典 |
| **Rule-Based Decision** | 基于一组简单的规则（例如 "rms > 0.5 AND crest < 1.2"）做出决策。 | 字典 | 字典 (布尔值) |
| **Similarity Classifier** | 通过计算与参考特征的相似度来进行分类。 | 字典 | 字典 (字符串) |
| **Anomaly Scorer** | 基于与健康状态的距离计算异常分数。 | 字典 | 字典 (浮点数) |
| **Classifier** | 基于提取的特征进行分类，输出故障类型。 | `(B, C')` | 字符串标签 |
| **Scoring** | 对比参考和测试信号的特征，给出一个健康评分或异常分数。 | 多个 `(B, C')` | 浮点数值 |

</details>