好的，这是根据您提供的算子（工具）分类和维度约定，为五大类信号处理操作设计的元公式（Meta-Formulas）。

每个元公式都旨在成为该类别下所有具体算子的通用数学抽象，并明确体现了数据维度的变化。

---

### 1. `EXPAND` (升维)

这类算子通过引入新的轴或分解现有轴来增加数据的秩（维度）。

#### Meta-Formula

$$\mathcal{F}_{expand}: \mathbb{R}^{B \times L \times C} \to \mathbb{R}^{B \times D_1 \times D_2 \times \dots \times D_m \times C}$$

$$\mathbf{Y} = \mathcal{F}_{expand}(\mathbf{X}; \theta)$$

**其中 (Where):**
* `$\mathbf{X}$` 是输入信号张量，`$\mathbf{Y}$` 是输出张量。
* `$\mathcal{F}_{expand}$` 代表任何一个升维算子。
* `$\theta$` 代表算子的参数集合，例如窗口长度、步长、小波类型等。
* `$\mathbb{R}^{B \times L \times C}$` 是输入数据的空间，代表批大小为 `B`、信号长度为 `L`、通道数为 `C` 的时域信号。
* `$\mathbb{R}^{B \times D_1 \times \dots \times D_m \times C}$` 是输出数据的空间，其中 `m ≥ 1`。原有的 `L` 轴被 `m` 个新轴 `$D_1, \dots, D_m$` 替代，例如时频轴。

**实例化示例 (STFT):**
* `$\mathcal{F}_{expand}$` 成为短时傅里叶变换算子。
* `$\theta$` 包含窗口类型、窗口长度 `n_fft` 和跳跃长度 `hop_length`。
* 维度变化为：`$L \to (F, T)$`。
* 公式实例化为: `$\mathcal{F}_{stft}: \mathbb{R}^{B \times L \times C} \to \mathbb{R}^{B \times F \times T \times C}$`。

---

### 2. `TRANSFORM` (同维)

这类算子在不改变数据秩（维度数量）的前提下，对数值或域进行变换。

#### Meta-Formula

$$\mathcal{F}_{transform}: \mathbb{R}^{D_{in}} \to \mathbb{R}^{D_{out}} \quad \text{s.t.} \quad \text{rank}(D_{in}) = \text{rank}(D_{out})$$

$$\mathbf{Y} = \mathcal{F}_{transform}(\mathbf{X}; \theta)$$

**其中 (Where):**
* `$\text{rank}(D)$` 表示维度 `D` 的秩（轴的数量）。此约束是同维变换的核心。
* `$D_{in} = (B, \dots, d_i, \dots, C)$` 是输入维度。
* `$D_{out} = (B, \dots, d'_i, \dots, C)$` 是输出维度。`$d_i$` 可能变为 `$d'_i$`（如FFT中 `L` 变 `F`），或保持不变（如滤波）。

**实例化示例 (FFT):**
* `$\mathcal{F}_{transform}$` 成为快速傅里叶变换算子。
* `$\theta$` 可以为空或包含特定FFT算法的参数。
* 维度变化为：`$(B, L, C) \to (B, F, C)$`。秩保持为3不变。
* 公式实例化为: `$\mathcal{F}_{fft}: \mathbb{R}^{B \times L \times C} \to \mathbb{R}^{B \times F \times C}$`。

---

### 3. `AGGREGATE` (降维)

这类算子通过对一个或多个轴进行聚合运算（如求和、平均）来降低数据的秩。

#### Meta-Formula

$$\mathcal{F}_{aggregate}: \mathbb{R}^{B \times L \times C} \to \mathbb{R}^{B \times C'}$$

$$\mathbf{y} = \mathcal{F}_{aggregate}(\mathbf{X}; \theta)$$

**其中 (Where):**
* `$\mathbf{y}$` 是输出的特征向量矩阵。
* `$\mathcal{F}_{aggregate}$` 代表任何一个降维算子，如统计特征提取。
* 算子作用于 `L` 轴，将其聚合为一个或多个特征。
* `$C'$` 是从每个通道提取出的特征数量。如果只提取一个特征（如均值），则 `$C'=1 \times C$`；如果提取多个特征，则 `$C'` 会更大。

**实例化示例 (时域统计特征 - 均值):**
* `$\mathcal{F}_{aggregate}$` 成为均值算子 `$\frac{1}{L}\sum_{l=1}^{L}(\cdot)$`。
* `$\theta$` 为空。
* 维度变化为：`$(B, L, C) \to (B, 1, C)`，通常会压缩为 `$(B, C)$`。
* 公式实例化为: `$\mathcal{F}_{mean}: \mathbb{R}^{B \times L \times C} \to \mathbb{R}^{B \times C}$`，其中输出的第二维代表每个通道的均值。

---

### 4. `MULTI-VARIABLE` (多变量)

这类算子接收两个或多个张量作为输入，以进行比较、组合或计算它们之间的关系。

#### Meta-Formula

$$\mathcal{F}_{multi}: (\mathbb{R}^{D_1} \times \mathbb{R}^{D_2}) \to \mathbb{R}^{D_{out}}$$

$$\mathbf{Z} = \mathcal{F}_{multi}(\mathbf{X}, \mathbf{Y}; \theta)$$

**其中 (Where):**
* `$\mathbf{X}$` 和 `$\mathbf{Y}$` 是两个输入张量，它们来自维度空间 `$D_1$` 和 `$D_2$`。在很多情况下，`$D_1 = D_2$`。
* `$\times$` 表示空间的笛卡尔积，代表输入是成对的。
* `$\mathbf{Z}$` 是输出张量，其维度 `$D_{out}$` 取决于具体算子。

**实例化示例 (Subtract):**
* `$\mathcal{F}_{multi}$` 成为逐元素相减的算子。
* 输入维度必须相同：`$D_1 = D_2 = (B, L, C)$`。
* 输出维度也保持不变：`$D_{out} = (B, L, C)$`。
* 公式实例化为: `$\mathcal{F}_{sub}: (\mathbb{R}^{B \times L \times C} \times \mathbb{R}^{B \times L \times C}) \to \mathbb{R}^{B \times L \times C}$`。

---

### 5. `DECISION` (决策)

这类算子将数值型信号或特征数据映射到非数值的、语义化的输出空间，如标签、分数或判断。

#### Meta-Formula

$$\mathcal{F}_{decision}: \mathbb{R}^{B \times C'} \to \mathbb{S}^B$$

$$\text{Result} = \mathcal{F}_{decision}(\mathbf{y}; \theta, \mathcal{R})$$

**其中 (Where):**
* `$\mathbf{y}$` 是输入的特征矩阵，通常是 `AGGREGATE` 算子的输出。
* `$\mathcal{R}$` 代表决策所需的规则集、阈值或参考模型。
* `$\mathbb{S}$` 代表输出的“语义空间”（Semantic Space），它不是一个标准的数值空间。
* `$\mathbb{S}^B$` 表示为批处理中的每个样本都生成一个语义输出。
* `$\mathbb{S} \in \{\text{String}, \text{Boolean}, \text{Float (as score)}, \text{Dictionary}, \dots\}$`。

**实例化示例 (Threshold):**
* `$\mathcal{F}_{decision}$` 成为阈值判断算子。
* 输入 `$\mathbf{y}$` 可以是一个标量特征，`$(B, 1)$`。
* `$\mathcal{R}$` 是一个具体的阈值，例如 `$\tau=0.5$`。
* 决策规则是 `$\mathbf{y} > \tau$`。
* 输出空间 `$\mathbb{S}$` 是布尔值 `$\{True, False\}$`。
* 公式实例化为: `$\mathcal{F}_{thresh}: \mathbb{R}^{B \times 1} \to \{\text{True}, \text{False}\}^B$`。