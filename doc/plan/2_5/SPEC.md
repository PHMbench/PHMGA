## SPEC（不可变约定 / Contract）—— Unified-X 可解释网络集成到 PHMGA

本 SPEC 是“外环（Agent/Graph）+ 内环（可微网络 TSPN）”闭环的唯一真相来源。任何实现/智能体必须先满足本 SPEC，再做优化扩展。

---

### 1) 文件边界（Immutable Config）

- `case.yaml`：只描述数据/任务/运行路径/流程开关（数据集名称可任意；由 data_factory 解析）。
- `model_config.yaml`：只描述**可解释网络结构 + 训练/解释超参**；外环智能体唯一允许修改的配置文件。
- 兼容输入：`Unified_X_fault_diagnosis/configs/config_basic.yaml`
  - 允许作为“legacy 输入”，但必须被转换/适配到 `model_config.yaml` 的结构字段（严禁把 `data_dir` 等数据路径带入 model_config）。

---

### 2) 数据与张量唯一真相（Single Source of Truth）

- 统一张量形状：`(B, L, C)`（numpy/torch 一致）
  - `B` batch size
  - `L` 信号长度
  - `C` 通道数
- 多通道在 DAG 中表现为多个节点：`ch1..chC`（每个节点保存 `{sample_id: (1, L, 1)}`）。
- 训练前必须融合为 `x_all`：`{sample_id: (1, L, C)}`，融合顺序必须等于 `dag_state.channels`，并将顺序落盘到 `channels.json`。

---

### 3) 标签边界（防泄露红线）

- `labels_ref`（train/val 可见）与 `labels_tst`（默认不可见）必须物理隔离。
- 训练/结构搜索阶段禁止读取 `labels_tst` 参与任何指标或决策；除非显式开关 `allow_test_labels_for_reporting=true` 仅用于报告。

---

### 4) Torch-side 算子 token I/O Contract（必须严格、可测试）

**核心原则（建议固定口径）：**
- TSPN 的所有 Signal Processing（SP）token 输出必须是 **real**（float32/float64）。
- 输出 shape 必须保持 `(B, L, C)`，其中 `L` 不变。
- 若 token 天然会输出 complex 或改变长度（如 `FFT`），必须在 token 内完成“对齐/回投影策略”。

#### 4.1 统一 token 命名（与 Unified 对齐）

- 基础 token：`I`, `WF`, `HT`, `FFT`
- Unified 扩展 token（待迁移）：`Morlet`, `Laplace`, `order1_MA`, `order2_MA`, `order1_DF`, `order2_DF`, `Log`, `Squ`, `sin`, `LNO`, `add`, `mul`, `div`

#### 4.2 token 合同表（最小必须覆盖）

| Token | 输入 -> 输出 | dtype | L 是否保持 | 是否需要 `fs` | 备注/对齐策略 |
|---|---|---|---|---|---|
| `I` | `(B,L,C)->(B,L,C)` | real | 是 | 否 | identity |
| `WF` | `(B,L,C)->(B,L,C)` | real | 是 | 是 | 频域高斯滤波；`fc/fb` 归一化频率→Hz：`f_hz = omega * fs` |
| `HT` | `(B,L,C)->(B,L,C)` | real | 是 | 否 | Hilbert envelope（与 Unified 行为一致：abs(ifft(...)))） |
| `FFT` | `(B,L,C)->(B,L,C)` | real | 是 | 是 | **强制 real**：MVP 采用 `abs(rfft)`；再用**线性插值**把 `[0, fs/2]` 的频域幅值拉伸回 `L` 点（推荐；物理含义更直观） |
| `add/mul/div` | `(B,L,2C)->(B,L,2C)` | real | 是 | 否 | 二元算子：拆分通道对半做运算，再 repeat/concat 回原通道数（与 Unified 2-arity 逻辑一致） |

> 备注：Unified 代码中 `FFT` 直接返回 complex rfft；在 PHMGA 集成必须落到 real contract，否则会击穿后续 nn 模块/解释性闭环。

---

### 5) Feature Extract token Contract（与 Unified `Feature_extract.py` 对齐）

统一约定：所有 feature token 输入 `(B, C, L)`，输出 `(B, C, 1)`；最终拼接为 `(B, C*F)` 进入分类器。

#### 5.1 需要支持的 feature tokens（来自 config_basic.yaml）

基础：`Mean`, `Std`, `Var`, `Entropy`, `Max`, `Min`, `AbsMean`, `Kurtosis`, `RMS`, `CrestFactor`, `Skewness`, `ClearanceFactor`, `ShapeFactor`

> 若 Unified 里存在 delta/高阶特征（例如 `*Delta`），先作为 P1/P2 扩展项，不阻塞 P0 可复现基线。

#### 5.2 数值稳定性要求

- 所有除法项必须加 `eps`（例如 `1e-12`），避免 NaN。
- `Entropy` 必须使用稳定实现（推荐 `softmax/log_softmax` 组合），避免 `log(0)`。

---

### 6) 训练语义红线

- 禁止 `.data` 与 forward 内 in-place 覆盖参数（会破坏 autograd）。
- “删算子/删连接”的实现必须是**软删除**：把对应 gate 权重设为接近 0（例如 `1e-6`），并通过 `disabled_ops[op_uid]=1e-6` 落盘，保证可复现。

---

### 7) 稳定标识（op_uid）与可复现映射

- 每个算子必须有稳定 `op_uid`：`L{layer_idx}:{token}:{occurrence_idx}`（与 PHMGA 现有实现一致）。
- artifacts 必须包含：
  - `op_uid -> module_key` 映射快照
  - `op_uid -> gate_value`（包括 disabled）

---

### 8) 最小 artifacts schema（论文复现必需）

每次 run 目录：`save/<case_name>/<timestamp>/`
- `case.yaml`、`model_config.yaml`（原样拷贝）
- `metrics.json`、`predictions.csv`、`confusion_matrix.csv`
- `channels.json`、`label_to_index.json`、`sample_ids.json`
- `checkpoint_best.pt`、`checkpoint_last.pt`
- `explain/`：
  - `operator_importance.json`
  - `wavefilters_params.json`（若启用）
  - `feature_stats.json`（新增：对齐 Unified 的 feature set）
  - `crud_history.json`（外环 patch 演化链）
