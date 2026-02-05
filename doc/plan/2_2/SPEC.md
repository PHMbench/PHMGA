## SPEC（不可变约定 / Contract）

本文件是该项目最重要的“不可变约定”。任何 agent/代码实现都必须先满足本 SPEC，再做优化与扩展。

---

### 1) 文件边界（降低耦合、确保可复现）

- `case.yaml`：只包含数据与任务（`data/task`）、运行环境（路径/保存目录/随机种子/流程开关）。
- `model_config.yaml`：只包含 TSPN 的结构/训练/解释配置；**外环 Agent 唯一允许修改的文件**。
- 任何实验复现只依赖：`case.yaml + model_config.yaml + artifacts/`，不依赖硬编码路径或数据集名称。

---

### 2) 数据与张量唯一真相（Single Source of Truth）

#### 2.1 样本与标识
- `sample_id`：字符串或整数，跨全流程稳定，用于对齐预测/标签/产物。
- `channels`：通道名称列表，顺序稳定（来自 `dag_state.channels`），用于跨通道融合与解释。

#### 2.2 张量形状约定（numpy/torch 一致）
- 信号张量统一约定为 `(B, L, C)`：
  - `B`：batch size
  - `L`：时间长度（input length）
  - `C`：通道数
- 多通道在 DAG 中是多个节点 `ch1..chC`，每个节点持有 `{sample_id: (1, L, 1)}`。
- 训练前必须构造融合视图 `x_all`：`{sample_id: (1, L, C)}`。
  - **融合顺序红线**：融合 parent 顺序必须等于 `dag_state.channels`（例如 `"ch1,ch2,...,chC"`），并将该顺序写入 `channels.json`。

---

### 3) 标签边界（防泄露红线）

- 必须物理隔离：`labels_ref`（训练/验证可见）与 `labels_tst`（默认不可见）。
- **训练/搜索阶段**：
  - 任何指标计算只能使用 `labels_ref`。
  - 若代码尝试读取/使用 `labels_tst` 参与训练或验证，必须报错。
- **评估/报告阶段**：
  - 只有在显式开关 `allow_test_labels_for_reporting=true` 时，才允许使用 `labels_tst` 计算 test 指标；
  - 否则 test 阶段只输出预测，不给分数。

---

### 4) Torch-side 算子 token I/O Contract（必须表格化）

> 原则：**TSPN 的每个 Signal Processing（SP）token 输出必须为实数，并且保持 `(B, L, C)` 的 `L` 不变**。  
> 若某 token 天然会改变长度或 dtype（如 FFT），必须在 token 内部实现“对齐/回投影策略”，并写入本表。

| Token | 期望输出 dtype | 期望输出 shape | 是否需要 fs 才可解释到 Hz | 备注（对齐策略/限制） |
|---|---|---|---|---|
| `I` | float32 | `(B,L,C)` | 否 | Identity |
| `WF` | float32 | `(B,L,C)` | 是 | `omega`→Hz：`f_hz = omega * fs`；滤波器数量/维度必须与该层通道设计一致（不得隐式用 `scale` 猜） |
| `HT` | float32 | `(B,L,C)` | 否 | Hilbert envelope（若有频率解释也需注明） |
| `FFT` | float32 | `(B,L,C)` | 是 | MVP 强制 `abs(rfft)` 并对齐回 `L`（pad/interp 任选其一，需固定口径） |

---

### 5) PyTorch 训练语义红线（禁止 `.data` 与 in-place 覆盖参数）

- **红线**：训练图中的参数不得通过 `.data` 或 forward 内 in-place 覆盖来改变数值。
- 门控（gating）必须显式建模：例如 `gate_logits`（Parameter），前向只计算 `gate = softmax(gate_logits)`，不写回权重本体。

---

### 6) “软删除（disabled ops）”的稳定标识（op_uid）

- 禁用逻辑不能依赖 `ModuleDict` 的临时 key（如 `I/I_1`），因为它会随插入/重排而漂移。
- 每个算子必须有稳定 `op_uid`：
  - `op_uid = "L{layer_idx}:{token}:{occurrence_idx}"`
  - 示例：`L2:WF:0`、`L2:WF:1`
- 产物必须保存映射快照：
  - `op_uid -> module_key`（构建后的真实 key）
  - `op_uid -> gate_value`（例如 `1e-6`）
- `model_config.yaml` 里的禁用字段应以 `op_uid` 为键（而不是 `module_key`）。

---

### 7) 产物（Artifacts）最小 schema（论文复现必需）

每次 run 目录：`save/<case_name>/<timestamp>/`
- `case.yaml`、`model_config.yaml`（原样拷贝）
- `metrics.json`、`confusion_matrix.csv`、`predictions.csv`
- `label_to_index.json`、`channels.json`、`sample_ids.json`
- `checkpoint_best.pt`、`checkpoint_last.pt`
- `explain/`：
  - `operator_importance.*`
  - `wavefilters_params.*`（若启用）
  - `crud_history.json`（before/after config + diff_summary + op_uid 映射）
