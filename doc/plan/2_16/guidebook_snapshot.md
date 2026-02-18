# Guidebook Snapshot (synced from `doc/plan/2_10/guidebook.md`)

同步日期：2026-02-17  
来源文件：`doc/plan/2_10/guidebook.md`

---

# PHMGA 实验执行指导书（v2.10 融合执行版）

目标：同一份文档同时支持论文实验矩阵（NMI 风格）与工程可执行落地（PHMGA 当前代码契约）。

## A. NMI 风格实验矩阵（论文层）

### A1. 科学问题
- `H1`：反思闭环是否有效（`A0 > A1`）。
- `H2`：神经符号先验初始化是否有效（`A0 > A2`）。
- `H3`：框架是否对不同 LLM backend 具有普适性。

### A2. 三维实验矩阵与字段映射

#### 维度 A：LLM Backends
- `M1`: `gemini-2.5-flash`
- `M2`: `gemini-3-flash`
- `M3`: `GLM-4.5-Flash`
- `M4`: `GLM-4.7-Flash`

字段映射：
- `LLM_PROVIDER`
- `QUERY_GENERATOR_MODEL`
- 可选统一：`PHM_MODEL`, `REFLECTION_MODEL`, `ANSWER_MODEL`

#### 维度 B：Datasets / Domain Shift
- `D1 Ottawa`：变转速轴承（跨工况）。
- `D2 Gearbox`：`RM_101_THU_GEARBOX`（MCC5-THU 系）。

字段映射：
- `data.source_mode`（`vibench` 或 `fixed_ids`）
- `data.dataset_name`
- `data.metadata_file`
- `data.data_dir`
- `data.vibench_code_root`

#### 维度 C：Ablation
- `A0 (Full)`：Plan -> Execute -> Reflect + init_from_dag + TSPN 训练。
- `A1 (No-Reflect)`：关闭反思回路或 `max_iterations=0`。
- `A2 (No-Prior)`：禁用 `init_weights_from_metadata`（结构保留，参数随机）。

### A3. 结果表模板

#### 主结果（跨模型）
| Dataset | Metric | Gemini-2.5 | Gemini-3 | GLM-4.5 | GLM-4.7 | Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Ottawa | Target Acc / Macro-F1 |  |  |  |  |  |
| RM_101_THU_GEARBOX | Target Acc / Macro-F1 |  |  |  |  |  |

#### 消融（机制有效性）
| Dataset | Model | A0 (Full) | A1 (No-Reflect) | A2 (No-Prior) |
| --- | --- | --- | --- | --- |
| Ottawa |  |  |  |  |
| RM_101_THU_GEARBOX |  |  |  |  |

## B. 工程执行手册（落地层）

### B1. 运行前事实与默认约定
- 默认环境：`conda run -n agent`（等价 `conda activate agent`）。
- 数据目录：`/home/user/data/PHMbenchdata/PHM-Vibench/raw/RM_101_THU_GEARBOX`。
- 当前样本统计：总 `192`，其中 `* copy.csv` 为 `83`，有效非 copy 为 `109`。
- 元数据默认：`Name=RM_101_THU_GEARBOX`, `Dataset_id=101`, `Sample_rate=12800`, `Sample_lenth=768000`, `Channel=8`。

### B2. 元数据契约（`gear_metadata.xlsx`）
- 必填列：`Id, Dataset_id, Name, File, Description, Label, Fault level, Domain_id, Domain_description, Sample_rate, Sample_lenth, Channel, Fault_Diagnosis`。
- `File` 必须相对 `data_dir/raw/RM_101_THU_GEARBOX/`。
- 默认排除 `* copy.csv`；如保留，需在报告明确声明重复样本策略。

### B3. 配置模板（主推 `vibench`）

```yaml
name: "exp_gearbox_rm101"
save_dir: "/home/user/LQ/B_Signal/PHMGA/save"
state_save_path: "/home/user/LQ/B_Signal/PHMGA/save/exp_gearbox_rm101/built_state.pkl"
report_path: "/home/user/LQ/B_Signal/PHMGA/save/exp_gearbox_rm101/final_report.md"

run_executor: true
train_backend: tspn
allow_test_labels_for_reporting: false

model:
  profile: tspn_basic
  config_path: "config/model_tspn_basic.yaml"
  autofit_dims: true
  autofit_num_classes: true

data:
  backend: vibench
  source_mode: vibench
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "gear_metadata.xlsx"
  dataset_name: "RM_101_THU_GEARBOX"
  vibench_code_root: "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2"
```

兼容 `fixed_ids` 模式：仅用于可复现实验对照，新数据接入优先 `vibench`。

### B4. 质量门禁（当前实现）
- Builder 迭代上限：`max_builder_iterations` 生效，避免无限回环。
- Provider/Model/Key 预检：`preflight` 检查 provider 配置一致性与 key/base。
- 状态完整性：`save_state` 生成 `*.sha256`；`load_state` 默认校验，可用 `PHM_ALLOW_UNVERIFIED_STATE=1` 本地调试放开。
- 可选依赖：`graphviz`, `nolds` 缺失时给 warning，不阻塞主链路。
- `train_backend` 白名单：仅允许 `shallow|tspn|both`，非法值 fail-fast。

### B5. 固定命令链

```bash
conda run -n agent python main.py preflight --config <case_yaml>
conda run -n agent python main.py case1 --config <case_yaml>
scripts/run_case.sh --config <case_yaml>
```

### B5.1 RM101 验收运行示例（已验证）

```bash
conda run -n agent python main.py preflight --config config/case_exp_gearbox_rm101.yaml
FAKE_LLM=true PHM_REPORT_MODE=template conda run -n agent python main.py case1 --config config/case_exp_gearbox_rm101.yaml
```

产物目录示例：
- `save/exp_gearbox_rm101/20260217-112000`（训练产物）
- `save/exp_gearbox_rm101/run-1771298258/logs/events.jsonl`（执行日志）
- `save/exp_gearbox_rm101/final_report.md`（最终报告）

在线联通（可选）：

```bash
conda run -n agent python tests/llm_smoke_test_glm.py
```

### B6. 产物映射（工程 -> 论文）
- `metrics.json`：主结果指标（Acc/F1）。
- `dataset_manifest.json`：数据来源与切分证据。
- `config_resolve.json`：运行时配置解析与自动对齐证据。
- `model_config.resolved.yaml`：最终生效模型配置。
- `preflight_report.json`：运行时预检结果。
- `final_report.md`：诊断叙事与关键证据摘要。

### B7. 常见错误
- `metadata_file not found`：检查 `data_dir` 与 `metadata_file` 拼接路径。
- `dataset_name not found`：检查 metadata 的 `Name` 列。
- `n_test=0`：仅可报告 train/val，不可宣称泛化结论。

## C. 无痛切换 SOP

切换数据集只改：
- `data.dataset_name`
- `data.metadata_file`（必要时）
- `data.data_dir`（必要时）

切换模型结构只改：
- `model.profile`
- 可选 `model.config_path`

标准入口保持不变：

```bash
conda run -n agent python main.py case1 --config config/case_exp_gearbox_rm101.yaml
```

## D. 论文实验一键代码（可直接执行）

### D1. 准备矩阵配置

复制模板并按你的实际实验改参数：

```bash
cp config/paper_matrix.example.yaml config/paper_matrix.yaml
```

模板文件：
- `config/paper_matrix.example.yaml`
- 可改维度：`matrix.llm` / `matrix.datasets` / `matrix.ablations`

### D2. 运行完整实验矩阵

```bash
conda run -n agent python scripts/paper/run_experiment_matrix.py \
  --matrix-config config/paper_matrix.yaml \
  --output-root save/paper_matrix
```

说明：
- 脚本会自动遍历 `LLM × Dataset × Ablation`。
- 每个组合会自动生成独立 case 配置到 `save/paper_matrix/_resolved_cases/`。
- 每个组合会输出日志到 `save/paper_matrix/_logs/<combo>/`。
- 全部组合写入清单：`save/paper_matrix/manifest.jsonl`。

### D3. 汇总结果为论文表格原始数据

```bash
conda run -n agent python scripts/paper/collect_matrix_results.py \
  --manifest save/paper_matrix/manifest.jsonl \
  --output-dir save/paper_matrix
```

产物：
- `save/paper_matrix/paper_main_results.csv`
- `save/paper_matrix/paper_main_results.md`

建议将该 CSV 再导入你的绘图脚本，生成论文图表（Accuracy/F1 曲线与消融柱状图）。

### D4. 消融与代码映射（当前实现）

- `A0_full`：`PHM_ABLATION_MODE=full`（默认完整闭环）
- `A1_no_reflect`：`PHM_ABLATION_MODE=no_reflect`（内部强制 `builder.max_iterations<=1`）
- `A2_no_prior`：`PHM_ABLATION_MODE=no_prior`（禁用 `init_weights_from_metadata`）

代码位置：
- ablation 解析：`src/cases/case1.py`
- no_prior 生效点：`src/agents/deep_model_train_agent.py`

## E. 你还需要补齐的内容（我无法自动推断）

1. **Ottawa 的最终论文入口配置**
   - 现在模板默认是 `config/tspn_case_exp_ottawa.yaml`（fixed_ids）。
   - 若你要做“严格 dataset_name/vibench 统一”，请提供 Ottawa 在 metadata 中的 `Name` 精确值与对应 case yaml。

2. **在线 LLM 实验网络可达性**
   - 当前若 DNS/网络不可达，只能跑离线（`fake_llm: true`）。
   - 你需要确认实验机可访问对应 provider endpoint（GLM/Gemini）。

3. **论文 Baseline/SOTA 对照值**
   - 本仓库不会自动给出 ResNet/CNN 对照结果。
   - 你需要提供 baseline 实验结果或让我再补一套 baseline 训练脚本。

4. **最终图表样式与统计检验**
   - 若需要 NMI 投稿级图表（含置信区间/显著性检验），请确认你想用的统计方法（如 5 seeds + t-test / bootstrap）。
