## Paper 目标文档：Agent 驱动的可解释网络 TSPN（可跨数据集）

### 0) 核心一句话
用一个外层智能体（Agent）在“外界”自动优化 TSPN 的结构与关键超参数；用一个内环训练器在“内界”完成训练与评估；两者通过可复现的反馈闭环迭代更新，最终在 **3 个任务/数据集**上输出故障诊断结果、消融实验与结构化决策证据。

---

### 0.1 范围与文件边界（减少耦合）
- `case.yaml`：只放数据与任务（`data/task`）、流程开关（如 `train_backend`）、保存路径等“运行环境信息”
- `model_config.yaml`：只放 TSPN 结构/训练/解释（外环智能体唯一允许修改的文件）
- 论文叙事只依赖这两份配置与产物，不依赖硬编码路径或数据集名称

### 1) 本质机制：双层闭环（Outer Agent / Inner Training）

#### 1.1 内环：训练与评估（固定口径）
输入：
- 数据（任意数据集）：通过 data_factory 读取并构造训练/验证/测试（或 episode）划分
- 当前 `model_config.yaml`（结构 + 超参 + 解释性开关）

输出（反馈给外环）：
- 指标：`val_macro_f1/val_acc`（主目标）+ `test_macro_f1/test_acc` + confusion matrix
- 解释性统计：operator importance、WF 参数分布（如启用）、以及稳定性摘要（不同 seed/扰动）
- 决策输出：每个 `sample_id` 的 `pred/true/proba/confidence` + 证据摘要（结构快照 + top-k 算子）
约束（必须写死，避免“看起来能跑但不可复现”）：
- 训练/搜索阶段只能读取 `labels_ref`；`labels_tst` 仅用于最终评估（如果存在）
- 多通道融合顺序固定为 `dag_state.channels`（并写入 `channels.json`）

#### 1.2 外环：智能体优化（结构/超参/软删除）
输入：
- 内环反馈（指标 + 解释性 + 失败原因）
- 约束（shape/整除/白名单算子/编辑预算）

输出：
- 更新后的 `model_config.yaml`（仅允许修改白名单字段）
- `diff_summary`（结构化变更摘要：增/删/改哪些算子与参数）

外环能力（CRUD）：
- Read：导出当前网络结构快照（layer→op→重复编号、features、out_channels/scale）
- Create：新增算子/层（受编辑预算限制）
- Update：调整算子超参与训练超参
- Delete（软删除）：不移除模块；将对应 gate/权重段衰减到 `1e-6`（可选冻结），并在报告中标记 disabled

---

### 2) 可跨数据集：统一数据入口（data_factory）
结论：**数据集名称可以任意**，只要 data_factory 的 reader registry 中存在对应实现即可。

论文口径要求：
- data_factory 负责：dataset reader 选择、task wrapper、DataLoader（default eager）或 ID-based lazy（episodic/少样本更推荐）
- 训练代码只依赖统一 batch 结构（例如 `x/y/sample_id/domain_id`），不写死某个 metadata/h5 的路径格式

多通道信号的图表示（与 PHMGA 一致）：
- DAG 内：多通道是多个节点 `ch1..chC`（每个节点 `{sample_id: (1,L,1)}`）
- 训练前：通过跨节点融合得到 `x_all`（`{sample_id: (1,L,C)}`），融合顺序必须固定为 `dag_state.channels`

---

### 3) 论文要交付什么（结果 + 消融 + 决策）

#### 3.1 每个数据集必须给出
- 诊断结果：Accuracy、Macro-F1、Confusion Matrix（必做）
- 决策结果：`pred/true/proba/confidence`（按 `sample_id` 对齐）
- 证据链（可解释）：结构快照 + operator importance top-k（每层）+（可选）WF 参数统计

#### 3.2 消融实验（最小集合）
1) 无 Agent（固定结构） vs 有 Agent（搜索/编辑）
2) 无 CRUD vs 有 CRUD（软删除启用）
3) 多通道融合消融：不融合（单通道） vs 融合（`x_all`）
4) 关键算子消融：去掉/软删除 `WF`、`HT`、`FFT`（至少两项）
5) Few-shot DG（Task C）专属：`k_shot`、`n_way`、episode 数、sampler 策略
（可选 baseline）若论文需要对比传统方法：加入 “特征→浅层分类器（shallow ML）” 作为额外对照即可，但不作为本文目标的核心依赖。

---

### 4) 三个任务/数据集（论文承诺）

#### Task A：常规 5 类诊断（PHM-Vibench）
- 配置参考：`config/case_exp2.yaml`
- 目标：5 类故障诊断 + 决策证据

#### Task B：变转速 3 类诊断（Ottawa）
- 配置参考：`config/case_exp_ottawa.yaml`
- 目标：变工况下稳健诊断 + 决策证据（频率解释需结合 `fs`）

#### Task C：少样本 Domain Generalization（Few-shot DG）
- 目标：跨域泛化 + 可解释性稳定性
- 任务形态：episodic（K-shot support + query）
- 数据集名称可任意，但必须存在 reader；推荐 `data.factory_name: "id"` 以支持 episodic sampler

---

### 5) 复现要求（必须落盘的产物）
每次 run 保存到 `save/<case_name>/<timestamp>/`：
- `case.yaml`、`model_config.yaml`（原样拷贝）
- `metrics.json`、`confusion_matrix.csv`、`predictions.csv`
- `label_to_index.json`、`channels.json`、`sample_ids.json`
- `checkpoint_best.pt`、`checkpoint_last.pt`
- `explain/`：`operator_importance.*`、`wavefilters_params.*`（如启用）、`crud_history.json`

---

### 6) 最小里程碑（从能跑到能写论文）
1) 打通 Task A/B：agent→train→report 全流程，产物落盘完整
2) 打通 Task C：few-shot DG 训练/评估与消融（k-shot/episode/sampler）
3) 加入 CRUD：软删除到 `1e-6` 并形成对比表
4) 固化报告模板：每个 case 自动生成“结果 + 消融 + 决策证据链”Markdown
