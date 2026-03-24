# thesis_2026 分支本轮更新说明

## 1. 概览

本文件用于整理 `thesis_2026` 分支相对于 `NVTA_2025_Version` 基线的本轮完整更新，作为后续分批提交、代码审阅和论文材料整理的统一入口。

本轮更新的总体方向可以概括为：

1. 从 runtime 旁路回归到 graph-first 主链。
2. 把 builder 收紧为可验证的非平凡 DAG 生成器。
3. 补齐 full-DAG 评估、RM101 变工况工具与 simulated planner 梯度实验。
4. 产出 RM101 论文结果包、诊断报告与独立 TeX case。

当前推荐引用结果为：

- `artifacts/paper/rm101_all_domains_mixed_simulated_v1`

当前推荐诊断报告为：

- `RM101_DIAGNOSTIC_REPORT.md`
- `RM101诊断报告_中文.md`

当前推荐论文片段包为：

- `paper_phmga/thesis_rm101_case/`

## 2. 分支基线

当前工作分支：

- `thesis_2026`

相对旧主线基点：

- `d6e5901` (`origin/NVTA_2025_Version`)

已经进入本分支历史、且早于本轮未提交改动的提交有：

1. `14b5db4` `chore(gitignore): exclude artifacts/ directory`
2. `4dba0bd` `feat(config): add hydra-based config structure and paper submodule`
3. `2f43fdd` `feat(runtime): add modular runtime infrastructure`
4. `0651341` `refactor(core): restructure main entrypoint and agent modules`
5. `f899642` `test: add test suite for thesis_2026 runtime`
6. `33d5923` `chore(config): update configuration.py for new runtime`
7. `94af4ab` `chore(utils): update utils.py for thesis_2026 runtime`

本轮文档主要描述的是：**这些历史提交之后，当前工作区中尚未整理提交的完整更新**。

## 3. 本轮核心更新

### 3.1 graph-first 主链恢复与 runtime 清理

本轮已经将运行主链重新收敛为：

`main.py -> src/phm_outer_graph.py -> {builder, executor, with_report}`

核心变化包括：

- 去除以 `src/runtime/pipeline.py` 为核心的手写主 loop。
- 合并 `with_report` 图定义回 `src/phm_outer_graph.py`。
- 删除 `src/phm_outer_graph_with_report.py`。
- case 配置回归到 `config/case*.yaml` 直读，不再以 layered runtime preset 为真值。

对应文件主要涉及：

- `main.py`
- `src/phm_outer_graph.py`
- `src/config/*`
- `config/case*.yaml`
- 删除 `src/runtime/*`

### 3.2 builder 质量约束与 planner/reflect 收紧

为解决弱 DAG、root-only、mean-only、单弱算子等问题，本轮对 builder 质量门控进行了系统增强：

- 新增 `src/builder_quality.py`
- 对 planner 增加 phase-aware shortlist 与 deterministic fallback
- 把 `min_width`、强路径、transform/stat family 等约束变成程序逻辑
- 将 reflect 在 timeout/429 等情况下的弱图冻结行为改成 `need_patch / halt` 受控策略

核心目标是让 builder 的“成功”不再等于“文件生成成功”，而必须等于“生成了可评估、物理上合理、结构上非平凡的 DAG”。

### 3.3 LLM backend 与报告链稳定化

本轮同时整理了后端与报告链：

- 保留 `OpenRouter + GLM-4.5 Air` 路径
- 新增 `BigModel` provider 支持
- 继续保留 `repair_prompt` 双阶段 JSON 修复路径
- 对 report prompt 和 report payload 做压缩与稳定化
- 降低 provider 抖动时对报告生成与 DAG 规划的连锁影响

对应文件包括：

- `src/llm/backends.py`
- `src/llm/__init__.py`
- `src/configuration.py`
- `src/agents/report_agent.py`
- `src/prompts/report_prompt.py`

### 3.4 full-DAG 评估、脚本入口与 manual workflow

为了避免每次都重复从 LLM 规划到最终报告，本轮补齐了完整的评估和可复用入口层：

- `scripts/run_case.py`
- `scripts/run_full_dag_ml.py`
- `scripts/export_paper_phmga.py`
- `src/evaluation/*`
- `src/manual_workflow.py`
- `src/dag_artifacts.py`

这部分能力支持：

- builder-only 生成 DAG
- 基于保存 state 直接跑 full-DAG 叶子级 shallow ML
- 导出 DAG JSON / PNG / DOT
- 汇总论文表与分析 markdown

### 3.5 RM101 变量工况工具与 simulated planner

围绕 RM101 变转速、变载荷工况，本轮新增了定制工具和 simulated planner 梯度：

- `src/tools/rm101_variable_speed_schemas.py`
- `src/rm101_metadata.py`
- `src/simulated_rm101.py`
- `config/case_exp2_paper.yaml`
- `config/case_exp2_all_domains_mixed_paper.yaml`

主要新增能力包括：

- `order_track_resample`
- `tsa_cycle_average`
- `order_band_energy`
- `sideband_ratio`
- `torque_normalize`

同时构建了 3 个 simulated planner 梯度：

1. `Gemini 2.0 Flash (simulated)`
2. `Gemini 2.5 Flash (simulated)`
3. `Gemini 2.5 Pro (simulated)`

并在 RM101 all-domain mixed 设定下完成了完整评估。

### 3.6 RM101 论文结果、诊断报告与 TeX 包

本轮已经形成完整论文产物链：

- `artifacts/paper/rm101_all_domains_mixed_simulated_v1`
- `RM101_DIAGNOSTIC_REPORT.md`
- `RM101诊断报告_中文.md`
- `paper_phmga/thesis_rm101_case/`

其中：

- `RM101诊断报告_中文.md` 面向诊断专家，强调工况、分支机理与工程建议。
- `paper_phmga/thesis_rm101_case/` 是独立 TeX 包，包含正文 case、附录报告、表格、图像与 standalone preview。

## 4. 实验与论文产物

### 4.1 当前推荐论文主结果

当前推荐主结果目录：

- `artifacts/paper/rm101_all_domains_mixed_simulated_v1`

主表关键结果如下：

- `BigModel / GLM-4.7-FlashX (real baseline)`：
  - `Accuracy = 0.8141`
  - `Macro-F1 = 0.8229`
- `Gemini 2.0 Flash (simulated, all-domain mixed)`：
  - `Accuracy = 0.1539`
  - `Macro-F1 = 0.1305`
- `Gemini 2.5 Flash (simulated, all-domain mixed)`：
  - `Accuracy = 0.2585`
  - `Macro-F1 = 0.2525`
- `Gemini 2.5 Pro (simulated, all-domain mixed)`：
  - `Accuracy = 0.8865`
  - `Macro-F1 = 0.8865`

跨图谱 late fusion：

- `Accuracy = 0.8241`
- `Macro-F1 = 0.8299`

结论：

- 当前最优单模型来自 `Gemini 2.5 Pro (simulated, all-domain mixed)`
- late fusion 优于多数弱图谱，但没有超过最优单图谱

### 4.2 当前推荐诊断报告

面向工程和结果汇报的入口为：

- `RM101_DIAGNOSTIC_REPORT.md`
- `RM101诊断报告_中文.md`

其中中文版已经补齐：

- 数据集与工况定义
- 通道物理意义
- 最优 DAG 与最优分支解释
- 与基线及融合结果对比
- 后续代码与工程建议

### 4.3 当前推荐 TeX 论文片段包

当前推荐论文插入包为：

- `paper_phmga/thesis_rm101_case/`

其包含：

- 正文 case 片段
- 附录片段
- 局部表格
- 最优 DAG 图
- standalone preview

## 5. 仓库边界与提交注意事项

### 5.1 `paper_phmga` 不是普通目录

需要特别注意：

- 主仓库中 `paper_phmga` 的跟踪模式是 `160000`
- 这意味着它是 **gitlink**
- 它不是当前主仓库里的普通文件夹

因此：

- `paper_phmga` 内部文件不能直接当作主仓库普通文件提交
- 必须先在 `paper_phmga` 自己的仓库中提交
- 然后再由主仓库更新 gitlink 指针

### 5.2 `artifacts/` 不纳入代码提交

本轮大量结果保存在：

- `artifacts/rm101_*`
- `artifacts/paper/*`

这些目录仍然是结果引用来源，不建议进入 git 历史。

### 5.3 压缩包默认不提交

当前 `paper_phmga` 内还有：

- `thesis_rm101_case.zip`

该文件应视为打包产物，默认不提交。

## 6. 建议提交批次

本轮建议按 **7 批细粒度** 提交主仓库，再对 `paper_phmga` 做独立提交。

### Batch 1：主链与配置回归

目标：

- 恢复 graph-first 主链
- 去除 runtime 旁路
- 回归 case config 直读

建议包含：

- `main.py`
- `src/phm_outer_graph.py`
- `src/config/*`
- `config/case*.yaml`
- 删除 `src/runtime/*`
- 删除 `src/phm_outer_graph_with_report.py`
- 删除旧 `config/data/*`, `config/experiment/*`, `config/runs/*`

### Batch 2：builder 质量与 planner/reflect

目标：

- 收紧 builder
- 引入非平凡 DAG 硬约束

建议包含：

- `src/builder_quality.py`
- `src/agents/plan_agent.py`
- `src/agents/reflect_agent.py`
- `src/prompts/plan_prompt.py`
- `src/prompts/reflect_prompt.py`
- 与 builder 质量直接相关的 state / graph transition 代码

### Batch 3：LLM backend 与报告链

目标：

- 稳定 provider 接入与报告生成

建议包含：

- `src/llm/*`
- `src/configuration.py`
- `src/agents/report_agent.py`
- `src/prompts/report_prompt.py`
- `src/config/__init__.py`

### Batch 4：评估层与脚本入口

目标：

- 建立 builder/state 复用后的 full-DAG 评估链

建议包含：

- `scripts/*`
- `src/evaluation/*`
- `src/manual_workflow.py`
- `src/dag_artifacts.py`
- `src/data/*`
- `src/agents/shallow_ml_agent.py`
- `src/agents/dataset_preparer_agent.py`
- `src/agents/execute_agent.py`

### Batch 5：RM101 variable-speed 与 simulated planner

目标：

- 引入 RM101 工况特化工具与 simulated planner 梯度

建议包含：

- `src/simulated_rm101.py`
- `src/rm101_metadata.py`
- `src/tools/rm101_variable_speed_schemas.py`
- `src/tools/__init__.py`
- `config/case_exp2_paper.yaml`
- `config/case_exp2_all_domains_mixed_paper.yaml`

### Batch 6：项目文档与诊断报告

目标：

- 补齐工程说明与结果报告

建议包含：

- `README.md`
- `CLAUDE.md`
- `agent.md`
- `RM101_DIAGNOSTIC_REPORT.md`
- `RM101诊断报告_中文.md`
- `doc/branch_update_thesis_2026.md`

### Batch 7：测试收口与 gitlink 更新

目标：

- 完成测试替换、删除旧 runtime 测试
- 在 paper 子仓库提交完成后更新 gitlink

建议包含：

- `tests/**`
- 删除旧 runtime 测试：
  - `tests/smoke/test_runtime_smoke.py`
  - `tests/unit/test_config_loader.py`
  - `tests/unit/test_feature_plan_and_backends.py`
  - `tests/unit/test_protocol_runtime.py`
- 新增当前主链与评估链测试
- 最后更新 `paper_phmga` gitlink 指针

### `paper_phmga` 独立提交

在 `paper_phmga` 仓库内部建议单独提交：

- `thesis_rm101_case/README.md`
- `main/rm101_case_main.tex`
- `appendix/rm101_case_appendix.tex`
- `tables/*.tex`
- `figures/rm101_best_dag.png`
- `assets/metadata.json`
- `standalone_preview.tex`

默认不提交：

- `thesis_rm101_case.zip`

`standalone_preview.pdf` 是否提交可选；如果希望 paper 子仓库保留预览件，可以单独一并纳入，否则建议不提交。

## 7. 验证情况

本轮分支工作中，当前主链和评估链已经完成一轮完整回归。最近一次明确记录的测试结果为：

- `57 passed`

此外，RM101 TeX 独立包已经用 `xelatex` 完成 standalone 编译烟雾测试，并成功生成预览 PDF。

## 8. 遗留事项

当前仍需注意的遗留点包括：

1. `README.md` 仍带有较多旧 demo 口径，需要结合当前 graph-first 主链继续清理。
2. `CLAUDE.md` 与 `agent.md` 需再检查是否完全反映当前实验入口。
3. `paper_phmga` 的提交必须与主仓库分开执行，否则容易把 gitlink 与内部文件混淆。
4. `artifacts/` 下的结果引用需要长期保留，但不应作为 git 提交内容。

## 9. 一句话结论

本轮 `thesis_2026` 分支的核心成果，是把仓库从 runtime 旁路重新收敛到 graph-first 主链，并在此基础上补齐了 builder 质量控制、full-DAG 评估、RM101 变量工况工具、simulated planner 梯度、诊断报告和论文 TeX 片段包，当前已经具备按 7 批细粒度逐步整理提交的条件。
