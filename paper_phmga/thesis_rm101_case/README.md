# thesis_rm101_case

本目录是 RM101 论文 case 的独立 TeX 包，可直接并入外部 thesis 主工程。

## 用法

- 正文插入：
  - `\input{paper_phmga/thesis_rm101_case/main/rm101_case_main.tex}`
- 附录插入：
  - `\input{paper_phmga/thesis_rm101_case/appendix/rm101_case_appendix.tex}`

## 内容来源

本包对应当前 RM101 主结果：

- `artifacts/paper/rm101_all_domains_mixed_simulated_v1`

最佳单模型图谱与结果来自：

- `artifacts/rm101_all_domains/openrouter__google__gemini-2.5-pro__simulated_mixed_paper_v1`

## 目录说明

- `main/rm101_case_main.tex`
  - 正文用 case study 片段
- `appendix/rm101_case_appendix.tex`
  - 附录用补充诊断报告
- `tables/*.tex`
  - 正文和附录直接 `\input{}` 的表格
- `figures/rm101_best_dag.png`
  - 最优 DAG 图
- `assets/metadata.json`
  - 来源与关键指标的结构化摘要
- `standalone_preview.tex`
  - 独立编译烟雾测试文件

## 说明

- 这套内容不直接修改外部 thesis 主文件。
- 正文中的 3 个 Gemini 结果固定标注为 `simulated planner variants`，不表述为真实在线 API 测试结果。
- 当前正文采用完整四行主表，不只写最优结果。
