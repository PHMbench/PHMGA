# 2.23 最小代码修复分支（S3）

本目录用于承接 `2.22` 中触发的 **S3 结构错误**最小修复：

- 错误现象：`out_total must be divisible by num_ops`
- 触发范围：`m1_gemini25` 的 RM101 三个组合
- 修复原则：最小改动、可回归、仅修复 active 阻塞链路

## 文档索引

- `s3_min_fix_plan.md`：修复方案、测试与回归步骤

## 交付边界

本分支不扩展新功能，不修改矩阵协议，不引入 baseline/统计检验。
