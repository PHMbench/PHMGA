# 00 — Gap Assessment（与对比基线的差距评估）

## 1) 当前硬事实

### 对比目标（来自外部对比文档）

- 文件：`/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2/paper/LQ_vibench_fix/report/paper_rm101_dg_experiments_2_22_2_23.md`
- 基线：`WKN` 在 RM101 DG 下 `4-seed mean test_acc = 79.75%`
- 判定阈值：仅当 `4-seed mean test_acc > 79.75%` 才能宣称“超过对比模型”

### PHMGA 当前最佳（截至本仓库现有产物）

- 文件：`save/paper_matrix/paper_main_results.csv`
- 组合：`m2_gemini3 + RM101 + A0_full`
- 指标：`test_acc = 0.6197916667`（61.98%）

## 2) 差距计算

- 当前最优：`61.98%`
- 目标基线：`79.75%`
- 差距：`-17.77 pp`

**结论（当前时点）**：未达标，不能宣称超过对比模型。

## 3) 为什么当前结论还不够

1. 目前仅有 Gemini-3 既有结果可用，且不是 2_24 协议下的完整 4-seed 双主线结果。  
2. 当前 `save/paper_matrix` 以矩阵流程为主，`train_backend` 主要是 `tspn`，`shallow` 主线尚未按统一协议完成。  
3. 需要在 `2_24` 中严格补齐：
   - 协议一致（DG 域划分明确）
   - 双主线一致（`shallow` 与 `tspn`）
   - 4-seed 可统计（mean/std）

## 4) 2_24 目标判定（不可修改）

- **Pass**：任一 Gemini-3 主线满足 `4-seed mean test_acc > 79.75%`
- **Fail**：否则输出“未超过，差距 X pp”并给出下一轮最小改进建议

