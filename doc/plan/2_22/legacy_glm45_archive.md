# Legacy 归档：GLM-4.5 与历史目录证据

本页仅用于归档历史失败证据，不参与 active matrix 统计或重跑。

## 归档范围

1. `save/paper_matrix/m3_glm45`（GLM-4.5，已下线）
2. `save/paper_matrix/m4_glm47`（旧标签历史目录）

## 历史快照（保留口径）

### `m3_glm45`（已下线）

- 历史状态：`0/4` 成功
- 典型失败：`403 model access`
- 证据示例：
  - `save/paper_matrix/m3_glm45/_logs/ottawa__m3_glm45__A0_full/run.log`

### `m4_glm47`（历史目录）

- 历史状态：`0/6` 成功
- 典型失败：
  - `403 model access`（多数）
  - `Layer 1: out_total=16 must be divisible by num_ops=6`（结构错误）
- 证据示例：
  - `save/paper_matrix/m4_glm47/_logs/rm101__m4_glm47__A2_no_prior/run.log`

## 执行约束

1. 禁止将上述目录作为 active matrix 重跑入口。
2. 禁止将上述目录合并进 active 成功率或论文主表。
3. 若需引用，只能作为“历史失败证据”在附录展示。
