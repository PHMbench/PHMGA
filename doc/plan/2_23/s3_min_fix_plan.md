# S3 最小修复计划：`out_total % num_ops != 0`

## 背景

在 `m1_gemini25` 的 RM101 三个组合中复现同类失败：

- `Layer 1: out_total=16 must be divisible by num_ops=6`

该错误发生在 TSPN 构模阶段，属于 active matrix 阻塞项。

## 修复目标

1. 保持当前配置协议不变（不要求手工改 case/model yaml）。
2. 当 `out_total` 与 `num_ops` 不整除时自动修正到最近可整除值（向上取整）。
3. 在 manifest 中输出修正记录，便于审计与复现。
4. 增加单元测试锁定行为，防止回归。

## 最小改动方案（已落地）

### 1) 构模自动修正

- 文件：`src/model/explainable/builder.py`
- 变更：
  - 新增 `_resolve_out_channels_for_layers(...)`
  - 计算各层 `num_ops` 的 LCM，并将 `out_total` 调整到可整除值
  - 若发生调整，输出 warning 日志
  - `manifest` 新增 `channel_adjustment` 字段

### 2) 回归测试

- 文件：`tests/test_tspn_builder_divisibility_autofix.py`
- 覆盖：
  - 输入 `out_total=16`, `num_ops=6`
  - 断言自动调整为 `out_total=24`
  - 断言模型可前向运行

## 验收步骤

1. `PHM_ENABLE_TORCH_TESTS=1 pytest -q tests/test_tspn_builder_divisibility_autofix.py`
2. 重跑 `m1_gemini25` 的 RM101 三组合（A0/A1/A2）
3. 检查 run log 中不再出现 `must be divisible by num_ops`
4. 检查 `model_manifest.json`/manifest 中包含 `channel_adjustment`

## 风险与边界

1. 该修复会增加层通道数，可能增加训练成本。
2. 该修复不处理其它训练错误（数据、依赖、权限），只处理 S3。
3. 若后续出现性能波动，需要在 2.24+ 单独评估通道策略，不在本分支扩展。
