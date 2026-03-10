# 05 — Risk Register（风险登记与分流）

---

## R1. RM101 metadata 异常（人工修复路径）

- 触发信号：
  - CSV 读入失败（tokenize/column mismatch）
  - `preflight` 或 `case1` 在数据加载阶段报错
- 动作：
  1. 在 `gear_metadata.xlsx` 中移除/替换异常行
  2. 在本文件记录 `drop_list`（文件名、原因、日期）
  3. 重跑同 seed 同配置
- 证据：
  - `run.log` 首个 ERROR
  - 修复后 `preflight` 通过日志

---

## R2. 资源终止（`rc=137`）

- 触发信号：进程被 kill，无业务异常堆栈。
- 动作：
  1. 先改 `train_profile=fast` 验证通路
  2. 通路通过后回到 `standard/highacc`
  3. 单组合最多两轮重试
- 升级条件：两轮后仍 `137`，记录为 `blocked_resource`，不继续盲跑。

---

## R3. LLM 门禁失败（权限/路由）

- 触发信号：
  - `check_llm_gate.py --gate a2` 失败
  - 返回 `401/403` 或空响应
- 动作：
  1. 核对 `.env` 的 key/base
  2. 核对 case yaml 的 `llm.provider/model`
  3. 通过 Gate 后再放行矩阵
- 升级条件：连续失败 2 次，标记 `blocked_provider`，暂停该主线。

---

## R4. 协议偏差（protocol_invalid）

- 触发信号：缺少任一锁定字段（见 `01_protocol_lock.md`）。
- 动作：
  1. 直接标记 `protocol_invalid`
  2. 不进入主表，不参与 mean/std
  3. 修正配置后重跑

---

## R5. 结构错误（S3: `out_total % num_ops != 0`）

- 触发信号：训练阶段出现整除约束错误。
- 动作：
  1. 优先用当前稳定模板重跑验证
  2. 若 active 复现 ≥2 次，转入 `doc/plan/2_23/s3_min_fix_plan.md`
- 说明：此项属于“可触发最小代码修复”的唯一技术风险之一。

---

## R6. 解释性证据缺失

- 触发信号：`operator_importance` / `wavefilters_params` / `feature_stats` / `predictions.csv` 缺失。
- 动作：
  1. 标记 run 为 `evidence_incomplete`
  2. 仅用于性能统计，不进入解释性主图
  3. 补跑或降级为附录说明

---

## R7. 胜出口径误用

- 触发信号：未完成 4-seed mean 就写“超过基线”。
- 动作：
  1. 强制改回 interim 语气
  2. 仅在 `4-seed mean > 79.75%` 时允许最终胜出结论

---

## R8. Gemini-3 在线联通阻断（网络层）

- 触发信号（2026-02-24 实测）：
  - `check_llm_gate.py --gate a2` 返回 `APIConnectionError: Connection error`
  - 直连 `OPENAI_BASE_URL=https://api.v3.cm/v1` 出现 `NameResolutionError`（DNS 不可达）
- 动作：
  1. 先确认运行机网络策略（DNS/出口）是否允许访问 provider base
  2. 若环境带 `HTTP_PROXY/HTTPS_PROXY`，确认代理可用；不可用则移除或改为可达代理
  3. Gate-A2 未通过时，在线实验全部标记 `blocked_network`
- 证据：
  - `save/paper_matrix/2_24/_resolved_cases/gate_a_rm101_gemini3.yaml`
  - 终端日志：`GATE_A2_FAIL invoke: APIConnectionError`

---

## R9. Direct-ML（shallow）主线无训练指标产物

- 触发信号（2026-02-24 seed0 实测）：
  - run 成功结束，但 case 目录仅有 `final_report`、`run.log`，缺 `metrics.json/predictions.csv`
  - `built_state.pkl` 中 `ml_results` 为空
- 动作：
  1. 2_24 统计阶段将该 run 标记为 `evidence_incomplete`
  2. 不纳入 “4-seed mean test_acc” 主判定
  3. 若后续继续双主线对比，需先在下一轮最小代码修复中补齐 shallow 指标落盘
- 证据：
  - `save/paper_matrix/2_24/paper_rm101__m2_gemini3__shallow__seed0__A0_full/final_report.md`
  - `save/paper_matrix/2_24/paper_rm101__m2_gemini3__shallow__seed0__A0_full/run-*/logs/run.log`

---

## Drop List（人工维护）

| Date | File/ID | Action | Reason | Owner |
|---|---|---|---|---|
|  |  |  |  |  |
