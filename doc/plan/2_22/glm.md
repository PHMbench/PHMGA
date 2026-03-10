# PHMGA 2.22 总控执行单（Control Plane）

**日期**: 2026-02-23  
**定位**: 2.22 失败恢复的总控入口页（不重复维护细节事实）

---

## 1) 目标与原则

### 目标
1. 恢复 active matrix 失败组合（`m1/m2/m3_glm47`）。
2. 形成可复现实验证据链（manifest + logs + results）。
3. 产出可直接进入论文中期更新的结果口径。

### 原则
- 优先恢复，不先改业务代码。
- legacy 目录仅归档，不进入 active KPI。
- 同类故障在 active 复现 2 次后，才开启最小代码修复分支（`2_23`）。

---

## 2) 当前状态快照（与基线文档一致）

> 来源：`doc/plan/2_22/01_failure_baseline.md`

- `m1_gemini25`: `4/6`（RM101 仍有两个组合为 `S3` 结构错误）
- `m2_gemini3`: `6/6`（当前已跑满）
- `m3_glm47`: `manifest.jsonl` 缺失（尚未形成 active run）
- legacy（归档，不参与 active 统计）：
  - 详见 `doc/plan/2_22/legacy_glm45_archive.md`

---

## 3) 执行主线（只保留编排）

> 详细命令见：`doc/plan/2_22/03_execution_runbook.md`

1. **Phase 0**：基线冻结（只读）
2. **Phase 1**：GLM-4.7 联通门禁
   - Gate-A1（`zai`）
   - Gate-A2（PHMGA `get_llm().invoke("OK")`）
3. **Phase 2**：先修可跑失败（`M2 -> M1`）
4. **Phase 3**：激活 `m3_glm47`（pilot -> full matrix）
5. **Phase 4**：去重汇总（`manifest_dedup` + `paper_main_results.csv` + `analysis_draft.md`）

---

## 4) Gate-A1 前置条件（强制）

执行 Gate 前，统一使用 `scripts/paper/check_llm_gate.py`（内部 `load_dotenv`）：

```bash
conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/check_llm_gate.py \
  --gate a1 \
  --provider glm \
  --model glm-4.7-flash
```

- 若失败：先安装 `zai` 或修复 `.env` 中 `GLM_API_KEY/GLM_API_BASE`。  
- 若暂时无法通过：记录限制，并以 Gate-A2 作为最小可执行门禁（必须在结果中标注“缺少 Gate-A1 证据”）。

---

## 5) 分流策略（摘要）

> 详细策略见：`doc/plan/2_22/02_recovery_strategy.md`

- `S1: rc=137`：资源层重跑策略（`fast` -> 次轮受控重跑）
- `S2: 403`：权限/路由门禁策略（未过 Gate 不放行）
- `S3: out_total % num_ops`：已在 active 复现，进入 `2_23` 最小修复分支

---

## 6) active / legacy 边界（硬约束）

### Active（允许执行）
- `save/paper_matrix/m1_gemini25`
- `save/paper_matrix/m2_gemini3`
- `save/paper_matrix/m3_glm47`

### Legacy（仅证据归档，禁止并入 active 统计）
- `doc/plan/2_22/legacy_glm45_archive.md` 中登记目录与证据

---

## 7) 条件分支：最小代码修复（仅触发式）

以下任一条件触发，才开 `doc/plan/2_23/`：

1. `S1/S2/S3` 同类问题在 active 复现 ≥ 2 次；
2. 同一 combo 在两轮受控重跑后仍无法推进；
3. 阻塞影响 active matrix 覆盖率主目标。

在未触发前，不在 2.22 主线中插入代码改造。

---

## 8) 并行执行与论文更新入口

- 并行分工看板：`doc/plan/2_22/04_teammate_board.md`
- 论文中期更新模板：`doc/plan/2_22/05_interim_paper_update.md`

---

## 9) DoD（总控页验收）

1. `glm.md` 状态数字与 `01_failure_baseline.md` 一致。
2. `glm.md` 执行顺序与 `03_execution_runbook.md` 一致。
3. 明确 legacy 禁止进入 active 统计。
4. Gate-A1 包含 `zai` 前置检查与失败分流说明。
