# 2.22 执行状态与 BUG 报告

**执行日期**: 2026-02-23
**执行人**: Claude Code

---

## 执行状态摘要

### Phase 0: 基线快照（恢复后）

| LLM | Combos | OK | Failed | 成功率 |
|-----|--------|----|----|----|----|
| m1_gemini25 | 6 | 4 | 2 | 66.7% |
| m2_gemini3 | 6 | 6 | 0 | 100% |
| m3_glm47 | 0 | 0 | 0 | Gate 未通过 |
| **Total** | **12** | **10** | **2** | **83.3%** |

### 详细状态

#### m1_gemini25 (4/6 OK)
- ✅ ottawa__m1_gemini25__A0_full: ok
- ✅ ottawa__m1_gemini25__A1_no_reflect: ok
- ✅ ottawa__m1_gemini25__A2_no_prior: ok
- ✅ rm101__m1_gemini25__A0_full: ok（S3 已规避）
- ❌ rm101__m1_gemini25__A1_no_reflect: failed (rc=2, S3: out_total % num_ops)
- ❌ rm101__m1_gemini25__A2_no_prior: failed (rc=2, S3: out_total % num_ops)

#### m2_gemini3 (6/6 OK) - 100%
- ✅ ottawa__m2_gemini3__A0_full: ok
- ✅ ottawa__m2_gemini3__A1_no_reflect: ok
- ✅ ottawa__m2_gemini3__A2_no_prior: ok (重跑成功，之前 rc=137)
- ✅ rm101__m2_gemini3__A0_full: ok
- ✅ rm101__m2_gemini3__A1_no_reflect: ok
- ✅ rm101__m2_gemini3__A2_no_prior: ok

#### m3_glm47 (0/0)
- 已尝试 Gate 流程，但当前无 active manifest（未形成可统计组合）

---

## 发现的 BUG

### BUG-1: nolds 包版本兼容性问题

**严重性**: 高
**状态**: 已修复（临时）

**问题描述**:
- `nolds==0.6.3` 在 Python 3.10 环境中存在 `importlib.resources.files()` 兼容性问题
- 错误信息: `TypeError: 'nolds.datasets' is not a package`

**影响**:
- 阻止 RM101 数据集的 preflight 检查通过
- 导致所有 M1 RM101 组合初始失败

**临时解决方案**:
```bash
pip install 'nolds<0.6.0'
```

**长期修复建议**:
- 在 `requirements.txt` 中固定 nolds 版本为 `<0.6.0`
- 或等待 nolds 发布兼容 Python 3.10 的新版本

---

### BUG-2: TSPN 结构可除性错误 (S3)

**严重性**: 高（阻塞 M1 RM101）
**状态**: 已修复（待重跑验证）

**问题描述**:
- RM101 配置中 `out_channels=4, scale=4` 导致 `out_total=16`
- `parallel_ops_per_layer=6` 与 `out_total=16` 不兼容（16 % 6 ≠ 0）
- 错误信息: `Layer 1: out_total=16 must be divisible by num_ops=6`

**证据路径**:
- `save/paper_matrix/m1_gemini25/_logs/rm101__m1_gemini25__A0_full/run.log`
- `save/paper_matrix/m1_gemini25/paper_rm101__m1_gemini25__A0_full/20260223-161642/model_config.resolved.yaml`

**影响**:
- M1 RM101 的 3 个组合全部失败（A0, A1, A2）
- 这是在 active 中复现的 S3 类型错误，已满足开启 `doc/plan/2_23/` 的触发条件

**触发条件**（根据 2.22 计划）:
- ✅ 同类问题在 active 复现 ≥ 2 次（实际复现 3 次）
- ✅ 阻塞影响 active matrix 覆盖率主目标

**修复方案（已实施）**:
1. 在 `src/model/explainable/builder.py` 增加自动可整除修正（向上取整）
2. 在 manifest 输出 `channel_adjustment` 记录修正信息
3. 新增回归测试 `tests/test_tspn_builder_divisibility_autofix.py`

---

## Gate 执行状态

### Gate-A1: zai 直连测试
**状态**: ✅ PASSED
**证据**: `scripts/paper/check_llm_gate.py --gate a1` 输出 `GATE_A1_PASS`

### Gate-A2: PHMGA 路由测试
**状态**: ✅ PASSED
**证据**: `scripts/paper/check_llm_gate.py --gate a2` 输出 `GATE_A2_PASS`

---

## 下一步行动

### 立即行动
1. **执行 doc/plan/2_23/** - S3 最小修复后的重跑验证
2. 重新运行 M1 RM101 的 3 个失败组合

### Phase 3: M3 激活（待执行）
- 执行 m3_glm47 pilot (ottawa A0)
- 根据 pilot 结果决定是否运行全矩阵

### Phase 4: 去重汇总
- 生成 manifest_dedup.jsonl
- 合并 all_llm_manifest.jsonl
- 生成 paper_main_results.csv

---

## 配置变更记录

### 已修改
- `agent` conda 环境: 降级 `nolds<0.6.0`
- 添加依赖: `librosa`, `antropy`

### 待修改
- `config/case_exp_gearbox_rm101.yaml`: `parallel_ops_per_layer: 6 → 4`
