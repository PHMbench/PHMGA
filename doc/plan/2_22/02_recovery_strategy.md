# 2.22 分流恢复策略（固定动作）

本文件定义三类失败的固定处理策略，避免重复试错。

---

## S1: `rc=137`（资源层）

### 触发条件

- `manifest` 中 `return_code=137`
- `run.log` 末尾出现 `已杀死` / `Killed` / `conda run ... failed`

### 处理动作（无代码优先）

1. 使用 `fast` 档位重跑失败 combo：
```bash
scripts/paper/run_combo.sh \
  --llm-tag <llm_tag> \
  --provider <provider> \
  --model <model> \
  --dataset-tag <ottawa|rm101> \
  --case-config <case_yaml> \
  --ablation-tag <ablation_tag> \
  --ablation-mode <full|no_reflect|no_prior> \
  --output-root save/paper_matrix/<llm_tag> \
  --train-profile fast \
  --env agent
```
2. 若仍失败，再用 `standard + no_reflect` 做通路确认。
3. 同一 combo 最多重跑 2 轮，避免盲目消耗。

### 成功判据

- 对应 combo 最新记录 `status=ok`。

### 升级路径

- 连续 2 轮仍 `137`：记为 `resource_blocked`，转 `04_teammate_board.md` 记录并暂停该 combo。

---

## S2: `403 model access`（权限/路由层）

### 触发条件

- `run.log` 出现 `llm_call.fail` 且 `Error code: 403`

### 处理动作

1. 检查 `.env`：`GLM_API_KEY`, `GLM_API_BASE`。
2. 检查 case `llm`：`provider=glm`, `query_generator_model=GLM-4.7-Flash`。
3. 先通过 Gate-A1（`zai`）与 Gate-A2（PHMGA）。
4. 通过后仅重跑单个 pilot combo，不直接全矩阵。

### 成功判据

- Gate-A1/A2 均通过，且 pilot 组合 `status=ok`。

### 升级路径

- 若 Gate-A1 失败：标记 `permission_blocked`，不放行 `m3_glm47` 全矩阵。

---

## S3: `out_total must be divisible by num_ops`（结构层）

### 触发条件

- `run.log` 出现：
  - `out_total=<n> must be divisible by num_ops=<k>`

### 处理动作（无代码优先）

1. 不复用有问题的历史 state：
   - 删除该 combo 的 `built_state.pkl` 后重跑。
2. 使用稳定模板：
   - `--train-profile fast` 先验证通路。
3. 若仍报错：
   - 固定使用 `config/model_tspn_basic.yaml`（模板网络）进行排错轮。

### 成功判据

- 训练阶段不再出现除不尽报错，combo 可完整结束。

### 升级路径

- 同类错误复现 2 次：开启最小代码修复计划（`doc/plan/2_23`）。

---

## 执行纪律

1. 禁止将 `m3_glm45/m4_glm47` 作为 active matrix 重跑。  
2. 任何失败必须记录 `error_class + evidence_path + next_action`。  
3. 只按 `03_execution_runbook.md` 规定顺序推进。  
