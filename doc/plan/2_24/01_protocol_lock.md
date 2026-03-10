# 01 — Protocol Lock（协议锁定）

本文件定义 2_24 的唯一可比协议。任何不满足以下字段的 run，统一标记 `protocol_invalid`，不进入主表统计。

---

## 1) 任务协议（必须）

用于 RM101 DG 对比时，case yaml 必须显式包含：

```yaml
data:
  task_type: DG
  dataset_name: RM_101_THU_GEARBOX
  source_domain_id: [0,1,2,3,4,5,6,7,8]
  target_domain_id: [9,10,11]
```

## 2) LLM 协议（Gemini-3）

```yaml
llm:
  provider: openai_compatible
  query_generator_model: gemini-3-flash-preview
  phm_model: gemini-3-flash-preview
  reflection_model: gemini-3-flash-preview
  answer_model: gemini-3-flash-preview
```

说明：
- provider/model 以 case yaml 为准；`.env` 只放 key/base。  
- 矩阵阶段默认 `PHM_REPORT_MODE=template`，减少报告生成阻塞。

## 3) 双主线定义（必须同协议）

### 主线 A：Direct ML
- `train_backend: shallow`

### 主线 B：TSPN
- `train_backend: tspn`

可选：
- `train_backend: both` 仅用于一致性检查，不作为主结论输入。

## 4) 统计协议（必须）

- seeds 固定：`[0, 42, 123, 999]`
- 主指标：`test_acc`
- checkpoint 选择：`val_loss`
- 标准差：population std（`ddof=0`）

## 5) 胜出判定（硬约束）

- 仅当某主线满足 `4-seed mean test_acc > 79.75%`，可写“超过对比模型”。  
- 否则必须写“未超过，差距 X pp”。

## 6) protocol_invalid 判定规则

以下任一成立，run 不进入主表：

1. 缺 `task_type` 或 `source_domain_id` 或 `target_domain_id`。  
2. `dataset_name` 非 `RM_101_THU_GEARBOX`。  
3. `llm.provider/model` 与 Gemini-3 路线不一致。  
4. 不是固定 seed 集（0/42/123/999）之一。  

