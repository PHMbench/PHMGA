## AGENT_IO（外环/内环智能体输入输出规范）

目标：把闭环跑稳、跑可复现。所有 agent 的输出必须严格结构化，禁止混入自然语言噪声。

---

### 1) 内环训练器输出：`TrainReport`（JSON）

必须字段（最小集合）：
```json
{
  "run_id": "string",
  "dataset_id": "string",
  "task_id": "string",
  "split_protocol": {
    "train": "ref_train|episodic_train",
    "val": "ref_val|episodic_val",
    "test": "tst|episodic_test",
    "allow_test_labels_for_reporting": false
  },
  "metrics": {
    "val_macro_f1": 0.0,
    "val_acc": 0.0,
    "test_macro_f1": null,
    "test_acc": null
  },
  "confusion_matrix": {
    "labels": ["0","1"],
    "matrix": [[0,0],[0,0]]
  },
  "explain_summary": {
    "operator_importance": [
      {"layer": 1, "topk": [{"op_uid":"L1:WF:0","score":0.7}]}
    ],
    "wavefilters": {
      "enabled": true,
      "fs_hz": 3125,
      "bands_hz_top": [[2000,4000]]
    }
  },
  "error_modes": [
    {"code": "CONFUSION_HIGH_FREQ", "detail": "classA->classB"}
  ],
  "artifacts": {
    "metrics_json": "path",
    "predictions_csv": "path",
    "operator_importance": "path",
    "crud_history": "path"
  }
}
```

约束：
- 若 `allow_test_labels_for_reporting=false`，则 `test_*` 指标必须为 `null`（只允许输出预测）。
- 必须包含 `op_uid`（见 SPEC），不得仅给 module_key。

---

### 2) 外环智能体输出：`ConfigPatch`（JSON）

外环永远只输出**白名单字段**的 patch；训练器只接受 patch 后的 `model_config.yaml`。

```json
{
  "diff_summary": "string",
  "reason_codes": ["OVERFIT", "CONFUSION_HIGH_FREQ"],
  "config_patch": {
    "model": {
      "signal_processing_configs": {
        "layer1": ["I", "WF", "HT"]
      },
      "disabled_ops": {
        "L2:WF:1": 1e-6
      }
    },
    "train": {
      "learning_rate": 0.001,
      "num_epochs": 20
    }
  }
}
```

约束：
- `disabled_ops` 必须使用 `op_uid` 作为 key。
- 禁止修改任何路径、保存目录、任意 import、任意 reader 名称等（只允许 model/train/explain/preprocess）。

---

### 3) 反思/决策依据：`reason_codes`（最小枚举）

- `INVALID_DIVISIBILITY`：`out_channels*scale` 不可整除 module_num
- `INVALID_SHAPE`：`in_dim/in_channels` 与数据不匹配
- `OVERFIT`：train/val gap 过大
- `UNDERFIT`：train/val 都低
- `CONFUSION_HIGH_FREQ`：高频相关类别混淆（需结合解释性报告）
- `OP_REDUNDANT`：某层算子重要性近似均匀/冗余
- `OP_COLLAPSE`：重要性塌陷到单一算子但性能不佳

---

### 4) 必须的 Smoke Run（10 秒级）

外环每次生成新配置后，先跑 `debug_mode=true`：
- `epoch=1`
- `batch_size=2`
- `subset=10 samples` 或 `n_episodes=2`

Smoke 通过才允许进入正式训练；否则返回 `INVALID_*` reason 并回退/再生成。
