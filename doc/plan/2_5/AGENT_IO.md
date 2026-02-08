## AGENT_IO（外环/内环智能体输入输出规范）—— Unified-X 可解释网络闭环

目标：让闭环“可复现、可诊断、可审计”。外环只改配置，内环只训练与产证据。

---

### 1) 内环训练器输出：`TrainReport`（JSON）

**最小必须字段**（建议与现有 `doc/plan/2_2/AGENT_IO.md` 保持兼容）：

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
    },
    "feature_stats": {
      "tokens": ["Mean","Std"],
      "per_token": {"Mean": {"mean_abs": 0.0}}
    }
  },
  "error_modes": [
    {"code": "CONFUSION_HIGH_FREQ", "detail": "classA->classB"}
  ],
  "artifacts": {
    "metrics_json": "path",
    "predictions_csv": "path",
    "operator_importance": "path",
    "wavefilters_params": "path",
    "feature_stats": "path",
    "crud_history": "path"
  }
}
```

约束：
- 若 `allow_test_labels_for_reporting=false`，`test_*` 必须为 `null`（只允许输出预测）。
- `operator_importance` 必须使用 `op_uid`，不得仅用 module_key。

---

### 2) 外环智能体输出：`ConfigPatch`（JSON，白名单字段）

外环只输出**可验证、可应用**的 patch；训练器只接受 patch 后的 `model_config.yaml`。

```json
{
  "diff_summary": "string",
  "reason_codes": ["OVERFIT", "CONFUSION_HIGH_FREQ"],
  "config_patch": {
    "model": {
      "signal_processing_configs": {
        "layer1": ["I", "WF", "HT"]
      },
      "feature_extractor_configs": ["Mean", "Std", "Kurtosis"],
      "disabled_ops": {
        "L2:WF:1": 1e-6
      }
    },
    "train": {
      "learning_rate": 0.001,
      "num_epochs": 20,
      "batch_size": 64
    }
  }
}
```

约束（硬）：
- 只允许修改：`model.*`、`train.*`、`explain.*`（字段白名单由 Pydantic schema 校验）。
- 禁止修改任何路径、任何 reader 名称、任何 python import（否则不可复现）。
- `disabled_ops` key 必须是 `op_uid`。

---

### 3) `reason_codes`（最小枚举）

- `INVALID_DIVISIBILITY`：`out_channels*scale` 不可整除 op 数量
- `INVALID_SHAPE`：`in_dim/in_channels` 与数据不匹配
- `COMPLEX_DTYPE`：某 token 产生 complex 未对齐为 real
- `OVERFIT` / `UNDERFIT`
- `CONFUSION_HIGH_FREQ` / `CONFUSION_LOW_FREQ`
- `OP_REDUNDANT`：重要性近似均匀
- `OP_COLLAPSE`：塌陷到单一算子但性能不佳

---

### 4) 必须的 Smoke Run（10 秒级）

每次外环生成新配置后，必须先跑 smoke：
- `epochs=1`
- `batch_size=2`
- `subset=10 samples`（或 `n_episodes=2`）

Smoke 不通过：直接返回 `INVALID_*` / `COMPLEX_DTYPE` 并回退配置（不得进入长训练）。

