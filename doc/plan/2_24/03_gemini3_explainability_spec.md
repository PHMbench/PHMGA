# 03 — Gemini-3 可解释性规范（Markdown + PNG）

本规范用于 Gemini-3 双主线（`shallow` / `tspn`）的解释性证据统一输出，确保能直接进入论文。

---

## 1) 每个成功 run 的最小证据

从每个 run 的 `run_dir` 收集：

1. `explain/operator_importance.json`
2. `explain/wavefilters_params.json`
3. `explain/feature_stats.json`
4. `predictions.csv`

若缺失任一项，需在 `05_risk_register.md` 记录：
- `run_id`
- 缺失文件
- 是否影响论文主图

---

## 2) 必出图（PNG）

## 图1：Operator Importance（层内算子权重）

- 输入：`operator_importance.json`
- 图型：水平条形图（每层 top-k 算子）
- 输出：`fig_operator_importance_<backend>_seed<seed>.png`

## 图2：WF 参数分布（频带解释）

- 输入：`wavefilters_params.json`
- 图型：
  - 有 `fs_hz` 时：绘制 `fc_hz/fb_hz`
  - 无 `fs_hz` 时：绘制 `fc_norm/fb_norm`
- 输出：`fig_wf_band_<backend>_seed<seed>.png`

## 图3：错误模式图（混淆矩阵）

- 输入：`predictions.csv`
- 图型：normalized confusion matrix
- 输出：`fig_confusion_<backend>_seed<seed>.png`

---

## 3) 跨 seed 稳定性（核心）

每条主线单独统计：

1. **Top-K 算子重合率**
   - 每个 seed 取 top-k（建议 k=5）
   - 计算两两 Jaccard，再取平均
2. **WF 参数稳定性**
   - `fc`/`fb` 的 seed 间均值和标准差
3. **错误模式一致性**
   - 混淆矩阵按 seed 取平均后观察主误判对

建议输出：
- `explainability_stability_<backend>.md`
- `explainability_stability_<backend>.csv`

---

## 4) 解释性与性能联动写法（论文要求）

必须同时回答两个问题：

1. 性能层：是否超过 `79.75%`？  
2. 机制层：最关键的算子与频带是什么？跨 seed 是否稳定？

若性能未超过基线，仍可陈述解释性优势，但必须使用限制语气：
- “解释性证据支持诊断可读性提升，但当前准确率仍未超过 WKN 基线。”

---

## 5) 快速绘图命令模板（单 run）

```bash
python - <<'PY'
import json, pandas as pd, pathlib, matplotlib.pyplot as plt

run_dir = pathlib.Path("REPLACE_WITH_RUN_DIR")
out_dir = run_dir / "explain" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# 图1: operator importance
op_path = run_dir / "explain" / "operator_importance.json"
if op_path.exists():
    data = json.loads(op_path.read_text(encoding="utf-8"))
    labels, vals = [], []
    for layer in data.get("layers", []):
        for op in layer.get("operators", [])[:5]:
            labels.append(f"L{layer.get('layer_idx')}:{op.get('op_uid')}")
            vals.append(float(op.get("importance", 0.0)))
    if labels:
        plt.figure(figsize=(10, 4))
        plt.barh(labels, vals)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_operator_importance.png", dpi=180)
        plt.close()

# 图3: confusion (requires predictions.csv)
pred_path = run_dir / "predictions.csv"
if pred_path.exists():
    df = pd.read_csv(pred_path)
    if {"y_true", "y_pred"}.issubset(df.columns):
        mat = pd.crosstab(df["y_true"], df["y_pred"], normalize="index")
        plt.figure(figsize=(6, 5))
        plt.imshow(mat.values, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(mat.columns)), mat.columns, rotation=45)
        plt.yticks(range(len(mat.index)), mat.index)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_confusion.png", dpi=180)
        plt.close()
PY
```

---

## 6) 主表最小字段（解释性）

建议在论文附表增加：

| backend | seed | top1_operator | top5_overlap_mean | wf_param_std | main_error_pair | notes |
|---|---:|---|---:|---:|---|---|

