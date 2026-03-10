# Bug 修复代码证据汇总

**验证日期**: 2026-02-17

---

## 证据锚点列表

### Builder 迭代限制
```python
# 文件: src/phm_outer_graph.py
# 行号: 94-99
def _should_continue(state: PHMState) -> str:
    if not bool(getattr(state, "needs_revision", False)):
        return END
    iteration_count = int(getattr(state, "iteration_count", 0) or 0)
    max_iterations = int(getattr(state, "max_builder_iterations", 50) or 50)
    return "plan" if iteration_count < max_iterations else END
```

### Reflect 迭代计数
```python
# 文件: src/agents/reflect_agent.py
# 行号: 158-162
return {
    "needs_revision": needs_revision,
    "reflection_history": history,
    "iteration_count": int(state.iteration_count) + 1,
}
```

### Inquirer 数值稳定性
```python
# 文件: src/agents/inquirer_agent.py
# 行号: 7-22
def _calc_metric(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
    if metric == "pearson":
        std_a = float(np.std(a))
        std_b = float(np.std(b))
        if std_a < 1e-12 or std_b < 1e-12:
            return 1.0
        r = np.corrcoef(a, b)[0, 1]
        if np.isnan(r):
            return 1.0
        return float(1 - r)
```

### Backend 白名单
```python
# 文件: src/phm_outer_graph.py
# 行号: 142-148
def _train_models(state: PHMState) -> dict:
    backend = (getattr(state, "train_backend", None) or "shallow").lower()
    allowed_backends = {"shallow", "tspn", "both"}
    if backend not in allowed_backends:
        raise ValueError(
            f"Invalid train_backend '{backend}'. Expected one of: {sorted(allowed_backends)}"
        )
```

---

## 按文件索引

| 文件 | 修复数量 | 关键行号 |
|------|----------|----------|
| src/phm_outer_graph.py | 2 | 94-99, 142-148 |
| src/agents/reflect_agent.py | 1 | 161 |
| src/agents/inquirer_agent.py | 1 | 7-22 |
| src/agents/dataset_preparer_agent.py | 1 | ~11 |
| src/tools/multi_schemas.py | 1 | ~71 |
| src/tools/aggregate_schemas.py | 1 | ~304 |
| src/tools/transform_schemas.py | 2 | ~69, ~166 |
| src/model/explainable/tspn.py | 1 | ~246 |
| src/agents/deep_model_train_agent.py | 1 | ~827 |
| src/configuration.py | 1 | ~113 |
| src/utils/preflight.py | 1 | ~114 |
| src/utils/__init__.py | 1 | ~590 |
| src/utils.py | 1 | ~384 |

---

## Git 提交历史 (相关)

建议使用以下命令查看相关提交:
```bash
git log --oneline --grep="iteration\|backend\|NaN\|cosine\|pearson" --all
```
