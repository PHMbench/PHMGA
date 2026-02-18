# BUG 报告: shallow_ml_agent 模块

**审查者**: Agent 5 (bug-review-team)
**审查日期**: 2026-02-15
**报告版本**: 1.0

---

## 1. 审查范围

| 文件路径 | 代码行数 | 状态 |
|---------|---------|------|
| `src/agents/shallow_ml_agent.py` | 179 行 | 已审查 |

---

## 2. BUG 汇总

| 严重程度 | 数量 |
|---------|------|
| 高 | 3 |
| 中 | 4 |
| 低 | 2 |
| **总计** | **9** |

---

## 3. 高严重程度 BUG

### 3.1 pandas 导入失败后的静默处理问题

**位置**: `src/agents/shallow_ml_agent.py:10-12, 37-42, 141-154`

**代码片段**:
```python
try:
    import pandas as pd
except ImportError:
    pd = None

# ...

def shallow_ml_agent(...):
    if not datasets or pd is None:
        return {
            "models": {},
            "ensemble_metrics": {"accuracy": 0.0, "f1": 0.0},
            "metrics_markdown": "Pandas not available or no datasets provided.",
        }

# ...

metrics_df = pd.DataFrame.from_dict(metrics_data, orient="index")  # 未检查 pd 是否为 None
```

**问题描述**:
1. 函数开始时检查 `pd is None` 并提前返回
2. 但在函数末尾（第 141 行）直接使用 `pd.DataFrame.from_dict()` 而未再次检查
3. 如果 pandas 在函数执行过程中变为不可用（理论上不太可能，但代码不一致），会导致 `AttributeError: 'NoneType' object has no attribute 'DataFrame'`

**建议修复**:
```python
# 在函数末尾使用 pandas 前再次检查
if pd is not None:
    metrics_data = {node_id: m["metrics"] for node_id, m in models.items()}
    metrics_df = pd.DataFrame.from_dict(metrics_data, orient="index")
    # ...
else:
    metrics_markdown = "Pandas not available."
```

---

### 3.2 空数组 `np.stack()` 可能导致 ValueError

**位置**: `src/agents/shallow_ml_agent.py:123-124, 132`

**代码片段**:
```python
valid_probas = [
    probas[node_id] for node_id in high_quality_node_ids
    if node_id in probas and probas[node_id].shape[0] == len(y_truth)
]
if valid_probas:
    avg_proba = np.mean(np.stack(valid_probas), axis=0)  # 如果 valid_probas 为空，np.stack 会失败

valid_preds = [
    predictions[node_id] for node_id in high_quality_node_ids
    if node_id in predictions and len(predictions[node_id]) == len(y_truth)
]
if valid_preds:
    preds_arr = np.stack(valid_preds)  # 同样的问题
```

**问题描述**:
虽然代码中有 `if valid_probas:` 和 `if valid_preds:` 的检查，但这个检查只针对列表是否为空。然而，`np.stack()` 在以下情况下仍可能失败：
1. 数组维度不一致时
2. 数组形状不匹配时

这会导致运行时 `ValueError` 而没有被捕获。

**建议修复**:
```python
if valid_probas:
    try:
        # 检查所有数组形状是否一致
        if len(set(p.shape for p in valid_probas)) == 1:
            avg_proba = np.mean(np.stack(valid_probas), axis=0)
            ensemble_pred = np.argmax(avg_proba, axis=1)
        else:
            # 形状不一致，跳过 soft_voting
            pass
    except (ValueError, np.AxisError) as e:
        print(f"--- Ensemble: Soft voting failed due to shape mismatch: {e}")
```

---

### 3.3 `scipy.stats.mode` 在 SciPy 1.11+ 中的行为变更

**位置**: `src/agents/shallow_ml_agent.py:133`

**代码片段**:
```python
ensemble_pred = mode(preds_arr, axis=0, keepdims=False).mode
```

**问题描述**:
在 SciPy 1.11.0+ 中，`scipy.stats.mode` 的返回值从 namedtuple 改为 `ModeResult` 对象，且 `.mode` 属性的访问方式可能因版本而异。更重要的是：
1. 当 `preds_arr` 中所有值都相同且存在 NaN 时，结果可能不可预测
2. 当存在平局（tie）时，`mode` 返回第一个遇到的值，但没有警告用户

**建议修复**:
```python
from scipy.stats import mode as scipy_mode

mode_result = scipy_mode(preds_arr, axis=0, keepdims=False, nan_policy='omit')
if hasattr(mode_result, 'mode'):
    ensemble_pred = mode_result.mode
else:
    # 降级处理
    ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds_arr)
```

---

## 4. 中严重程度 BUG

### 4.1 `X_train` 和 `X_test` 维度不一致可能导致模型崩溃

**位置**: `src/agents/shallow_ml_agent.py:56-79`

**代码片段**:
```python
if not all(isinstance(arr, np.ndarray) for arr in [X_train, y_train, X_test, y_test]):
    continue

est = _build_estimator(algorithm)

# ... 没有检查 X_train 和 X_test 的特征维度是否一致

est.fit(X_train, y_train)
```

**问题描述**:
代码只检查了所有数组是否为 `np.ndarray` 类型，但没有验证：
1. `X_train.shape[1] == X_test.shape[1]`（特征维度一致）
2. `X_train.shape[0] == y_train.shape[0]`（样本数一致）
3. `X_test.shape[0] == y_test.shape[0]`（测试样本数一致）

如果这些条件不满足，`est.fit()` 会抛出难以理解的错误。

**建议修复**:
```python
if not all(isinstance(arr, np.ndarray) for arr in [X_train, y_train, X_test, y_test]):
    continue

# 添加维度验证
if X_train.ndim != 2 or X_test.ndim != 2:
    continue
if X_train.shape[1] != X_test.shape[1]:
    continue  # 特征维度不一致
if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
    continue  # 样本数不匹配
```

---

### 4.2 `algorithm` 参数大小写敏感但只有部分情况处理

**位置**: `src/agents/shallow_ml_agent.py:20-24`

**代码片段**:
```python
def _build_estimator(algorithm: str) -> Any:
    """Return a scikit-learn estimator based on ``algorithm``."""
    if algorithm.upper() == "SVM":
        return SVC(probability=True, random_state=42)
    return RandomForestClassifier(random_state=42)
```

**问题描述**:
1. 只检查了 `SVM`（大小写不敏感），但 `RandomForest` 是大小写敏感的
2. 如果用户传入 `"randomforest"` 或 `"random_forest"`，会得到 `RandomForestClassifier`，但这不符合预期
3. 无效的算法名称会静默返回 `RandomForestClassifier`，没有警告

**建议修复**:
```python
def _build_estimator(algorithm: str) -> Any:
    """Return a scikit-learn estimator based on ``algorithm``."""
    algo_normalized = algorithm.upper().replace("_", "").replace(" ", "")

    if algo_normalized == "SVM" or algo_normalized == "SVC":
        return SVC(probability=True, random_state=42)
    elif algo_normalized == "RANDOMFOREST" or algo_normalized == "RF":
        return RandomForestClassifier(random_state=42)
    else:
        import warnings
        warnings.warn(f"Unknown algorithm '{algorithm}', defaulting to RandomForest")
        return RandomForestClassifier(random_state=42)
```

---

### 4.3 `ensemble_method` 参数未验证

**位置**: `src/agents/shallow_ml_agent.py:27-33, 118`

**代码片段**:
```python
def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]],
    *,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",  # 未验证
    cv_folds: int = 5,
) -> Dict[str, Any]:

# ...

if ensemble_method == "soft_voting" and probas:
    # soft voting 逻辑
else:
    # hard voting 逻辑（默认）
```

**问题描述**:
1. `ensemble_method` 参数没有验证，用户可以传入任意值
2. 如果传入 `"soft"`、`"hard"` 或其他拼写错误，代码会默认使用 hard voting 而不警告用户
3. 这可能导致用户以为使用了 soft voting，实际却用了 hard voting

**建议修复**:
```python
def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]],
    *,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    cv_folds: int = 5,
) -> Dict[str, Any]:
    # 验证 ensemble_method
    valid_methods = {"hard_voting", "soft_voting"}
    if ensemble_method not in valid_methods:
        import warnings
        warnings.warn(f"Invalid ensemble_method '{ensemble_method}', defaulting to 'hard_voting'")
        ensemble_method = "hard_voting"
```

---

### 4.4 `cv_folds` 参数边界条件检查不足

**位置**: `src/agents/shallow_ml_agent.py:32, 63-64`

**代码片段**:
```python
cv_folds: int = 5,

# ...

if cv_folds >= 2 and len(np.unique(y_train)) > 1:
    cv = min(cv_folds, len(np.unique(y_train)))
```

**问题描述**:
1. `cv_folds` 可以是负数或非常大的数，但只检查了 `>= 2`
2. 如果 `cv_folds` 大于训练集样本数，`cross_validate` 会报错
3. 代码中 `cv = min(cv_folds, len(np.unique(y_train)))` 只限制了类别数，而不是样本数

**建议修复**:
```python
# 验证 cv_folds
if cv_folds < 2:
    cv_folds = 2  # 或者禁用交叉验证
min_samples_per_fold = 2
max_possible_folds = len(y_train) // min_samples_per_fold
cv_folds = min(cv_folds, max_possible_folds)

if cv_folds >= 2 and len(np.unique(y_train)) > 1:
    cv = min(cv_folds, len(np.unique(y_train)))
```

---

## 5. 低严重程度 BUG

### 5.1 重复导入 numpy

**位置**: `src/agents/shallow_ml_agent.py:8, 158`

**代码片段**:
```python
import numpy as np  # 第 8 行

# ...

if __name__ == "__main__":
    import numpy as np  # 第 158 行，重复导入
```

**问题描述**:
在 `if __name__ == "__main__":` 块中重复导入 `numpy`，虽然 Python 不会重复执行导入，但这是不必要的代码。

**建议修复**:
移除第 158 行的重复导入。

---

### 5.2 缺少类型注解

**位置**: `src/agents/shallow_ml_agent.py:20, 27`

**代码片段**:
```python
def _build_estimator(algorithm: str) -> Any:  # 返回类型过于宽泛
    # ...

def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]],  # 嵌套字典结构未明确定义
    # ...
) -> Dict[str, Any]:  # 返回类型过于宽泛
```

**问题描述**:
函数使用了 `Any` 作为返回类型，降低了类型安全性。IDE 无法提供准确的代码补全和类型检查。

**建议修复**:
```python
from typing import TypedDict, Union
from sklearn.base import BaseEstimator

class MetricsDict(TypedDict):
    accuracy: float
    f1: float
    cv_accuracy: float
    cv_f1: float
    cv_accuracy_std: float
    cv_f1_std: float

class ModelDict(TypedDict):
    metrics: MetricsDict
    model_b64: str

class ShallowMLResult(TypedDict):
    models: Dict[str, ModelDict]
    ensemble_metrics: Dict[str, float]
    metrics_markdown: str

def _build_estimator(algorithm: str) -> BaseEstimator:
    # ...

def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]],
    *,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    cv_folds: int = 5,
) -> ShallowMLResult:
    # ...
```

---

## 6. 统计汇总

| 严重程度 | 数量 | 百分比 |
|---------|------|--------|
| 高 | 3 | 33.3% |
| 中 | 4 | 44.4% |
| 低 | 2 | 22.2% |
| **总计** | **9** | **100%** |

---

## 7. 审查结论

`shallow_ml_agent.py` 模块整体结构清晰，但存在以下主要问题：

1. **输入验证不足**: 多个关键参数（`algorithm`、`ensemble_method`、`cv_folds`）缺少验证
2. **维度检查缺失**: 没有验证训练集和测试集的特征维度一致性
3. **异常处理不完善**: 部分操作（如 `np.stack`）可能在特定条件下抛出未捕获的异常

**建议优先修复高严重程度的 BUG**，特别是输入验证和维度检查问题。

---

*报告生成时间: 2026-02-15*
