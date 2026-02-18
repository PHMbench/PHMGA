# Dataset Preparer Agent Bug Report

**审查日期**: 2026-02-15
**审查者**: Agent 4 (bug-review-team)
**审查范围**: `src/agents/dataset_preparer_agent.py`

## 1. 审查范围

| 文件 | 代码行数 | 主要功能 |
|------|----------|----------|
| `src/agents/dataset_preparer_agent.py` | 151 | 数据集准备代理：从 DAG 节点收集特征并组装训练/测试数据集 |

## 2. 发现的 BUG

### 2.1 高严重程度 (High)

#### BUG-001: NPZ 文件资源未正确关闭

**文件位置**: `src/agents/dataset_preparer_agent.py:69`

**代码片段**:
```python
else:
    # .npz: Iterate through the sample IDs found in the archive
    with data as npz:
        for sample_id in npz.files:
            if sample_id in labels_map:
                feature = npz[sample_id]
                features_list.append(feature.reshape(1, -1))
                labels_list.append(labels_map[sample_id])
```

**问题描述**:
当 `np.load()` 返回 `np.ndarray` 类型（即 `.npy` 文件）时，代码没有使用上下文管理器，而在 `.npz` 文件情况下使用了 `with data as npz`。然而，对于 `.npy` 文件，`data` 对象是一个 numpy 数组，`with` 语句的行为是不明确的，且 `np.ndarray` 不支持上下文管理器协议。实际上，`.npz` 文件应该通过 `np.load()` 返回的 `NpzFile` 对象来访问，该对象支持上下文管理器。

**建议修复**:
```python
else:
    # .npz: Iterate through the sample IDs found in the archive
    # np.load returns NpzFile for .npz files which supports context manager
    npz_file = np.load(feature_path, allow_pickle=False)
    try:
        for sample_id in npz_file.files:
            if sample_id in labels_map:
                feature = npz_file[sample_id]
                features_list.append(feature.reshape(1, -1))
                labels_list.append(labels_map[sample_id])
            else:
                print(
                    f"Warning: Sample ID '{sample_id}' found in feature file but not in labels map. Skipping."
                )
    finally:
        npz_file.close()
```

---

#### BUG-002: 空字典检查逻辑错误导致标签映射可能包含 None 值

**文件位置**: `src/agents/dataset_preparer_agent.py:48`

**代码片段**:
```python
def _build_dataset_from_features(feature_path: str, labels_map: Dict[str, Any], *, flatten: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a dataset (features and labels) by matching sample IDs from a feature
    file with a provided labels dictionary.
    """
    if not feature_path or not os.path.exists(feature_path) or not labels_map:
        return np.array([]), np.array([])
```

**问题描述**:
当 `labels_map` 中所有键的值都为 `None` 或 falsy 值时，`not labels_map` 仍然返回 `False`（因为字典非空），这会导致继续处理并在后续代码中产生 `None` 标签。此外，后续代码中 `labels_map.get(sample_id)` 可能返回 `None`，这会在最终标签数组中引入 `None` 值，导致训练时出现错误。

**建议修复**:
```python
def _build_dataset_from_features(feature_path: str, labels_map: Dict[str, Any], *, flatten: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a dataset (features and labels) by matching sample IDs from a feature
    file with a provided labels dictionary.
    """
    if not feature_path or not os.path.exists(feature_path) or not labels_map:
        return np.array([]), np.array([])

    features_list: list[np.ndarray] = []
    labels_list: list[Any] = []

    data = np.load(feature_path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        sample_ids = list(labels_map.keys())
        sample_id = sample_ids[0] if sample_ids else "sample"
        label = labels_map.get(sample_id)
        # Validate label is not None
        if label is None:
            print(f"Warning: Label for sample_id '{sample_id}' is None. Skipping.")
            return np.array([]), np.array([])
        # ... rest of the code
    else:
        npz_file = np.load(feature_path, allow_pickle=False)
        try:
            for sample_id in npz_file.files:
                if sample_id in labels_map:
                    label = labels_map[sample_id]
                    if label is None:
                        print(f"Warning: Label for sample_id '{sample_id}' is None. Skipping.")
                        continue
                    feature = npz_file[sample_id]
                    features_list.append(feature.reshape(1, -1))
                    labels_list.append(label)
                # ... rest of the code
        finally:
            npz_file.close()
```

---

#### BUG-003: DAG 遍历可能导致无限循环

**文件位置**: `src/agents/dataset_preparer_agent.py:24-30`

**代码片段**:
```python
# Keep moving to the parent until a node with no parents (the root) is found.
# This assumes a single-parent lineage for processed nodes, which is typical.
while current_node.parents:
    parent_id = current_node.parents[0]
    parent_node = all_nodes.get(parent_id)
    if not parent_node:
        # This should not happen in a well-formed DAG
        return {}, {}
    current_node = parent_node
```

**问题描述**:
如果 DAG 中存在循环（虽然理论上不应该，但代码没有防护），或者父节点链中某个节点的 `parents` 列表指向其子节点，这个 `while` 循环将无限循环。代码应该添加最大深度限制或循环检测机制。

**建议修复**:
```python
# Keep moving to the parent until a node with no parents (the root) is found.
# This assumes a single-parent lineage for processed nodes, which is typical.
visited = set()
MAX_TRAVERSAL_DEPTH = 1000
depth = 0

while current_node.parents:
    # Cycle detection
    if current_node.node_id in visited:
        print(f"Warning: Cycle detected in DAG traversal from node {node_id}.")
        return {}, {}
    visited.add(current_node.node_id)

    # Depth limit protection
    depth += 1
    if depth > MAX_TRAVERSAL_DEPTH:
        print(f"Warning: Maximum traversal depth exceeded from node {node_id}.")
        return {}, {}

    parent_id = current_node.parents[0]
    parent_node = all_nodes.get(parent_id)
    if not parent_node:
        # This should not happen in a well-formed DAG
        return {}, {}
    current_node = parent_node
```

---

### 2.2 中等严重程度 (Medium)

#### BUG-004: 缺少对标签值类型的验证

**文件位置**: `src/agents/dataset_preparer_agent.py:43-86`

**问题描述**:
函数 `_build_dataset_from_features` 没有验证 `labels_map` 中的值是否是有效的标签类型。如果标签值是不支持的类型（如复杂对象、嵌套字典等），将导致后续 ML 训练失败。

**建议修复**:
在函数开头添加标签类型验证：
```python
def _validate_labels(labels_map: Dict[str, Any]) -> bool:
    """Validate that all labels are of supported types."""
    valid_types = (int, float, str, bool, np.number)
    for label in labels_map.values():
        if label is None or not isinstance(label, valid_types):
            return False
    return True
```

---

#### BUG-005: 空数组处理不一致

**文件位置**: `src/agents/dataset_preparer_agent.py:120-121`

**代码片段**:
```python
if X_train.size == 0 and X_test.size == 0:
    continue
```

**问题描述**:
只有当**训练集和测试集都为空**时才跳过节点。这意味着如果只有测试集为空但训练集非空，会继续创建数据集。然而，对于 `DataSetNode` 的创建，代码假设至少有一个非空数据集。如果 `X_train.size == 0` 但 `X_test.size > 0`（在 `allow_test_labels_for_reporting=False` 时不会发生，但如果状态改变），`shape` 会被设置为 `X_test.shape`。此外，如果测试集为空，`n_test` 会被正确设置为 0，这可能是预期行为，但逻辑不够明确。

**建议修复**:
明确处理只有训练集或只有测试集非空的情况：
```python
# Ensure at least one dataset is non-empty
if X_train.size == 0 and X_test.size == 0:
    continue

# Determine the shape for the dataset node based on the non-empty dataset
if X_train.size > 0:
    ds_shape = X_train.shape
elif X_test.size > 0:
    ds_shape = X_test.shape
else:
    continue  # Should not reach here
```

---

#### BUG-006: 使用 print 而非适当的日志记录

**文件位置**: 多处 (第 76-78, 107 行)

**代码片段**:
```python
print(
    f"Warning: Sample ID '{sample_id}' found in feature file but not in labels map. Skipping."
)
```

**问题描述**:
代码使用 `print()` 进行警告输出，而不是使用 Python 的 `logging` 模块。这使得在生产环境中难以控制日志级别、格式和输出目标。

**建议修复**:
```python
import logging

logger = logging.getLogger(__name__)

# Replace print statements with:
logger.warning("Sample ID '%s' found in feature file but not in labels map. Skipping.", sample_id)
```

---

#### BUG-007: 状态属性访问未进行防御性检查

**文件位置**: `src/agents/dataset_preparer_agent.py:117`

**代码片段**:
```python
allow_test = bool(getattr(state, "allow_test_labels_for_reporting", False))
```

**问题描述**:
使用 `getattr` 带默认值是好的做法，但这表明 `state` 对象可能没有定义这个属性。然而，根据 `PHMState` 的定义，`allow_test_labels_for_reporting` 是一个已定义的字段，所以 `getattr` 是多余的。如果状态对象来自旧版本或不同的模式，应该有更明确的版本兼容性处理。

**建议修复**:
移除 `getattr`，直接访问属性（因为 Pydantic 模型有默认值）：
```python
allow_test = state.allow_test_labels_for_reporting
```

---

### 2.3 低严重程度 (Low)

#### BUG-008: 硬编码的警告消息未国际化

**文件位置**: 第 76-78, 107 行

**问题描述**:
警告消息是硬编码的英文字符串，没有考虑国际化需求。

**建议修复**:
考虑使用国际化框架或将消息提取到常量模块。

---

#### BUG-009: 函数返回类型注解可以更精确

**文件位置**: `src/agents/dataset_preparer_agent.py:88`

**代码片段**:
```python
def dataset_preparer_agent(state: PHMState, *, config: Dict | None = None) -> Dict:
```

**问题描述**:
返回类型 `Dict` 不够精确。可以使用 `TypedDict` 来明确定义返回的结构。

**建议修复**:
```python
from typing import TypedDict

class DatasetPreparerResult(TypedDict):
    datasets: Dict[str, Dict[str, Any]]
    n_nodes: int

def dataset_preparer_agent(state: PHMState, *, config: Dict | None = None) -> DatasetPreparerResult:
```

---

#### BUG-010: 未处理的边缘情况：空的 sample_ids 列表

**文件位置**: `src/agents/dataset_preparer_agent.py:59-60`

**代码片段**:
```python
sample_ids = list(labels_map.keys())
sample_id = sample_ids[0] if sample_ids else "sample"
```

**问题描述**:
当 `labels_map` 为空时，`sample_id` 被设置为硬编码的 `"sample"`。这个值可能不在任何地方定义，会导致 `labels_map.get(sample_id)` 返回 `None`，进而在第 63 行创建一个全 `None` 的标签数组。

**建议修复**:
```python
sample_ids = list(labels_map.keys())
if not sample_ids:
    return np.array([]), np.array([])
sample_id = sample_ids[0]
```

---

## 3. 统计汇总

| 严重程度 | 数量 | 占比 |
|----------|------|------|
| 高 (High) | 3 | 30% |
| 中 (Medium) | 4 | 40% |
| 低 (Low) | 3 | 30% |
| **总计** | **10** | **100%** |

## 4. 建议优先级

1. **立即修复**: BUG-001 (资源泄漏), BUG-002 (None 标签), BUG-003 (无限循环)
2. **近期修复**: BUG-004 (类型验证), BUG-005 (空数组处理), BUG-006 (日志记录)
3. **后续优化**: BUG-007 (属性访问), BUG-008 (国际化), BUG-009 (类型注解), BUG-010 (边缘情况)

## 5. 总体评估

`dataset_preparer_agent.py` 模块整体结构清晰，但在以下方面需要改进：

- **资源管理**: NPZ 文件需要显式关闭
- **健壮性**: 需要更好的输入验证和边缘情况处理
- **可维护性**: 应使用标准日志库而非 print
- **安全性**: DAG 遍历需要循环检测和深度限制

建议在修复高严重程度 BUG 后，添加单元测试以覆盖边缘情况。
