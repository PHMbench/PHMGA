# BUG 报告: utils 和 states 模块审查

**审查人**: Agent 8
**审查日期**: 2026-02-15
**审查范围**: `src/utils.py`, `src/states/phm_states.py`

---

## 1. 审查范围

| 文件 | 代码行数 | 主要功能 |
|------|----------|----------|
| `src/utils.py` | 377 | 状态加载/保存、信号数据处理、引用处理、辅助函数 |
| `src/states/phm_states.py` | 449 | 状态模型定义（PHMState、DAGState、节点类等） |

**总计**: 826 行代码

---

## 2. 高危 BUG

### BUG-01: pickle 反序列化安全漏洞（严重）

**文件位置**: `src/utils.py:362-375`

**代码片段**:
```python
def load_state(filepath: str):
    """
    使用pickle从磁盘加载状态对象。
    """
    try:
        print(f"\n--- Loading state from {filepath} ---")
        with open(filepath, "rb") as f:
            state = pickle.load(f)  # 安全漏洞：无验证地反序列化
        print("...done.")
        print(f"Successfully loaded state with {len(state.dag_state.nodes)} nodes.")
        return state
    except Exception as e:
        print(f"Error loading state: {e}")
        return None
```

**问题描述**:
1. `pickle.load()` 可以执行任意 Python 代码，这是已知的安全风险
2. 没有 HMAC 签名验证或任何完整性检查
3. 如果攻击者可以篡改保存的状态文件，可以在加载时执行任意代码
4. 在文档中明确提到了 `state_save_path` 功能，但没有安全警告

**建议修复**:
```python
import hmac
import hashlib

def load_state(filepath: str, signature_path: str | None = None, secret_key: str | None = None):
    """
    使用pickle从磁盘加载状态对象，支持可选的HMAC签名验证。

    Args:
        filepath: 状态文件路径
        signature_path: 可选的HMAC签名文件路径
        secret_key: 用于验证签名的密钥（应在环境变量中配置）

    警告: pickle可能执行任意代码，仅加载可信来源的状态文件！
    """
    try:
        print(f"\n--- Loading state from {filepath} ---")

        # 如果提供签名，先验证
        if signature_path and secret_key:
            with open(filepath, "rb") as f:
                data = f.read()

            with open(signature_path, "rb") as f:
                stored_sig = f.read()

            expected_sig = hmac.new(
                secret_key.encode(), data, hashlib.sha256
            ).digest()

            if not hmac.compare_digest(stored_sig, expected_sig):
                raise ValueError("State file signature verification failed!")

            from io import BytesIO
            state = pickle.loads(data)
        else:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

        print("...done.")
        print(f"Successfully loaded state with {len(state.dag_state.nodes)} nodes.")
        return state
    except Exception as e:
        print(f"Error loading state: {e}")
        return None
```

---

### BUG-02: HDF5 文件句柄泄漏（资源泄漏）

**文件位置**: `src/utils.py:207-249`

**代码片段**:
```python
def load_signal_data(metadata_path: str, h5_path: str, ids_to_load: list[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """..."""
    print(f"Loading data for IDs: {ids_to_load}")

    try:
        metadata_df = pd.read_excel(metadata_path)
        h5_file = h5py.File(h5_path, 'r')  # 打开文件
    except Exception as e:
        print(f"Error loading data files: {e}")
        return {}, {}

    signals = {}
    labels = {}
    for sample_id in ids_to_load:
        # ... 处理数据 ...

    h5_file.close()  # 如果循环中发生异常，不会执行到这里！
    return signals, labels
```

**问题描述**:
1. 如果在 `for sample_id in ids_to_load:` 循环中发生异常（如 `KeyError`、`IndexError`、数据处理错误等），`h5_file.close()` 不会被执行
2. 这会导致文件句柄泄漏，长时间运行可能导致系统资源耗尽
3. 类似的模式也出现在 `save_state()` 函数中的 `os.makedirs()` 和 `with open()` 组合

**建议修复**:
```python
def load_signal_data(metadata_path: str, h5_path: str, ids_to_load: list[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """..."""
    print(f"Loading data for IDs: {ids_to_load}")

    signals = {}
    labels = {}

    try:
        metadata_df = pd.read_excel(metadata_path)
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return {}, {}

    # 使用 with 语句确保文件句柄正确关闭
    try:
        with h5py.File(h5_path, 'r') as h5_file:
            for sample_id in ids_to_load:
                sample_info = metadata_df[metadata_df['Id'] == sample_id]
                if sample_info.empty:
                    print(f"Warning: ID {sample_id} not found in metadata.")
                    continue

                label = sample_info['Label'].iloc[0]
                sample_length = int(sample_info['Sample_lenth'].iloc[0])
                num_channels = int(sample_info['Channel'].iloc[0])

                try:
                    signal_data = h5_file[str(sample_id)][()]
                    signal_data = np.squeeze(signal_data)

                    if signal_data.shape == (sample_length, num_channels):
                        signals[str(sample_id)] = signal_data.reshape(1, sample_length, num_channels)
                        labels[str(sample_id)] = label
                    else:
                        print(f"Warning: Shape mismatch for ID {sample_id}. Expected {(sample_length, num_channels)}, got {signal_data.shape}")

                except KeyError:
                    print(f"Warning: ID {sample_id} not found in HDF5 file.")

    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        return {}, {}

    return signals, labels
```

---

### BUG-03: DAGState 节点反序列化类型混淆风险

**文件位置**: `src/states/phm_states.py:328-348`

**代码片段**:
```python
def load(self, path: str) -> None:
    """从指定路径加载 DAG 状态."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        self.state = DAGState(**data)  # 直接实例化，无验证
        self.g = nx.DiGraph()
        for n in self.state.nodes.values():
            if isinstance(n, dict):
                node_stage = n.get("stage", "processed")
                if node_stage == "input":
                    node_obj = InputData(**n)
                elif node_stage == "dataset":
                    node_obj = DataSetNode(**n)
                else:
                    node_obj = ProcessedData(**n)  # 默认 fallback
            else:
                node_obj = n
            self._add_node(node_obj)
```

**问题描述**:
1. 从 JSON 加载时，如果 `stage` 字段被篡改为非预期值（如 `"malicious"`），会默认创建 `ProcessedData`
2. `ProcessedData` 有一个必填字段 `source_signal_id`，但代码中没有检查
3. 如果数据被篡改，可能导致 `ValidationError` 或数据不一致
4. 对 `PHMOperator` 类型的处理缺失（虽然 `to_dot` 和 `_build_dot_source` 中检查了它）

**建议修复**:
```python
def load(self, path: str) -> None:
    """从指定路径加载 DAG 状态."""
    import json
    from ..tools.signal_processing_schemas import PHMOperator

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 验证必要字段
    if "nodes" not in data or "channels" not in data:
        raise ValueError("Invalid DAG state file: missing required fields")

    self.state = DAGState(**data)
    self.g = nx.DiGraph()

    for n in self.state.nodes.values():
        node_obj = None

        if isinstance(n, dict):
            node_stage = n.get("stage", "processed")

            # 添加类型验证
            if node_stage == "input":
                node_obj = InputData(**n)
            elif node_stage == "dataset":
                node_obj = DataSetNode(**n)
            elif node_stage == "processed":
                # 检查必填字段
                if "source_signal_id" not in n:
                    raise ValueError(f"ProcessedData node missing source_signal_id: {n.get('node_id', 'unknown')}")
                node_obj = ProcessedData(**n)
            elif node_stage == "operator":
                # 处理 PHMOperator 类型
                node_obj = PHMOperator(**n)
            else:
                raise ValueError(f"Unknown node stage: {node_stage}")
        else:
            node_obj = n

        self._add_node(node_obj)
```

---

## 3. 中危 BUG

### BUG-04: `initialize_state` 中缺少对 `fs` 的验证

**文件位置**: `src/utils.py:273-277`

**代码片段**:
```python
# --- 确定通道数 ---
# 从第一个加载的信号中推断出通道数
first_sig_array = next(iter(ref_signals.values()))
num_channels = first_sig_array.shape[2] # Shape is (B, L, C)
channel_names = [f"ch{i+1}" for i in range(num_channels)]
```

**问题描述**:
1. 如果 `ref_signals` 为空，`next(iter(ref_signals.values()))` 会抛出 `StopIteration`
2. 虽然之前有 `if not ref_signals or not test_signals:` 检查，但如果在检查后和调用前有其他问题，仍可能失败
3. 注释说 `Shape is (B, L, C)`，但没有验证形状是否至少是 3D

**建议修复**:
```python
# --- 确定通道数 ---
# 从第一个加载的信号中推断出通道数
if not ref_signals:
    raise ValueError("No reference signals available for channel inference")

first_sig_array = next(iter(ref_signals.values()))

# 验证形状维度
if len(first_sig_array.shape) < 3:
    raise ValueError(f"Expected 3D signal array (B, L, C), got shape: {first_sig_array.shape}")

num_channels = first_sig_array.shape[2]
if num_channels <= 0:
    raise ValueError(f"Invalid number of channels: {num_channels}")

channel_names = [f"ch{i+1}" for i in range(num_channels)]
```

---

### BUG-05: `get_node_data` 返回值类型不一致

**文件位置**: `src/states/phm_states.py:351-360`

**代码片段**:
```python
def get_node_data(state: "PHMState", node_id: str):
    """Utility to fetch raw array data from a node."""
    node = state.dag_state.nodes.get(node_id)
    if isinstance(node, InputData):
        return np.asarray(node.data.get("signal", []))
    if isinstance(node, ProcessedData):
        if isinstance(node.results, dict):
            return node.results  # 返回 dict
        return np.asarray(node.results) if node.results is not None else None  # 返回 ndarray
    return None  # 返回 None
```

**问题描述**:
1. 函数可能返回三种不同类型：`dict`、`ndarray`、`None`
2. 返回类型不一致会增加调用者的错误处理负担
3. 类型注解缺失（应添加 `-> Any | None` 或更具体的类型）
4. 没有处理 `DataSetNode` 类型

**建议修复**:
```python
def get_node_data(state: "PHMState", node_id: str) -> Any | None:
    """
    Utility to fetch raw array data from a node.

    Returns:
        For InputData: np.ndarray
        For ProcessedData: dict | np.ndarray | None
        For DataSetNode: dict | None
        Unknown/missing node: None
    """
    node = state.dag_state.nodes.get(node_id)
    if node is None:
        return None

    if isinstance(node, InputData):
        signal_data = node.data.get("signal", [])
        return np.asarray(signal_data) if signal_data is not None else np.array([])

    if isinstance(node, ProcessedData):
        if isinstance(node.results, dict):
            return node.results
        return np.asarray(node.results) if node.results is not None else None

    if isinstance(node, DataSetNode):
        # 假设 DataSetNode 在 meta 或其他字段中存储数据
        return node.meta if node.meta else None

    return None
```

---

### BUG-06: `dag_to_llm_payload` 缺少错误处理

**文件位置**: `src/utils.py:189-204`

**代码片段**:
```python
def dag_to_llm_payload(state: PHMState, max_nodes: int = 40) -> str:
    """Return a JSON string representing the latest portion of the DAG.

    Parameters
    ----------
    state : PHMState
        State whose internal DAG should be exported.
    max_nodes : int, optional
        Maximum number of nodes to include from the tail of the DAG.

    Returns
    -------
    str
        JSON payload for use in LLM prompts.
    """
    return state.tracker().export_json(max_nodes=max_nodes)
```

**问题描述**:
1. 没有验证 `max_nodes` 的合理性（负数、零、过大）
2. 如果 `state.tracker()` 失败（如 `DAGState` 损坏），会抛出未捕获的异常
3. `export_json` 内部的 `topological_sort` 可能在图有环时失败

**建议修复**:
```python
def dag_to_llm_payload(state: PHMState, max_nodes: int = 40) -> str:
    """Return a JSON string representing the latest portion of the DAG.

    Parameters
    ----------
    state : PHMState
        State whose internal DAG should be exported.
    max_nodes : int, optional
        Maximum number of nodes to include from the tail of the DAG.
        Must be positive and reasonable (1-1000).

    Returns
    -------
    str
        JSON payload for use in LLM prompts.
    """
    import json

    # 验证参数
    if not isinstance(max_nodes, int) or max_nodes <= 0:
        max_nodes = 40
    elif max_nodes > 1000:
        max_nodes = 1000

    try:
        tracker = state.tracker()
        return tracker.export_json(max_nodes=max_nodes)
    except Exception as e:
        # 返回空图结构而不是崩溃
        return json.dumps({"graph": [], "error": str(e)})
```

---

### BUG-07: `insert_citation_markers` 索引越界风险

**文件位置**: `src/utils.py:59-95`

**代码片段**:
```python
def insert_citation_markers(text, citations_list):
    """..."""
    # Sort citations by end_index in descending order...
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # ...
        end_idx = citation_info["end_index"]
        # ...
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text
```

**问题描述**:
1. 没有验证 `start_index` 和 `end_index` 是否在 `text` 的有效范围内
2. 如果 `end_index > len(text)`，Python 会接受（切片不会报错），但结果可能不符合预期
3. 如果 `start_index > end_index`，会产生空字符串插入点
4. 负索引可能导致意外的行为

**建议修复**:
```python
def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries...

    Returns:
        str: The text with citation markers inserted.

    Raises:
        ValueError: If citation indices are out of bounds.
    """
    if not citations_list:
        return text

    # 验证并过滤无效的引用
    valid_citations = []
    text_len = len(text)

    for citation in citations_list:
        start_idx = citation.get("start_index", 0)
        end_idx = citation.get("end_index", 0)

        # 验证索引
        if start_idx < 0:
            start_idx = max(0, text_len + start_idx)  # 处理负索引
        if end_idx < 0:
            end_idx = max(0, text_len + end_idx)

        # 边界检查
        if start_idx > text_len or end_idx > text_len:
            print(f"Warning: Citation indices out of bounds, skipping: start={start_idx}, end={end_idx}, len={text_len}")
            continue
        if start_idx > end_idx:
            print(f"Warning: start_index > end_index, skipping: start={start_idx}, end={end_idx}")
            continue

        valid_citations.append({
            **citation,
            "start_index": start_idx,
            "end_index": end_idx
        })

    # Sort by end_index descending
    sorted_citations = sorted(
        valid_citations,
        key=lambda c: (c["end_index"], c["start_index"]),
        reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info.get("segments", []):
            marker_to_insert += f" [{segment['label']}]({segment.get('short_url', '')})"
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text
```

---

## 4. 低危 BUG

### BUG-08: `save_state` 中的目录创建失败处理不完整

**文件位置**: `src/utils.py:347-360`

**代码片段**:
```python
def save_state(state, filepath: str):
    """..."""
    try:
        print(f"\n--- Saving state to {filepath} ---")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print("...done.")
        return True
    except Exception as e:
        print(f"Error saving state: {e}")
        return False
```

**问题描述**:
1. 如果 `filepath` 是相对路径（如 `state.pkl`），`os.path.dirname(filepath)` 返回空字符串
2. `os.makedirs("", exist_ok=True)` 不会创建任何目录，但文件可能写入失败
3. 没有检查父目录是否可写

**建议修复**:
```python
def save_state(state, filepath: str):
    """..."""
    try:
        print(f"\n--- Saving state to {filepath} ---")

        # 转换为绝对路径
        filepath = os.path.abspath(filepath)
        dir_path = os.path.dirname(filepath)

        if dir_path:  # 只有在目录路径非空时才创建
            os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print("...done.")
        return True
    except Exception as e:
        print(f"Error saving state: {e}")
        return False
```

---

### BUG-09: `get_research_topic` 缺少空值检查

**文件位置**: `src/utils.py:25-39`

**代码片段**:
```python
def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content  # 可能是 None
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"  # message.content 可能是 None
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"  # message.content 可能是 None
    return research_topic
```

**问题描述**:
1. 没有检查 `message.content` 是否为 `None`
2. 如果 `message.content` 是 `None`，会导致 `TypeError: can only concatenate str (not "NoneType") to str`
3. `messages` 可能为空列表，`messages[-1]` 会引发 `IndexError`

**建议修复**:
```python
def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    if not messages:
        return ""

    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        content = messages[-1].content
        research_topic = content if content is not None else ""
    else:
        research_topic = ""
        for message in messages:
            content = message.content if hasattr(message, 'content') else None
            content_str = content if content is not None else ""

            if isinstance(message, HumanMessage):
                research_topic += f"User: {content_str}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {content_str}\n"
    return research_topic
```

---

### BUG-10: `resolve_urls` 缺少异常处理

**文件位置**: `src/utils.py:42-56`

**代码片段**:
```python
def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloudshelf.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]  # 可能失败

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map
```

**问题描述**:
1. 列表推导式 `[site.web.uri for site in urls_to_resolve]` 可能因 `AttributeError` 失败
2. 没有验证 `id` 参数的类型
3. 使用 `id` 作为变量名会覆盖内置函数 `id()`

**建议修复**:
```python
def resolve_urls(urls_to_resolve: List[Any], url_id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.

    Args:
        urls_to_resolve: List of URL objects
        url_id: Unique identifier for this batch of URLs
    """
    prefix = "https://vertexaisearch.cloudshelf.google.com/id/"

    # 安全地提取 URI
    urls = []
    for site in urls_to_resolve:
        try:
            uri = site.web.uri
            if uri:  # 确保非空
                urls.append(uri)
        except AttributeError:
            continue

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{url_id}-{idx}"

    return resolved_map
```

---

### BUG-11: `_NodeBase.normalize_parents` 没有处理空字符串

**文件位置**: `src/states/phm_states.py:55-71`

**代码片段**:
```python
@field_validator("parents", mode="before")
@classmethod
def normalize_parents(cls, value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []  # 检查空字符串
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()  # strip 后可能为空字符串
            if text:  # 检查空字符串
                out.append(text)
        return out
    raise TypeError("parents must be a string or a list/tuple/set of strings")
```

**问题描述**:
1. 代码实际上已经正确处理了空字符串（`if text:`）
2. 但是没有验证父节点 ID 的格式（如长度限制、字符限制）
3. 没有检测循环引用（虽然后续 `add_node` 会检测）

**建议**:
当前实现基本正确，但可以添加格式验证：
```python
@field_validator("parents", mode="before")
@classmethod
def normalize_parents(cls, value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                # 可选：添加格式验证
                if len(text) > 256:
                    raise ValueError(f"Parent node ID too long: {len(text)} > 256")
                out.append(text)
        return out
    raise TypeError("parents must be a string or a list/tuple/set of strings")
```

---

## 5. 统计汇总

### 按严重程度分类

| 严重程度 | 数量 | BUG 编号 |
|----------|------|----------|
| 高危 | 3 | BUG-01, BUG-02, BUG-03 |
| 中危 | 4 | BUG-04, BUG-05, BUG-06, BUG-07 |
| 低危 | 4 | BUG-08, BUG-09, BUG-10, BUG-11 |

**总计**: 11 个 BUG

### 按问题类型分类

| 类型 | 数量 |
|------|------|
| 安全问题 | 1 (pickle 反序列化) |
| 资源管理 | 1 (文件句柄泄漏) |
| 输入验证 | 4 |
| 错误处理 | 3 |
| 类型安全 | 2 |

### 按文件分类

| 文件 | 高危 | 中危 | 低危 | 总计 |
|------|------|------|------|------|
| `src/utils.py` | 2 | 3 | 3 | 8 |
| `src/states/phm_states.py` | 1 | 1 | 1 | 3 |

---

## 6. 修复优先级建议

1. **立即修复**（高危）:
   - BUG-01: pickle 安全漏洞
   - BUG-02: HDF5 文件句柄泄漏
   - BUG-03: DAGState 反序列化风险

2. **尽快修复**（中危）:
   - BUG-04: `initialize_state` 验证不足
   - BUG-05: `get_node_data` 类型不一致
   - BUG-06: `dag_to_llm_payload` 错误处理
   - BUG-07: `insert_citation_markers` 边界检查

3. **后续改进**（低危）:
   - BUG-08: `save_state` 目录处理
   - BUG-09: `get_research_topic` 空值检查
   - BUG-10: `resolve_urls` 异常处理
   - BUG-11: `normalize_parents` 格式验证

---

**报告生成时间**: 2026-02-15
**审查人**: Agent 8
