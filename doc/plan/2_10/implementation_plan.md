# PHMGA Bug 修复实施计划

**创建日期**: 2026-02-16
**状态**: 待执行
**优先级**: 高

---

## 修复优先级概览

| 阶段 | 问题数量 | 预计工时 | 风险等级 |
|------|----------|----------|----------|
| 阶段1: 安全与资源泄漏 | 5 | 2-3天 | 高 |
| 阶段2: 无限循环与迭代控制 | 4 | 1-2天 | 高 |
| 阶段3: 输入验证与错误处理 | 7 | 2-3天 | 中 |
| 阶段4: 配置与类型安全 | 4 | 1-2天 | 中 |

---

## 阶段 1: 安全与资源泄漏修复 (高优先级)

### 1.1 Pickle 安全漏洞 (utils.py:362-375)

**文件**: `src/utils.py`

**问题描述**:
- `pickle.load()` 可执行任意代码，无完整性验证
- 如果攻击者篡改状态文件，可执行恶意代码

**修复方案**:
```python
def load_state(filepath: str, verify_signature: bool = True) -> PHMState | None:
    """
    使用pickle从磁盘加载状态对象。

    警告: pickle可能执行任意代码，仅加载可信来源的状态文件！

    Args:
        filepath: 状态文件路径
        verify_signature: 是否验证HMAC签名（需要环境变量STATE_SECRET_KEY）
    """
    import hmac
    import hashlib

    try:
        print(f"\n--- Loading state from {filepath} ---")

        # 可选的签名验证
        if verify_signature:
            secret_key = os.getenv("STATE_SECRET_KEY")
            if secret_key:
                sig_path = filepath + ".sig"
                if os.path.exists(sig_path):
                    with open(filepath, "rb") as f:
                        data = f.read()
                    with open(sig_path, "rb") as f:
                        stored_sig = f.read()

                    expected_sig = hmac.new(
                        secret_key.encode(), data, hashlib.sha256
                    ).digest()

                    if not hmac.compare_digest(stored_sig, expected_sig):
                        raise ValueError("State file signature verification failed!")

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        print("...done.")
        print(f"Successfully loaded state with {len(state.dag_state.nodes)} nodes.")
        return state
    except Exception as e:
        print(f"Error loading state: {e}")
        return None
```

### 1.2 HDF5 文件句柄泄漏 (utils.py:207-249)

**文件**: `src/utils.py`

**问题描述**:
- 异常时文件句柄未关闭
- 可能导致资源耗尽

**修复方案**:
```python
def load_signal_data(metadata_path: str, h5_path: str, ids_to_load: list[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """从真实的 metadata 和 HDF5 文件中加载信号数据和标签。"""
    print(f"Loading data for IDs: {ids_to_load}")

    try:
        metadata_df = pd.read_excel(metadata_path)
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return {}, {}

    signals = {}
    labels = {}

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

### 1.3 DAGState 反序列化类型混淆 (phm_states.py:328-348)

**文件**: `src/states/phm_states.py`

**问题描述**:
- 从 JSON 加载时，stage 字段被篡改可能导致类型错误
- 缺少对必填字段的验证

**修复方案**:
```python
def load(self, path: str) -> None:
    """从指定路径加载 DAG 状态."""
    import json

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
                from ..tools.signal_processing_schemas import PHMOperator
                node_obj = PHMOperator(**n)
            else:
                raise ValueError(f"Unknown node stage: {node_stage}")
        else:
            node_obj = n

        self._add_node(node_obj)

    # 重新计算 leaves
    self.state.leaves = [nid for nid in self.g.nodes() if self.g.out_degree(nid) == 0]
```

### 1.4 NaN 传播问题 (inquirer_agent.py:13-15)

**文件**: `src/agents/inquirer_agent.py`

**问题描述**:
- 常数数组导致 `np.corrcoef` 返回 NaN
- NaN 传播到相似度矩阵

**修复方案**:
```python
def _calc_metric(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if metric == "pearson":
        # 检查标准差，避免常数数组导致 NaN
        std_a = np.std(a)
        std_b = np.std(b)
        if std_a < 1e-12 or std_b < 1e-12:
            # 常数数组，相似度为 0
            return 1.0  # 1 - correlation，常数的相关性无意义
        r = np.corrcoef(a, b)[0, 1]
        if np.isnan(r):
            return 1.0
        return float(1 - r)
    raise ValueError(f"unknown metric {metric}")
```

### 1.5 Cosine 距离除零风险 (multi_schemas.py:95)

**文件**: `src/tools/multi_schemas.py`

**问题描述**:
- 零向量时除零产生 NaN

**修复方案**:
```python
elif self.metric == "cosine":
    # Returns cosine distance, not similarity
    norm1 = np.linalg.norm(vec1, axis=-1)
    norm2 = np.linalg.norm(vec2, axis=-1)
    denom = norm1 * norm2
    # 使用 np.divide 安全处理除零
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.divide(np.sum(vec1 * vec2, axis=-1), denom)
        similarity = np.nan_to_num(similarity, nan=0.0)
    return 1 - similarity
```

---

## 阶段 2: 无限循环与迭代控制修复

### 2.1 无限循环风险 (phm_outer_graph.py:121)

**文件**: `src/phm_outer_graph.py`

**问题描述**:
- 循环条件依赖 `needs_revision`，无最大迭代限制

**修复方案**:
```python
def build_builder_graph() -> Any:
    """构建负责迭代构建计算 DAG 的图。"""
    if not _LANGGRAPH_OK:  # pragma: no cover
        return _FallbackGraph(
            [
                ("plan", plan_agent),
                ("execute", execute_agent),
                ("reflect", lambda state: reflect_agent_node(state, stage="POST_EXECUTE")),
            ]
        )

    builder = StateGraph(PHMState)

    builder.add_node("plan", lambda state: _run_node("plan", plan_agent, state))
    builder.add_node("execute", lambda state: _run_node("execute", execute_agent, state))
    builder.add_node(
        "reflect", lambda state: _run_node("reflect", lambda s: reflect_agent_node(s, stage="POST_EXECUTE"), state)
    )

    builder.set_entry_point("plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")

    # 添加迭代限制，防止无限循环
    def should_continue(state: PHMState) -> str:
        max_iterations = getattr(state, "max_research_loops", 10)
        if state.iteration_count >= max_iterations:
            return END
        return "plan" if state.needs_revision else END

    builder.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "plan": "plan",
            END: END,
        },
    )

    return builder.compile()
```

### 2.2 迭代计数未更新 (reflect_agent.py:142-158)

**文件**: `src/agents/reflect_agent.py`

**问题描述**:
- `iteration_count` 永远不增加
- 迭代限制失效

**修复方案**:
```python
def reflect_agent_node(state: PHMState, *, stage: str) -> Dict[str, Any]:
    """Adapter for the outer graph using :class:`PHMState`."""
    try:
        dag_blueprint = json.loads(state.tracker().export_json())
    except Exception:
        dag_blueprint = {}
    issues = "\n".join(state.dag_state.error_log)
    result = reflect_agent(
        instruction=state.user_instruction,
        stage=stage,
        dag_blueprint=dag_blueprint,
        issues_summary=issues or None,
        state=state,
    )
    needs_revision = result["decision"] != "finish"
    history = state.reflection_history + [result["reason"]]

    # 更新迭代计数
    new_count = state.iteration_count + (1 if needs_revision else 0)

    return {
        "needs_revision": needs_revision,
        "reflection_history": history,
        "iteration_count": new_count
    }
```

### 2.3 Null dag_state 访问崩溃 (phm_outer_graph.py:169-188)

**文件**: `src/phm_outer_graph.py`

**问题描述**:
- `state.dag_state` 为 None 时崩溃

**修复方案**:
```python
def _init_dag_for_tspn(state: PHMState) -> dict:
    backend = (getattr(state, "train_backend", None) or "shallow").lower()
    if backend not in {"tspn", "both"}:
        return {}

    # 安全访问 dag_state.nodes
    nodes = getattr(getattr(state, "dag_state", None), "nodes", None) or {}

    # If builder already produced processed nodes, no need to re-init.
    if any(getattr(n, "method", None) for n in nodes.values()):
        return {}

    return dag_init_agent(state)
```

### 2.4 API Key 验证 (model/__init__.py:239-245)

**文件**: `src/model/__init__.py`

**问题描述**:
- API Key 为 None 时静默失败
- 运行时才报错，调试困难

**修复方案**:
```python
# --- Gemini (optional dependency) ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "langchain_google_genai is required for real LLM calls. "
        "Install compatible versions of langchain/langchain_google_genai, "
        "or set FAKE_LLM=true to run offline."
    ) from e

api_key = os.getenv("GEMINI_API_KEY")

# 验证 API Key
if not api_key and not fake_llm:
    raise ValueError(
        "Missing GEMINI_API_KEY environment variable. "
        "Set it or use FAKE_LLM=true for testing."
    )

return ChatGoogleGenerativeAI(
    model=model_name,
    temperature=temperature,
    max_retries=max_retries,
    api_key=api_key,
)
```

---

## 阶段 3: 输入验证与错误处理

### 3.1 validate_fs 参数 (utils.py:273-277)

```python
# --- 确定通道数 ---
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

### 3.2 get_node_data 类型一致性 (phm_states.py:351-360)

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
        return node.meta if node.meta else None

    return None
```

### 3.3 dag_to_llm_payload 错误处理 (utils.py:189-204)

```python
def dag_to_llm_payload(state: PHMState, max_nodes: int = 40) -> str:
    """Return a JSON string representing the latest portion of the DAG."""
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

### 3.4 insert_citation_markers 边界检查 (utils.py:59-95)

```python
def insert_citation_markers(text, citations_list):
    """Inserts citation markers into a text string based on start and end indices."""
    if not citations_list:
        return text

    # 验证并过滤无效的引用
    valid_citations = []
    text_len = len(text)

    for citation in citations_list:
        start_idx = citation.get("start_index", 0)
        end_idx = citation.get("end_index", 0)

        # 处理负索引
        if start_idx < 0:
            start_idx = max(0, text_len + start_idx)
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

### 3.5 save_state 目录处理 (utils.py:347-360)

```python
def save_state(state, filepath: str):
    """使用pickle将状态对象保存到磁盘。"""
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

### 3.6 get_research_topic 空值检查 (utils.py:25-39)

```python
def get_research_topic(messages: List[AnyMessage]) -> str:
    """Get the research topic from the messages."""
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

### 3.7 resolve_urls 错误处理 (utils.py:42-56)

```python
def resolve_urls(urls_to_resolve: List[Any], url_id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls to a short url with a unique id for each url.

    Args:
        urls_to_resolve: List of URL objects
        url_id: Unique identifier for this batch of URLs
    """
    prefix = "https://vertexaisearch.cloud.google.com/id/"

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

## 阶段 4: 配置与类型安全

### 4.1 train_backend 参数验证 (phm_outer_graph.py:137-163)

```python
def _train_models(state: PHMState) -> dict:
    backend = (getattr(state, "train_backend", None) or "shallow").lower()

    # 验证 backend 参数
    valid_backends = {"shallow", "tspn", "both"}
    if backend not in valid_backends:
        raise ValueError(
            f"Invalid train_backend={backend!r}. "
            f"Must be one of: {', '.join(sorted(valid_backends))}"
        )

    ml_results: Dict[str, Any] = dict(getattr(state, "ml_results", {}) or {})

    if backend in {"shallow", "both"}:
        # ... existing code ...
```

### 4.2 类型注解修复 (configuration.py:92-110)

```python
from typing import Any, Dict, Optional

@classmethod
def from_runnable_config(
    cls, config: Optional[Dict[str, Any]] = None
) -> "Configuration":
    """Create a Configuration instance from a config dict."""
    configurable = (
        config.get("configurable", {}) if config else {}
    )
    # ... existing code ...
```

### 4.3 train_backend 字符串方法安全 (phm_outer_graph.py:187-188)

```python
def _executor_path(state: PHMState) -> str:
    logger = get_current_logger()

    # 安全获取 backend
    backend_attr = getattr(state, "train_backend", None) or "shallow"
    if not isinstance(backend_attr, str):
        backend = "shallow"
    else:
        backend = backend_attr.strip().lower()

    path = "tspn_fast_path" if backend == "tspn" else "full_path"
    # ... existing code ...
```

---

## 测试验证计划

### 单元测试

1. **测试 HDF5 文件句柄正确关闭**
   ```python
   def test_h5_file_closed_on_error():
       # 模拟加载过程中出错
       # 验证文件句柄被正确关闭
   ```

2. **测试迭代计数限制**
   ```python
   def test_max_iterations_enforced():
       # 模拟 needs_revision 持续为 True
       # 验证在 max_iterations 后停止
   ```

3. **测试 NaN 处理**
   ```python
   def test_pearson_constant_array():
       # 测试常数数组的处理
       assert _calc_metric(np.ones(100), np.ones(100), "pearson") == 1.0
   ```

### 集成测试

1. **完整工作流测试**
   - 运行一个完整的 case
   - 验证所有节点正确执行
   - 验证资源正确释放

2. **错误恢复测试**
   - 模拟 LLM 返回无效 JSON
   - 验证工作流正确处理

---

## 执行顺序

1. **阶段1** - 安全与资源泄漏（必须优先完成）
2. **阶段2** - 无限循环与迭代控制（阻止系统挂起）
3. **阶段3** - 输入验证与错误处理（提高稳定性）
4. **阶段4** - 配置与类型安全（代码质量）

每个阶段完成后，运行相关测试验证修复效果。
