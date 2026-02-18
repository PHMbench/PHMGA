# Bug 修复快速参考

**更新日期**: 2026-02-16

---

## TOP 10 关键 Bug 快速修复指南

### 1. Pickle 安全漏洞 (src/utils.py:362-375)

**一行修复**: 添加警告注释
```python
def load_state(filepath: str):
    """使用pickle从磁盘加载状态对象。

    警告: pickle可能执行任意代码，仅加载可信来源的状态文件！
    """
```

**完整修复**: 添加可选的 HMAC 签名验证

---

### 2. HDF5 文件句柄泄漏 (src/utils.py:207-249)

**修复**: 使用 `with` 语句
```python
# 修改前
h5_file = h5py.File(h5_path, 'r')
# ... 处理 ...
h5_file.close()

# 修改后
with h5py.File(h5_path, 'r') as h5_file:
    # ... 处理 ...
```

---

### 3. 无限循环风险 (src/phm_outer_graph.py:121)

**修复**: 添加迭代限制
```python
# 修改条件边函数
def should_continue(state: PHMState) -> str:
    max_iterations = getattr(state, "max_research_loops", 10)
    if state.iteration_count >= max_iterations:
        return END
    return "plan" if state.needs_revision else END

builder.add_conditional_edges("reflect", should_continue, {...})
```

---

### 4. NaN 传播 (src/agents/inquirer_agent.py:13-15)

**修复**: 添加标准差检查
```python
if metric == "pearson":
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-12 or std_b < 1e-12:
        return 1.0  # 常数数组，无相关性
    r = np.corrcoef(a, b)[0, 1]
    return float(1 - r) if not np.isnan(r) else 1.0
```

---

### 5. Cosine 距离除零 (src/tools/multi_schemas.py:95)

**修复**: 安全除法
```python
elif self.metric == "cosine":
    norm1 = np.linalg.norm(vec1, axis=-1)
    norm2 = np.linalg.norm(vec2, axis=-1)
    denom = norm1 * norm2
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.divide(np.sum(vec1 * vec2, axis=-1), denom)
        similarity = np.nan_to_num(similarity, nan=0.0)
    return 1 - similarity
```

---

### 6. 迭代计数未更新 (src/agents/reflect_agent.py:158)

**修复**: 添加计数更新
```python
# 在返回值中添加
new_count = state.iteration_count + (1 if needs_revision else 0)
return {
    "needs_revision": needs_revision,
    "reflection_history": history,
    "iteration_count": new_count  # 新增
}
```

---

### 7. API Key 验证 (src/model/__init__.py:239)

**修复**: 添加验证
```python
api_key = os.getenv("GEMINI_API_KEY")
if not api_key and not fake_llm:
    raise ValueError(
        "Missing GEMINI_API_KEY environment variable. "
        "Set it or use FAKE_LLM=true for testing."
    )
```

---

### 8. get_node_data 类型不一致 (src/states/phm_states.py:351)

**修复**: 添加类型注解和 None 检查
```python
def get_node_data(state: "PHMState", node_id: str) -> Any | None:
    node = state.dag_state.nodes.get(node_id)
    if node is None:
        return None
    # ... 其余代码
```

---

### 9. dag_to_llm_payload 错误处理 (src/utils.py:189)

**修复**: 添加 try-except
```python
def dag_to_llm_payload(state: PHMState, max_nodes: int = 40) -> str:
    import json
    try:
        tracker = state.tracker()
        return tracker.export_json(max_nodes=max_nodes)
    except Exception as e:
        return json.dumps({"graph": [], "error": str(e)})
```

---

### 10. insert_citation_markers 边界检查 (src/utils.py:59)

**修复**: 添加边界验证
```python
for citation in citations_list:
    start_idx = citation.get("start_index", 0)
    end_idx = citation.get("end_index", 0)

    # 边界检查
    if start_idx > text_len or end_idx > text_len:
        continue  # 跳过越界的引用
    if start_idx > end_idx:
        continue  # 跳过无效范围
```

---

## 修复检查清单

### 阶段 1: 安全与资源泄漏
- [ ] utils.py: load_state 添加安全警告
- [ ] utils.py: load_signal_data 使用 with 语句
- [ ] phm_states.py: load 添加字段验证
- [ ] inquirer_agent.py: _calc_metric 添加标准差检查
- [ ] multi_schemas.py: DistanceOp 安全除法

### 阶段 2: 无限循环控制
- [ ] phm_outer_graph.py: should_continue 函数
- [ ] reflect_agent.py: 迭代计数更新
- [ ] phm_outer_graph.py: 安全访问 dag_state
- [ ] model/__init__.py: API Key 验证

### 阶段 3: 输入验证
- [ ] utils.py: initialize_state 形状验证
- [ ] phm_states.py: get_node_data 类型注解
- [ ] utils.py: dag_to_llm_payload 错误处理
- [ ] utils.py: insert_citation_markers 边界检查
- [ ] utils.py: save_state 目录处理
- [ ] utils.py: get_research_topic 空值检查
- [ ] utils.py: resolve_urls 错误处理

### 阶段 4: 配置验证
- [ ] phm_outer_graph.py: train_backend 验证
- [ ] configuration.py: 类型注解修复
- [ ] phm_outer_graph.py: backend 字符串安全

---

## 测试命令

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_utils.py
pytest tests/test_phm_states.py
pytest tests/test_phm_outer_graph.py

# 运行端到端测试
pytest tests/test_end2end.py

# 运行特定用例
pytest tests/test_utils.py::test_load_signal_data -v
```

---

## 验证步骤

1. **修复前**: 备份当前代码
   ```bash
   git stash push -m "Before bug fixes"
   ```

2. **修复**: 按阶段依次修复

3. **测试**: 每个阶段完成后运行测试
   ```bash
   pytest tests/ -v
   ```

4. **验证**: 运行一个完整 case
   ```bash
   python main.py case1
   ```

5. **提交**: 分阶段提交
   ```bash
   git commit -m "fix(utils): add resource management improvements"
   ```
