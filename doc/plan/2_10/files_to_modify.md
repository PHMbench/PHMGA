# Bug 修复文件清单

**更新日期**: 2026-02-16

---

## 需要修改的文件列表

### 高优先级文件 (安全/稳定性)

| 文件路径 | 修改行数 | Bug 数量 | 风险等级 |
|---------|----------|----------|----------|
| `src/utils.py` | ~50 | 7 | 高 |
| `src/phm_outer_graph.py` | ~30 | 4 | 高 |
| `src/agents/reflect_agent.py` | ~5 | 1 | 中 |
| `src/agents/inquirer_agent.py` | ~10 | 1 | 中 |
| `src/tools/multi_schemas.py` | ~8 | 1 | 中 |
| `src/model/__init__.py` | ~5 | 1 | 中 |
| `src/states/phm_states.py` | ~20 | 3 | 中 |
| `src/configuration.py` | ~3 | 1 | 低 |

---

## 详细修改清单

### 1. src/utils.py

**修改点 1**: `load_state` 函数 (362-375)
- 添加安全警告注释
- 添加可选 HMAC 签名验证
- 添加返回类型注解

**修改点 2**: `load_signal_data` 函数 (207-249)
- 使用 `with h5py.File()` 上下文管理器
- 分离 metadata 加载和 h5 加载的 try-except

**修改点 3**: `initialize_state` 函数 (273-277)
- 添加空 signals 检查
- 添加形状验证
- 添加通道数验证

**修改点 4**: `dag_to_llm_payload` 函数 (189-204)
- 添加 max_nodes 参数验证
- 添加 try-except 错误处理

**修改点 5**: `insert_citation_markers` 函数 (59-95)
- 添加索引边界检查
- 添加负索引处理
- 添加空索引检查

**修改点 6**: `save_state` 函数 (347-360)
- 使用绝对路径
- 添加目录存在检查

**修改点 7**: `get_research_topic` 函数 (25-39)
- 添加空 messages 检查
- 添加 content None 检查

**修改点 8**: `resolve_urls` 函数 (42-56)
- 添加 AttributeError 处理
- 重命名参数 id 为 url_id

---

### 2. src/phm_outer_graph.py

**修改点 1**: `build_builder_graph` 函数 (89-128)
- 添加 `should_continue` 内部函数
- 使用 `should_continue` 替换 lambda
- 添加迭代限制检查

**修改点 2**: `_train_models` 函数 (137-163)
- 添加 train_backend 参数验证
- 使用有效值集合检查

**修改点 3**: `_init_dag_for_tspn` 函数 (164-171)
- 添加安全访问 dag_state.nodes
- 使用 getattr 链式调用

**修改点 4**: `_executor_path` 函数 (186-199)
- 添加 backend 类型检查
- 安全调用 .strip() 和 .lower()

---

### 3. src/agents/reflect_agent.py

**修改点 1**: `reflect_agent_node` 函数 (142-158)
- 添加 iteration_count 更新逻辑
- 返回值中包含 `iteration_count`

---

### 4. src/agents/inquirer_agent.py

**修改点 1**: `_calc_metric` 函数 (7-16)
- 添加 cosine 分母阈值检查
- 添加 pearson 标准差检查
- 添加 NaN 处理

---

### 5. src/tools/multi_schemas.py

**修改点 1**: `DistanceOp.execute` 函数 (82-97)
- 添加 np.errstate 上下文
- 使用 np.divide 安全除法
- 添加 np.nan_to_num 处理

---

### 6. src/model/__init__.py

**修改点 1**: `get_llm` 函数 Gemini 分支 (239-245)
- 添加 api_key None 检查
- 添加明确的错误提示

---

### 7. src/states/phm_states.py

**修改点 1**: `DAGTracker.load` 函数 (328-348)
- 添加必要字段验证
- 添加 source_signal_id 检查
- 添加 unknown stage 错误

**修改点 2**: `get_node_data` 函数 (351-360)
- 添加返回类型注解
- 添加 node None 检查
- 添加 DataSetNode 处理

**修改点 3**: `normalize_parents` 验证器 (55-71)
- 添加格式验证 (可选)

---

### 8. src/configuration.py

**修改点 1**: `from_runnable_config` 函数 (92-110)
- 修改类型注解为 Dict[str, Any]
- 使用 .get() 方法访问

---

## 新增文件 (可选)

### 1. tests/test_bug_fixes.py
```python
"""测试 bug 修复的单元测试"""
# - test_load_state_security
# - test_h5_resource_cleanup
# - test_iteration_limit
# - test_nan_handling
# - test_api_key_validation
```

### 2. scripts/verify_fixes.py
```python
"""验证所有 bug 修复的脚本"""
# 检查关键函数是否存在
# 验证参数验证逻辑
# 运行回归测试
```

---

## 修改统计

| 类别 | 文件数 | 总修改行数 |
|------|--------|-----------|
| 安全修复 | 2 | ~60 |
| 资源管理 | 1 | ~15 |
| 循环控制 | 2 | ~35 |
| 输入验证 | 4 | ~40 |
| 类型安全 | 3 | ~15 |
| **总计** | **8** | **~165** |

---

## Git 提交建议

```bash
# 阶段 1: 安全与资源
git commit -m "fix(security): add pickle warning and HMAC verification option
fix(utils): use context manager for HDF5 file handling
fix(states): add validation for DAG deserialization"

# 阶段 2: 循环控制
git commit -m "fix(graph): add iteration limit to prevent infinite loops
fix(reflect): update iteration_count in reflect_agent_node
fix(model): add API key validation for Gemini"

# 阶段 3: 输入验证
git commit -m "fix(utils): add input validation to utility functions
fix(agents): add NaN handling in similarity calculations"

# 阶段 4: 类型安全
git commit -m "refactor(graph): add type annotations and validation
fix(config): correct type hints in Configuration"
```
