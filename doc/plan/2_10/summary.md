# PHMGA 代码仓库 BUG 审查 - 汇总报告

**审查日期**: 2026-02-16
**审查方式**: 10 个并行 Agent Team 成员分工审查
**报告路径**: `doc/plan/2_10/`

---

## 审查团队分工

| Agent ID | 审查模块 | 报告文件 |
|----------|----------|----------|
| Agent 1 | plan_agent | agent_01_plan_agent.md |
| Agent 2 | execute_agent | agent_02_execute_agent.md |
| Agent 3 | reflect_agent | agent_03_reflect_agent.md |
| Agent 4 | dataset_preparer | agent_04_dataset_preparer.md |
| Agent 5 | shallow_ml | agent_05_shallow_ml.md |
| Agent 6 | report_agent | agent_06_report_agent.md |
| Agent 7 | inquirer_agent | agent_07_inquirer_agent.md |
| Agent 8 | utils & states | agent_08_utils_states.md |
| Agent 9 | tools & schemas | agent_09_tools_schemas.md |
| Agent 10 | graph & config | agent_10_graph_config.md |

---

## 全局统计

| 严重程度 | 数量 | 百分比 |
|----------|------|--------|
| **高 (High)** | 23 | 27.4% |
| **中 (Medium)** | 41 | 48.8% |
| **低 (Low)** | 20 | 23.8% |
| **总计** | **84** | 100% |

### 按模块统计

| 模块 | 高 | 中 | 低 | 总计 |
|------|----|----|----|----|
| plan_agent | 3 | 5 | 6 | 14 |
| execute_agent | 3 | 4 | 4 | 11 |
| reflect_agent | 3 | 4 | 3 | 10 |
| dataset_preparer | 3 | 4 | 3 | 10 |
| shallow_ml | 3 | 4 | 2 | 9 |
| report_agent | 2 | 4 | 4 | 10 |
| inquirer_agent | 3 | 5 | 4 | 12 |
| utils_states | 3 | 4 | 4 | 11 |
| tools_schemas | 4 | 10 | 6 | 20 |
| graph_config | 4 | 6 | 4 | 14 |

---

## 按问题类型分类

| 问题类型 | 数量 |
|----------|------|
| 错误处理 | 28 |
| 输入验证 | 22 |
| 资源管理 | 8 |
| 数值稳定性 | 10 |
| 类型安全 | 8 |
| 逻辑缺陷 | 8 |

---

## 高优先级 BUG 摘要 (立即修复)

### 安全问题
1. **pickle 反序列化漏洞** (`src/utils.py:369`) - 可执行任意代码
2. **H5 文件句柄泄漏** (`src/utils.py:218`) - 资源泄漏
3. **API 密钥未验证** (`src/model.py:41`) - 运行时错误

### 数值稳定性
4. **cosine 距离 NaN** (`src/tools/multi_schemas.py:95`) - 零向量除零
5. **除零风险** (`src/tools/aggregate_schemas.py:129,142,155`) - 硬编码 epsilon
6. **pearson 相关 NaN** (`src/agents/inquirer_agent.py:13-15`) - 常数组导致 NaN

### 输入验证
7. **VMD/EMD 参数未验证** (`src/tools/expand_schemas.py:328`) - K 过大崩溃
8. **SavitzkyGolay 参数未验证** (`src/tools/transform_schemas.py:271`) - 约束检查缺失
9. **JSON 解析缺少验证** (`src/agents/plan_agent.py:134-141`) - LLM 响应错误

### 逻辑缺陷
10. **DAG 遍历无限循环** (`src/agents/dataset_preparer_agent.py:24-30`) - 无循环检测
11. **StopIteration 风险** (`src/agents/execute_agent.py:126,145`) - 空字典迭代
12. **WignerVille 索引错误** (`src/tools/expand_schemas.py:189`) - `tfr[n,n]` 应为 `tfr[n,n+tau]`

---

## 中优先级 BUG (尽快修复)

### 错误处理改进
- 裸露的 `except Exception` 捕获 (多处)
- 异常被静默吞没 (reflect_agent, inquirer_agent)
- JSON 序列化未处理不可序列化对象

### 输入验证加强
- 训练集/测试集维度一致性检查缺失 (shallow_ml_agent)
- fs 参数类型问题 (execute_agent:28)
- 节点存在性检查缺失 (comparator_tool.py)

### 资源管理
- NPZ 文件未正确关闭 (dataset_preparer_agent)
- 目录创建失败处理不完整 (utils.py)

---

## 低优先级问题 (后续改进)

- 未使用的导入和死代码
- 类型注解缺失
- 注释掉的代码和 TODO
- 性能优化建议 (apply_along_axis)
- 代码风格不一致

---

## 修复建议优先级

### 第一阶段 (立即修复)
1. 修复 pickle 安全漏洞 - 添加签名验证
2. 修复 H5 文件资源泄漏 - 使用 with 语句
3. 修复 cosine 距离 NaN - 添加零向量检查
4. 修复 VMD/EMD 输入验证 - 添加参数范围检查
5. 修复 JSON 解析验证 - 添加结构检查

### 第二阶段 (本周内)
1. 改进错误处理 - 使用特定异常类型
2. 添加输入验证 - 维度、参数范围
3. 修复除零问题 - 使用相对 epsilon
4. 添加 DAG 循环检测

### 第三阶段 (下个迭代)
1. 清理死代码和注释
2. 统一类型注解
3. 性能优化
4. 改进错误消息

---

## 审查覆盖率

| 指标 | 数值 |
|------|------|
| 审查文件数 | 29 |
| 审查代码行数 | ~4500 |
| 发现问题总数 | 84 |
| 高严重问题比例 | 27.4% |

---

## 下一步行动

1. **创建修复任务**: 根据优先级创建 Issue
2. **分配责任人**: 每个 BUG 分配给开发者
3. **设置里程碑**: 按阶段规划修复时间
4. **添加测试**: 针对每个 BUG 添加回归测试
5. **代码审查**: 修复后进行交叉审查

---

**报告生成**: 2026-02-16
**审查团队**: bug-review-team (10 agents)
