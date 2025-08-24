# Part 2: Building Blocks - 构建模块 🏗️

## 🎯 学习目标

通过本部分，您将掌握：
- 📊 **状态管理** - TypedDict和Annotated类型的高级应用
- 🔄 **LangGraph工作流** - 复杂图结构的构建和执行
- 🔀 **Router模式** - 智能路由和负载均衡
- 🏭 **系统集成** - 将多个组件组合成完整系统

## 📋 预计时长：2-3小时

---

## 🏗️ 核心概念

### 1. 状态管理 (State Management)

状态是Graph Agent的核心，它在节点间传递数据和控制流：

#### TypedDict状态定义
```python
class PHMState(TypedDict):
    equipment_id: str
    sensor_data: Dict[str, float]
    analysis_history: Annotated[List[str], operator.add]  # 自动累加
    confidence: float
```

#### 关键特性
- **类型安全**：IDE友好，编译时检查
- **自动累加**：`Annotated[List, operator.add]` 实现列表自动合并
- **状态合并**：LangGraph自动处理状态更新和合并

### 2. LangGraph工作流 (Workflow Management)

LangGraph允许创建复杂的有向图工作流：

#### 基础工作流构建
```python
workflow = StateGraph(WorkflowState)
workflow.add_node("preprocess", preprocessing_node)
workflow.add_node("analyze", analysis_node)
workflow.add_edge("preprocess", "analyze")
workflow.set_entry_point("preprocess")
```

#### 高级特性
- **条件路由**：基于状态动态选择执行路径
- **并行处理**：多个节点同时执行
- **循环控制**：支持循环和迭代
- **错误处理**：优雅的异常处理和恢复

### 3. Router模式 (Routing Patterns)

Router负责智能地选择不同的处理路径：

#### 路由类型
1. **规则路由**：基于预定义规则的确定性路由
2. **LLM路由**：使用语言模型进行智能决策
3. **负载均衡**：分散处理负载，优化性能
4. **自适应路由**：根据反馈动态调整路由策略

---

## 🛠️ 模块实现

### 状态管理模块 (`modules/state_management.py`)

#### 核心功能
- **状态定义**：多种TypedDict状态结构
- **状态更新**：模拟LangGraph的状态合并逻辑
- **历史管理**：状态版本控制和回滚
- **快照功能**：状态备份和恢复

```python
# 使用示例
manager = StateManager()
state = create_phm_diagnosis_state("PUMP-001", sensor_data)
updated_state = manager.update_state({"diagnosis": "异常"})
```

### LangGraph工作流模块 (`modules/langgraph_workflows.py`)

#### 工作流类型
1. **基础工作流**：线性处理流程
2. **条件工作流**：基于状态的分支逻辑
3. **并行工作流**：多路径同时执行
4. **复合工作流**：嵌套和组合工作流

```python
# 使用示例
workflow = PHMAnalysisWorkflow()
result = workflow.analyze(sensor_data)
```

### Router模式模块 (`modules/router_patterns.py`)

#### Router实现
- **BaseRouter**：路由器抽象基类
- **RuleBasedRouter**：规则驱动路由
- **LLMBasedRouter**：AI驱动路由
- **LoadBalancingRouter**：负载均衡路由
- **AdaptiveRouter**：自适应路由

```python
# 使用示例
router = LLMBasedRouter("gemini")
analysis_type = router.route(request_data)
```

---

## 📊 架构模式

### 状态驱动架构
```
Input → State Creation → Node Processing → State Update → Output
         ↓
    State History ← State Management → State Validation
```

### 工作流编排模式
```
Entry Point → Preprocessing → Feature Extraction → Classification
                ↓                    ↓                 ↓
              Validation         Parallel Analysis   Decision
                ↓                    ↓                 ↓
              Routing            Result Fusion        End
```

### Router决策模式
```
Request → Router → Analysis Type → Workflow Selection → Execution
    ↓        ↓           ↓               ↓                ↓
  Context   Rules     Priority      Load Balance      Result
```

---

## 🧪 实践示例

### 示例1：PHM诊断工作流

```python
# 创建诊断状态
state = create_phm_diagnosis_state("MOTOR-001", {
    "temperature": 85,
    "vibration": 6.2,
    "pressure": 1.8
})

# 执行工作流
workflow = AdvancedPHMWorkflow()
result = workflow.run_analysis(state["sensor_readings"])
```

### 示例2：智能路由系统

```python
# 创建路由器
router = AdaptiveRouter()

# 路由决策
analysis_type = router.route({
    "sensor_data": {"temperature": 90, "vibration": 8.0},
    "priority": "high",
    "context": "maintenance_window"
})
```

### 示例3：状态管理

```python
# 状态管理器
manager = StateManager()

# 更新状态
manager.update_state({
    "analysis_history": ["开始分析", "特征提取完成"],
    "confidence": 0.85
})

# 获取快照
snapshot = manager.get_state_snapshot()
```

---

## 🎯 设计原则

### 1. 单一职责原则
- 每个节点专注于单一功能
- Router只负责路由决策
- 状态管理器只处理状态操作

### 2. 开闭原则
- 易于扩展新的节点类型
- 支持自定义Router策略
- 可插拔的状态管理

### 3. 依赖倒置原则
- 基于抽象接口编程
- 依赖注入配置
- 松散耦合的组件设计

### 4. 组合优于继承
- 通过组合构建复杂工作流
- Router策略组合
- 状态管理器组合

---

## 📈 性能优化

### 状态管理优化
- **浅拷贝**：避免不必要的深拷贝
- **增量更新**：只更新变化的部分
- **内存管理**：及时清理历史状态

### 工作流优化
- **并行执行**：独立节点并行处理
- **缓存机制**：中间结果缓存
- **懒加载**：按需加载节点

### Router优化
- **决策缓存**：相似请求复用决策
- **负载预测**：基于历史数据预测负载
- **动态调整**：实时调整路由权重

---

## 🧪 实践练习

### 练习1：自定义状态结构
创建适合您业务场景的状态结构，包含必要的字段和类型注解。

### 练习2：设计工作流
设计一个包含至少5个节点的复杂工作流，实现特定的业务逻辑。

### 练习3：实现Router策略
实现一个基于机器学习的Router，根据历史数据学习最优路由策略。

### 练习4：性能测试
对不同的状态管理和工作流策略进行性能测试，找出最优配置。

---

## 🎓 学习检查点

完成本部分后，您应该能够：
- [ ] 设计复杂的TypedDict状态结构
- [ ] 创建多节点的LangGraph工作流
- [ ] 实现条件路由和并行处理
- [ ] 选择合适的Router策略
- [ ] 优化系统性能和资源使用
- [ ] 集成多个组件构建完整系统

---

## 🔗 相关资源

- [LangGraph官方文档](https://langchain-ai.github.io/langgraph/)
- [TypedDict文档](https://docs.python.org/3/library/typing.html#typing.TypedDict)
- [设计模式：策略模式](https://refactoring.guru/design-patterns/strategy)

## ➡️ 下一步

完成Part 2后，继续学习：
- **Part 3: Agent Architectures** - ReAct模式和多Agent系统
- **Part 4: PHM Integration** - 真实组件集成和生产部署