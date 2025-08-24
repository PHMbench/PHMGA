# Part 1: Foundations - Graph Agent 基础入门

## 🎯 学习目标

通过本部分，您将掌握：
- 🤖 **理解Agent基本概念** - 什么是智能体和Graph Agent
- 🔧 **LLM Provider使用** - 多种LLM的统一调用
- 🕸️ **Graph Agent实现** - 创建简单的图结构智能体
- 📊 **基础实践** - 完成简单的PHM诊断案例

## 📋 预计时长：1.5-2小时

---

## 🏗️ 核心概念

### 1. Agent基础
- **感知-思考-行动循环**：传统Agent的工作模式
- **状态管理**：Agent如何维护内部状态
- **决策机制**：从规则驱动到智能决策

### 2. LLM Provider统一接口
- **多Provider支持**：Google Gemini, OpenAI GPT, 通义千问, 智谱GLM
- **配置管理**：API密钥和参数设置
- **错误处理**：网络异常和API限制处理

### 3. Graph Agent架构
- **图结构 vs 线性结构**：灵活的执行路径
- **状态传递**：节点间的数据流
- **条件路由**：基于状态的动态分支

---

## 🛠️ 实现细节

### Agent基础实现 (`modules/agent_basics.py`)
```python
class SimpleAgent:
    """基础Agent实现"""
    def process(self, input_data):
        # 感知 -> 思考 -> 行动
        perception = self.perceive(input_data)
        decision = self.think(perception)
        return self.act(decision)
```

### LLM Provider封装 (`modules/llm_providers_unified.py`)
```python
def create_llm(provider="auto", **kwargs):
    """统一的LLM创建接口"""
    if provider == "google":
        return ChatGoogleGenerativeAI(**kwargs)
    elif provider == "openai":
        return ChatOpenAI(**kwargs)
    # ... 其他Provider
```

### Graph Agent示例 (`modules/graph_agent_intro.py`)
```python
class SimpleGraphAgent:
    """图结构Agent实现"""
    def __init__(self):
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        # 使用LangGraph构建图结构
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_node)
        # ...
        return workflow.compile()
```

---

## 📚 关键概念详解

### Agent vs 传统程序
| 特性 | 传统程序 | Agent |
|------|----------|-------|
| 执行模式 | 顺序执行 | 感知-决策-行动循环 |
| 状态管理 | 静态变量 | 动态状态更新 |
| 交互能力 | 有限 | 持续交互 |
| 适应性 | 固定逻辑 | 可学习和适应 |

### Graph Agent优势
1. **灵活路由**：根据状态动态选择执行路径
2. **并行处理**：多个节点可同时执行
3. **状态共享**：节点间共享和传递状态
4. **可视化**：清晰的图结构便于理解和调试

### LLM集成的价值
- **智能决策**：基于自然语言的推理
- **知识整合**：利用预训练知识
- **灵活交互**：自然语言接口
- **快速原型**：无需复杂规则编写

---

## 🧪 实践练习

### 练习1：创建简单Agent
修改 `SimpleAgent` 类，添加您自己的业务逻辑：
```python
class MyCustomAgent(SimpleAgent):
    def think(self, perception):
        # 添加您的决策逻辑
        pass
```

### 练习2：LLM Provider切换
尝试使用不同的LLM Provider解决同一个问题，比较它们的响应差异。

### 练习3：Graph工作流设计
设计一个包含3-5个节点的图结构工作流，实现特定的诊断逻辑。

---

## 🎓 学习检查点

完成本部分后，您应该能够：
- [ ] 解释Agent的基本工作原理
- [ ] 配置和使用至少一种LLM Provider
- [ ] 创建简单的Graph Agent
- [ ] 理解状态管理的基本概念
- [ ] 实现基础的PHM诊断逻辑

---

## 🔗 相关资源

- [LangChain官方文档](https://python.langchain.com/)
- [LangGraph教程](https://langchain-ai.github.io/langgraph/)
- [Agent设计模式](https://docs.langchain.com/docs/use-cases/agents)

## ➡️ 下一步

完成Part 1后，继续学习：
- **Part 2: Building Blocks** - 状态管理和复杂工作流
- **Part 3: Agent Architectures** - ReAct模式和多Agent系统