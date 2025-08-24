# PHM Graph Agent System Tutorial

## 从基础到高级：构建智能PHM诊断系统

本教程将带您从认识Agent概念开始，逐步深入到构建复杂的PHM Graph Agent系统。

---

## 第一部分：认识Agent - 智能体基础

### 1.1 什么是Agent？

Agent（智能体）是一个能够感知环境、做出决策并执行行动的自主实体。在AI系统中，Agent通过以下特征实现智能行为：

- **自主性**：能够独立做出决策
- **反应性**：响应环境变化
- **主动性**：主动达成目标
- **社交性**：与其他Agent协作

### 1.2 基础Agent示例

```python
# 示例1：最简单的Agent
class SimpleAgent:
    """一个简单的问答Agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.memory = []
    
    def perceive(self, input_text: str):
        """感知输入"""
        self.memory.append(input_text)
        return input_text
    
    def think(self, perception: str):
        """思考并决策"""
        if "hello" in perception.lower():
            return "greeting"
        elif "?" in perception:
            return "question"
        else:
            return "statement"
    
    def act(self, decision: str):
        """执行行动"""
        actions = {
            "greeting": f"Hello! I'm {self.name}. How can I help you?",
            "question": "That's an interesting question. Let me think...",
            "statement": "I understand. Please tell me more."
        }
        return actions.get(decision, "I'm processing your input...")
    
    def run(self, input_text: str) -> str:
        """完整的Agent循环"""
        perception = self.perceive(input_text)
        decision = self.think(perception)
        response = self.act(decision)
        return response

# 使用示例
agent = SimpleAgent("Assistant")
print(agent.run("Hello there!"))  # 输出: Hello! I'm Assistant. How can I help you?
print(agent.run("What is PHM?"))   # 输出: That's an interesting question. Let me think...
```

---

## 第二部分：使用LLM构建智能Agent

### 2.1 集成大语言模型

现代Agent通常使用LLM作为"大脑"来增强决策能力。

```python
# 示例2：使用不同LLM提供商的Agent

# 方式1：使用Google Gemini
import os
from src.model import get_llm_by_provider

# 设置API密钥
os.environ["GEMINI_API_KEY"] = "your_gemini_api_key"

class LLMAgent:
    """基于LLM的智能Agent"""
    
    def __init__(self, provider: str = "google", model: str = "gemini-2.5-flash"):
        self.llm = get_llm_by_provider(provider, model)
        self.conversation_history = []
    
    def process(self, user_input: str) -> str:
        """处理用户输入并生成响应"""
        # 构建提示词
        prompt = f"""
        You are a helpful assistant. 
        User: {user_input}
        Assistant:
        """
        
        # 调用LLM
        response = self.llm.invoke(prompt)
        
        # 保存对话历史
        self.conversation_history.append({
            "user": user_input,
            "assistant": response.content
        })
        
        return response.content

# 使用Google Gemini
gemini_agent = LLMAgent(provider="google", model="gemini-2.5-flash")
response = gemini_agent.process("Explain what PHM means in simple terms")
print(f"Gemini: {response}")

# 使用OpenAI GPT
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
gpt_agent = LLMAgent(provider="openai", model="gpt-4o-mini")
response = gpt_agent.process("What are the key components of a PHM system?")
print(f"GPT: {response}")

# 使用Mock进行测试（无需API密钥）
test_agent = LLMAgent(provider="mock", model="mock-model")
response = test_agent.process("Test input")
print(f"Mock: {response}")
```

### 2.2 结构化输出Agent

```python
# 示例3：生成结构化分析结果的Agent
from pydantic import BaseModel, Field
from typing import List
from src.model import get_llm

class DiagnosisResult(BaseModel):
    """诊断结果结构"""
    component: str = Field(description="被诊断的组件")
    status: str = Field(description="健康状态: healthy/warning/critical")
    confidence: float = Field(description="置信度 0-1")
    recommendations: List[str] = Field(description="建议措施")

class DiagnosticAgent:
    """诊断Agent，输出结构化结果"""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.3)  # 低温度for更确定的输出
        self.structured_llm = self.llm.with_structured_output(DiagnosisResult)
    
    def diagnose(self, sensor_data: dict) -> DiagnosisResult:
        """执行诊断并返回结构化结果"""
        prompt = f"""
        Analyze the following sensor data and provide a diagnosis:
        
        Sensor Data:
        - Temperature: {sensor_data.get('temperature', 'N/A')}°C
        - Vibration: {sensor_data.get('vibration', 'N/A')} mm/s
        - Pressure: {sensor_data.get('pressure', 'N/A')} bar
        
        Provide a structured diagnosis result.
        """
        
        result = self.structured_llm.invoke(prompt)
        return result

# 使用示例
diagnostic_agent = DiagnosticAgent()
sensor_data = {
    "temperature": 85,
    "vibration": 7.5,
    "pressure": 3.2
}

diagnosis = diagnostic_agent.diagnose(sensor_data)
print(f"Component: {diagnosis.component}")
print(f"Status: {diagnosis.status}")
print(f"Confidence: {diagnosis.confidence}")
print(f"Recommendations: {diagnosis.recommendations}")
```

---

## 第三部分：多Agent协作系统

### 3.1 构建Agent团队

```python
# 示例4：多Agent协作进行PHM分析
from enum import Enum
from typing import Dict, Any

class AgentRole(Enum):
    DATA_ANALYST = "data_analyst"
    FAULT_DETECTOR = "fault_detector"
    MAINTENANCE_PLANNER = "maintenance_planner"

class PHMAgentTeam:
    """PHM多Agent协作系统"""
    
    def __init__(self):
        # 为不同角色创建专门的Agent
        self.agents = {
            AgentRole.DATA_ANALYST: self._create_analyst_agent(),
            AgentRole.FAULT_DETECTOR: self._create_detector_agent(),
            AgentRole.MAINTENANCE_PLANNER: self._create_planner_agent()
        }
        self.workflow_state = {}
    
    def _create_analyst_agent(self):
        """创建数据分析Agent"""
        from src.model import get_llm
        llm = get_llm(temperature=0.5)
        
        def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
            prompt = f"""
            As a data analyst, analyze the following sensor data:
            {data}
            
            Identify patterns, anomalies, and trends.
            """
            analysis = llm.invoke(prompt)
            return {
                "analysis": analysis.content,
                "anomalies_detected": "anomaly" in analysis.content.lower()
            }
        
        return analyze
    
    def _create_detector_agent(self):
        """创建故障检测Agent"""
        from src.model import get_llm
        llm = get_llm(temperature=0.3)
        
        def detect(analysis: Dict[str, Any]) -> Dict[str, Any]:
            prompt = f"""
            As a fault detection specialist, based on this analysis:
            {analysis['analysis']}
            
            Determine if there are any faults or potential failures.
            """
            detection = llm.invoke(prompt)
            return {
                "fault_detected": analysis.get("anomalies_detected", False),
                "fault_description": detection.content
            }
        
        return detect
    
    def _create_planner_agent(self):
        """创建维护计划Agent"""
        from src.model import get_llm
        llm = get_llm(temperature=0.7)
        
        def plan(fault_info: Dict[str, Any]) -> Dict[str, Any]:
            if not fault_info.get("fault_detected"):
                return {"maintenance_required": False, "plan": "Continue normal operation"}
            
            prompt = f"""
            As a maintenance planner, create a maintenance plan for:
            {fault_info['fault_description']}
            
            Provide specific actions and timeline.
            """
            plan = llm.invoke(prompt)
            return {
                "maintenance_required": True,
                "plan": plan.content
            }
        
        return plan
    
    def process_phm_case(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理PHM案例的完整工作流"""
        # Step 1: 数据分析
        analysis_result = self.agents[AgentRole.DATA_ANALYST](sensor_data)
        self.workflow_state["analysis"] = analysis_result
        
        # Step 2: 故障检测
        fault_result = self.agents[AgentRole.FAULT_DETECTOR](analysis_result)
        self.workflow_state["fault_detection"] = fault_result
        
        # Step 3: 维护规划
        maintenance_result = self.agents[AgentRole.MAINTENANCE_PLANNER](fault_result)
        self.workflow_state["maintenance_plan"] = maintenance_result
        
        return {
            "sensor_data": sensor_data,
            "analysis": analysis_result,
            "fault_detection": fault_result,
            "maintenance_plan": maintenance_result
        }

# 使用示例
phm_team = PHMAgentTeam()
sensor_data = {
    "timestamp": "2024-01-20T10:00:00",
    "bearing_temperature": 95,  # 高温警告
    "vibration_rms": 12.5,      # 振动异常
    "rotation_speed": 1450
}

result = phm_team.process_phm_case(sensor_data)
print("Analysis:", result["analysis"])
print("Fault Detection:", result["fault_detection"])
print("Maintenance Plan:", result["maintenance_plan"])
```

---

## 第四部分：PHM Graph Agent - 高级实战

### 4.1 理解PHM Graph Agent架构

PHM Graph Agent使用图结构来组织复杂的诊断工作流：

```python
# 示例5：构建真实的PHM Graph Agent
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from src.model import get_llm
import operator

class PHMGraphState(TypedDict):
    """PHM图状态定义"""
    sensor_data: dict
    preprocessed_data: dict
    features: dict
    diagnosis: str
    confidence: float
    recommendations: Annotated[List[str], operator.add]
    history: Annotated[List[str], operator.add]

class PHMGraphAgent:
    """完整的PHM Graph Agent实现"""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.5)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建诊断工作流图"""
        workflow = StateGraph(PHMGraphState)
        
        # 添加节点
        workflow.add_node("preprocess", self.preprocess_node)
        workflow.add_node("extract_features", self.extract_features_node)
        workflow.add_node("diagnose", self.diagnose_node)
        workflow.add_node("recommend", self.recommend_node)
        
        # 定义边（工作流）
        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "extract_features")
        workflow.add_edge("extract_features", "diagnose")
        
        # 条件边：根据诊断结果决定是否需要推荐
        workflow.add_conditional_edges(
            "diagnose",
            self.should_recommend,
            {
                "recommend": "recommend",
                "end": END
            }
        )
        workflow.add_edge("recommend", END)
        
        return workflow.compile()
    
    def preprocess_node(self, state: PHMGraphState) -> dict:
        """预处理节点"""
        # 模拟数据预处理
        preprocessed = {
            "normalized_temp": state["sensor_data"].get("temperature", 0) / 100,
            "normalized_vib": state["sensor_data"].get("vibration", 0) / 10
        }
        
        return {
            "preprocessed_data": preprocessed,
            "history": [f"Preprocessed data at {state['sensor_data'].get('timestamp', 'unknown')}"]
        }
    
    def extract_features_node(self, state: PHMGraphState) -> dict:
        """特征提取节点"""
        # 使用LLM辅助特征工程
        prompt = f"""
        Extract key features from this preprocessed data:
        {state['preprocessed_data']}
        
        Identify important patterns for fault diagnosis.
        """
        
        features_text = self.llm.invoke(prompt).content
        
        # 模拟特征提取
        features = {
            "trend": "increasing" if state["preprocessed_data"]["normalized_temp"] > 0.8 else "stable",
            "vibration_level": "high" if state["preprocessed_data"]["normalized_vib"] > 0.7 else "normal",
            "llm_features": features_text
        }
        
        return {
            "features": features,
            "history": ["Extracted features from preprocessed data"]
        }
    
    def diagnose_node(self, state: PHMGraphState) -> dict:
        """诊断节点"""
        prompt = f"""
        Based on these features, provide a diagnosis:
        {state['features']}
        
        Original sensor data:
        {state['sensor_data']}
        
        Determine the health status and confidence level (0-1).
        """
        
        diagnosis = self.llm.invoke(prompt).content
        
        # 简单的置信度计算
        confidence = 0.85 if "high" in state["features"].get("vibration_level", "") else 0.95
        
        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "history": [f"Diagnosed with {confidence:.0%} confidence"]
        }
    
    def recommend_node(self, state: PHMGraphState) -> dict:
        """推荐节点"""
        prompt = f"""
        Based on this diagnosis:
        {state['diagnosis']}
        
        Confidence: {state['confidence']}
        Features: {state['features']}
        
        Provide 3 specific maintenance recommendations.
        """
        
        recommendations_text = self.llm.invoke(prompt).content
        recommendations = recommendations_text.split('\n')[:3]
        
        return {
            "recommendations": recommendations,
            "history": ["Generated maintenance recommendations"]
        }
    
    def should_recommend(self, state: PHMGraphState) -> str:
        """决定是否需要生成推荐"""
        # 如果置信度低于90%或检测到异常，生成推荐
        if state["confidence"] < 0.9 or "high" in state.get("features", {}).get("vibration_level", ""):
            return "recommend"
        return "end"
    
    def run(self, sensor_data: dict) -> dict:
        """运行完整的PHM诊断流程"""
        initial_state = {
            "sensor_data": sensor_data,
            "preprocessed_data": {},
            "features": {},
            "diagnosis": "",
            "confidence": 0.0,
            "recommendations": [],
            "history": []
        }
        
        result = self.graph.invoke(initial_state)
        return result

# 使用PHM Graph Agent
phm_graph_agent = PHMGraphAgent()

# 测试案例1：正常状态
normal_data = {
    "timestamp": "2024-01-20T10:00:00",
    "temperature": 65,
    "vibration": 3.2,
    "pressure": 2.8
}

result = phm_graph_agent.run(normal_data)
print("\n=== Normal Case ===")
print(f"Diagnosis: {result['diagnosis'][:100]}...")
print(f"Confidence: {result['confidence']:.0%}")
print(f"History: {result['history']}")

# 测试案例2：异常状态
abnormal_data = {
    "timestamp": "2024-01-20T11:00:00",
    "temperature": 92,
    "vibration": 8.5,
    "pressure": 4.2
}

result = phm_graph_agent.run(abnormal_data)
print("\n=== Abnormal Case ===")
print(f"Diagnosis: {result['diagnosis'][:100]}...")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Recommendations: {result['recommendations']}")
```

### 4.2 集成信号处理DAG

```python
# 示例6：完整的双层架构 - Graph Agent + Signal Processing DAG
from src.states.phm_states import PHMState, DAGState, InputData
from src.agents.data_analyst_agent import DataAnalystAgent
import numpy as np

class AdvancedPHMSystem:
    """高级PHM系统：结合Graph Agent和信号处理DAG"""
    
    def __init__(self):
        # 初始化不同的Agent
        self.data_analyst = DataAnalystAgent(config={"quick_mode": False})
        self.graph_agent = PHMGraphAgent()
        
    def process_bearing_fault_case(self, reference_signal: np.ndarray, test_signal: np.ndarray):
        """处理轴承故障诊断案例"""
        
        # Step 1: 创建PHM状态
        ref_data = InputData(
            node_id="ref_signal",
            parents=[],
            shape=reference_signal.shape,
            results={"signal": reference_signal},
            meta={"fs": 10000}  # 采样频率
        )
        
        test_data = InputData(
            node_id="test_signal",
            parents=[],
            shape=test_signal.shape,
            results={"signal": test_signal},
            meta={"fs": 10000}
        )
        
        dag_state = DAGState(
            user_instruction="Diagnose bearing fault",
            channels=["vibration"],
            nodes={"ref": ref_data, "test": test_data},
            leaves=["ref", "test"]
        )
        
        phm_state = PHMState(
            case_name="bearing_diagnosis",
            user_instruction="Analyze bearing vibration for fault detection",
            reference_signal=ref_data,
            test_signal=test_data,
            dag_state=dag_state
        )
        
        # Step 2: 数据分析Agent处理
        analysis_result = self.data_analyst.analyze(phm_state)
        
        # Step 3: 将分析结果传递给Graph Agent
        sensor_data = {
            "timestamp": "2024-01-20T12:00:00",
            "analysis_confidence": analysis_result.confidence,
            "detected_features": analysis_result.results,
            "signal_energy": float(np.mean(test_signal**2))
        }
        
        graph_result = self.graph_agent.run(sensor_data)
        
        # Step 4: 综合结果
        return {
            "signal_analysis": {
                "confidence": analysis_result.confidence,
                "features": analysis_result.results,
                "execution_time": analysis_result.execution_time
            },
            "graph_diagnosis": {
                "diagnosis": graph_result["diagnosis"],
                "recommendations": graph_result["recommendations"],
                "workflow_history": graph_result["history"]
            }
        }

# 实战演示
advanced_phm = AdvancedPHMSystem()

# 生成模拟信号
fs = 10000
t = np.linspace(0, 1, fs)

# 健康轴承信号
healthy_signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(len(t))
healthy_signal = healthy_signal.reshape(1, -1, 1)

# 故障轴承信号（包含冲击成分）
faulty_signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(len(t))
# 添加周期性冲击
for i in range(0, len(t), 200):
    if i+10 < len(t):
        faulty_signal[i:i+10] += 0.5 * np.exp(-0.5*np.arange(10))
faulty_signal = faulty_signal.reshape(1, -1, 1)

# 运行完整诊断
result = advanced_phm.process_bearing_fault_case(healthy_signal, faulty_signal)

print("\n=== Advanced PHM System Results ===")
print(f"Signal Analysis Confidence: {result['signal_analysis']['confidence']:.2%}")
print(f"Diagnosis: {result['graph_diagnosis']['diagnosis'][:200]}...")
print(f"Workflow Steps: {result['graph_diagnosis']['workflow_history']}")
```

---

## 第五部分：生产环境部署

### 5.1 配置管理和环境切换

```python
# 示例7：生产环境配置
import os
from src.states.phm_states import get_unified_state, reset_unified_state

class ProductionPHMAgent:
    """生产环境的PHM Agent配置"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self._configure_environment()
        
    def _configure_environment(self):
        """根据环境配置系统"""
        reset_unified_state()
        unified_state = get_unified_state()
        
        if self.environment == "production":
            # 生产环境：使用高性能模型
            unified_state.set('llm.provider', 'google')
            unified_state.set('llm.model', 'gemini-2.5-pro')
            unified_state.set('llm.temperature', 0.3)  # 更稳定的输出
            unified_state.set('processing.max_depth', 10)
            unified_state.set('processing.enable_cache', True)
            
        elif self.environment == "development":
            # 开发环境：使用快速模型
            unified_state.set('llm.provider', 'google')
            unified_state.set('llm.model', 'gemini-2.5-flash')
            unified_state.set('llm.temperature', 0.7)
            unified_state.set('processing.max_depth', 5)
            unified_state.set('processing.enable_cache', False)
            
        elif self.environment == "testing":
            # 测试环境：使用Mock
            unified_state.set('llm.provider', 'mock')
            unified_state.set('llm.model', 'mock-model')
            unified_state.set('processing.max_depth', 3)
        
        print(f"Configured for {self.environment} environment")
    
    def switch_provider(self, provider: str, model: str):
        """动态切换LLM提供商"""
        unified_state = get_unified_state()
        unified_state.set('llm.provider', provider)
        unified_state.set('llm.model', model)
        print(f"Switched to {provider}/{model}")
    
    def get_current_config(self):
        """获取当前配置"""
        unified_state = get_unified_state()
        return {
            "environment": self.environment,
            "llm_config": unified_state.get_llm_config(),
            "processing_config": unified_state.get_processing_config()
        }

# 使用示例
# 开发环境
dev_agent = ProductionPHMAgent("development")
print(dev_agent.get_current_config())

# 切换到生产环境
prod_agent = ProductionPHMAgent("production")
print(prod_agent.get_current_config())

# 动态切换提供商（例如：OpenAI出现问题时切换到Google）
prod_agent.switch_provider("openai", "gpt-4o")
```

### 5.2 错误处理和降级策略

```python
# 示例8：带有降级策略的健壮系统
from src.model import get_llm_factory

class RobustPHMAgent:
    """具有降级策略的健壮PHM Agent"""
    
    def __init__(self):
        self.factory = get_llm_factory()
        self.primary_provider = "google"
        self.fallback_provider = "openai"
        self.emergency_provider = "mock"
        
    def process_with_fallback(self, instruction: str):
        """带降级的处理流程"""
        providers = [
            (self.primary_provider, "gemini-2.5-pro"),
            (self.fallback_provider, "gpt-4o"),
            (self.emergency_provider, "mock-model")
        ]
        
        for provider, model in providers:
            try:
                print(f"Trying {provider}/{model}...")
                llm = self.factory.create_with_fallback(
                    preferred_provider=provider,
                    model=model,
                    fallback_provider=self.emergency_provider
                )
                
                response = llm.invoke(instruction)
                return {
                    "success": True,
                    "provider": provider,
                    "response": response.content
                }
                
            except Exception as e:
                print(f"Failed with {provider}: {e}")
                continue
        
        return {
            "success": False,
            "error": "All providers failed"
        }

# 测试降级策略
robust_agent = RobustPHMAgent()
result = robust_agent.process_with_fallback("Analyze system health")
print(f"Result: {result}")
```

---

## 总结与最佳实践

### 关键要点

1. **渐进式学习**：从简单Agent到复杂的Graph Agent系统
2. **多提供商支持**：灵活切换Google、OpenAI等LLM提供商
3. **结构化设计**：使用TypedDict和Pydantic确保类型安全
4. **双层架构**：Graph Agent处理工作流，DAG处理信号
5. **生产就绪**：包含错误处理、降级策略和环境配置

### 快速开始检查清单

- [ ] 设置API密钥：`GEMINI_API_KEY` 或 `OPENAI_API_KEY`
- [ ] 安装依赖：`pip install langchain langgraph langchain-google-genai`
- [ ] 选择合适的Agent类型（简单/LLM/Graph）
- [ ] 配置环境（开发/测试/生产）
- [ ] 实现错误处理和降级策略

### 下一步

1. 探索更多信号处理操作符
2. 自定义Agent节点和工作流
3. 集成实时数据流
4. 部署到生产环境

---

*本教程是PHM Graph Agent系统的完整学习路径，从基础概念到生产部署的全面指南。*