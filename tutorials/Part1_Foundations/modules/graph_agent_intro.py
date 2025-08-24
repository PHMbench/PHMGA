"""
Graph Agent入门模块

本模块介绍Graph Agent的核心概念，展示从传统Agent到Graph Agent的演进，
并提供基于LangGraph的实际实现示例。
"""

from typing import Dict, Any, List, TypedDict, Annotated, Optional
import operator
import time
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from .llm_providers_unified import create_llm


class SimpleGraphState(TypedDict):
    """简单Graph状态定义"""
    messages: Annotated[List[str], operator.add]
    current_step: str
    input_data: Any
    result: Optional[str]
    metadata: Dict[str, Any]


class DiagnosticGraphState(TypedDict):
    """诊断Graph状态定义"""
    sensor_data: Dict[str, float]
    analysis_result: Optional[str]
    diagnosis: Optional[str] 
    action_plan: Optional[str]
    confidence_score: float
    steps_completed: Annotated[List[str], operator.add]
    timestamp: str


class SimpleGraphAgent:
    """
    简单的Graph Agent实现
    
    演示Graph Agent的基本概念：
    - 状态管理
    - 节点定义 
    - 边和流程控制
    - 条件路由
    """
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建简单的Graph工作流"""
        
        # 定义节点函数
        def input_node(state: SimpleGraphState) -> SimpleGraphState:
            """输入处理节点"""
            return {
                "messages": ["📥 接收输入数据"],
                "current_step": "input_received",
                "metadata": {"timestamp": time.time()}
            }
        
        def analyze_node(state: SimpleGraphState) -> SimpleGraphState:
            """分析节点"""
            input_data = state.get("input_data", "")
            analysis = f"分析输入: {input_data}" if input_data else "分析空输入"
            
            return {
                "messages": [f"🔍 {analysis}"],
                "current_step": "analysis_complete",
                "result": f"分析结果: {analysis}"
            }
        
        def decide_node(state: SimpleGraphState) -> SimpleGraphState:
            """决策节点"""
            return {
                "messages": ["🤔 基于分析结果进行决策"],
                "current_step": "decision_made"
            }
        
        def action_node(state: SimpleGraphState) -> SimpleGraphState:
            """行动节点"""
            return {
                "messages": ["🎬 执行决策行动"],
                "current_step": "action_completed",
                "result": "行动已执行"
            }
        
        # 创建StateGraph
        workflow = StateGraph(SimpleGraphState)
        
        # 添加节点
        workflow.add_node("input", input_node)
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("decide", decide_node)
        workflow.add_node("action", action_node)
        
        # 定义流程（边）
        workflow.set_entry_point("input")
        workflow.add_edge("input", "analyze")
        workflow.add_edge("analyze", "decide")
        workflow.add_edge("decide", "action")
        workflow.add_edge("action", END)
        
        return workflow.compile()
    
    def run(self, input_data: Any) -> SimpleGraphState:
        """运行Graph Agent"""
        initial_state = {
            "messages": [],
            "current_step": "initialized",
            "input_data": input_data,
            "result": None,
            "metadata": {}
        }
        
        result = self.graph.invoke(initial_state)
        return result


class LLMGraphAgent:
    """
    集成LLM的Graph Agent
    
    展示如何在Graph节点中使用LLM进行智能决策
    """
    
    def __init__(self, llm_provider: str = "mock"):
        self.llm = create_llm(llm_provider, temperature=0.7)
        self.graph = self._build_llm_graph()
    
    def _build_llm_graph(self) -> StateGraph:
        """构建集成LLM的Graph"""
        
        def llm_analyze_node(state: SimpleGraphState) -> SimpleGraphState:
            """使用LLM进行分析的节点"""
            input_data = state.get("input_data", "")
            
            prompt = f"""
            作为一个智能分析助手，请分析以下输入并提供洞察：
            
            输入: {input_data}
            
            请提供：
            1. 关键信息识别
            2. 可能的问题或机会
            3. 建议的下一步行动
            
            请用简洁的中文回答。
            """
            
            try:
                response = self.llm.invoke(prompt)
                analysis = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                analysis = f"LLM分析失败: {e}"
            
            return {
                "messages": [f"🧠 LLM分析完成"],
                "current_step": "llm_analysis_complete",
                "result": analysis
            }
        
        def llm_decide_node(state: SimpleGraphState) -> SimpleGraphState:
            """使用LLM进行决策的节点"""
            analysis_result = state.get("result", "")
            
            prompt = f"""
            基于以下分析结果，请做出决策：
            
            分析结果: {analysis_result}
            
            请选择最合适的行动方案，并简要说明理由。
            可选行动: 继续监控、深入调查、立即干预、请求支援
            
            请用简洁的中文回答。
            """
            
            try:
                response = self.llm.invoke(prompt)
                decision = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                decision = f"LLM决策失败: {e}"
            
            return {
                "messages": [f"🎯 LLM决策完成"],
                "current_step": "llm_decision_made",
                "result": decision
            }
        
        def execute_node(state: SimpleGraphState) -> SimpleGraphState:
            """执行节点"""
            decision = state.get("result", "")
            
            return {
                "messages": [f"✅ 执行决策: {decision[:50]}..."],
                "current_step": "execution_complete"
            }
        
        # 构建Graph
        workflow = StateGraph(SimpleGraphState)
        
        workflow.add_node("llm_analyze", llm_analyze_node)
        workflow.add_node("llm_decide", llm_decide_node)
        workflow.add_node("execute", execute_node)
        
        workflow.set_entry_point("llm_analyze")
        workflow.add_edge("llm_analyze", "llm_decide")
        workflow.add_edge("llm_decide", "execute")
        workflow.add_edge("execute", END)
        
        return workflow.compile()
    
    def run(self, input_data: Any) -> SimpleGraphState:
        """运行LLM Graph Agent"""
        initial_state = {
            "messages": [],
            "current_step": "initialized",
            "input_data": input_data,
            "result": None,
            "metadata": {"llm_provider": self.llm.__class__.__name__}
        }
        
        result = self.graph.invoke(initial_state)
        return result


class ConditionalGraphAgent:
    """
    带条件路由的Graph Agent
    
    展示Graph Agent如何根据状态进行条件分支和循环
    """
    
    def __init__(self, llm_provider: str = "mock"):
        self.llm = create_llm(llm_provider, temperature=0.3)
        self.graph = self._build_conditional_graph()
    
    def _build_conditional_graph(self) -> StateGraph:
        """构建带条件路由的Graph"""
        
        def assess_node(state: DiagnosticGraphState) -> DiagnosticGraphState:
            """评估节点"""
            sensor_data = state["sensor_data"]
            
            # 计算异常分数
            anomaly_score = 0
            for key, value in sensor_data.items():
                if value > 0.8 or value < 0.2:
                    anomaly_score += abs(value - 0.5) * 2
            
            confidence = min(1.0, anomaly_score)
            
            assessment = f"传感器评估完成，异常分数: {anomaly_score:.2f}"
            
            return {
                "analysis_result": assessment,
                "confidence_score": confidence,
                "steps_completed": ["assessment"],
                "timestamp": str(time.time())
            }
        
        def detailed_analysis_node(state: DiagnosticGraphState) -> DiagnosticGraphState:
            """详细分析节点（条件触发）"""
            sensor_data = state["sensor_data"]
            
            prompt = f"""
            传感器数据显示异常，请进行详细分析：
            
            数据: {sensor_data}
            初步评估: {state.get('analysis_result', '')}
            
            请识别：
            1. 主要异常类型
            2. 可能的根本原因
            3. 风险等级评估
            """
            
            try:
                response = self.llm.invoke(prompt)
                detailed_analysis = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                detailed_analysis = f"详细分析失败: {e}"
            
            return {
                "diagnosis": detailed_analysis,
                "steps_completed": ["detailed_analysis"]
            }
        
        def quick_check_node(state: DiagnosticGraphState) -> DiagnosticGraphState:
            """快速检查节点（正常情况）"""
            return {
                "diagnosis": "系统状态正常，无需进一步行动",
                "action_plan": "继续例行监控",
                "steps_completed": ["quick_check"]
            }
        
        def action_planning_node(state: DiagnosticGraphState) -> DiagnosticGraphState:
            """行动规划节点"""
            diagnosis = state.get("diagnosis", "")
            
            prompt = f"""
            基于诊断结果制定行动计划：
            
            诊断: {diagnosis}
            
            请提供具体的行动步骤和时间安排。
            """
            
            try:
                response = self.llm.invoke(prompt)
                action_plan = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                action_plan = f"行动规划失败: {e}"
            
            return {
                "action_plan": action_plan,
                "steps_completed": ["action_planning"]
            }
        
        # 条件路由函数
        def route_after_assessment(state: DiagnosticGraphState) -> str:
            """评估后的路由决策"""
            confidence = state.get("confidence_score", 0)
            
            if confidence > 0.6:  # 高异常分数需要详细分析
                return "detailed_analysis"
            else:  # 低异常分数进行快速检查
                return "quick_check"
        
        def route_after_diagnosis(state: DiagnosticGraphState) -> str:
            """诊断后的路由决策"""
            diagnosis = state.get("diagnosis", "")
            
            if "正常" in diagnosis:
                return END  # 正常情况直接结束
            else:
                return "action_planning"  # 异常情况需要制定行动计划
        
        # 构建Graph
        workflow = StateGraph(DiagnosticGraphState)
        
        # 添加节点
        workflow.add_node("assess", assess_node)
        workflow.add_node("detailed_analysis", detailed_analysis_node)
        workflow.add_node("quick_check", quick_check_node)
        workflow.add_node("action_planning", action_planning_node)
        
        # 设置入口点
        workflow.set_entry_point("assess")
        
        # 条件路由
        workflow.add_conditional_edges(
            "assess",
            route_after_assessment,
            {
                "detailed_analysis": "detailed_analysis",
                "quick_check": "quick_check"
            }
        )
        
        # 详细分析后总是需要行动规划
        workflow.add_edge("detailed_analysis", "action_planning")
        
        # 快速检查的条件路由
        workflow.add_conditional_edges(
            "quick_check",
            route_after_diagnosis,
            {
                "action_planning": "action_planning",
                END: END
            }
        )
        
        # 行动规划后结束
        workflow.add_edge("action_planning", END)
        
        return workflow.compile()
    
    def run(self, sensor_data: Dict[str, float]) -> DiagnosticGraphState:
        """运行条件Graph Agent"""
        initial_state = {
            "sensor_data": sensor_data,
            "analysis_result": None,
            "diagnosis": None,
            "action_plan": None,
            "confidence_score": 0.0,
            "steps_completed": [],
            "timestamp": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result


def demonstrate_graph_agents():
    """演示不同类型的Graph Agent"""
    print("🕸️ Graph Agent演示")
    print("=" * 50)
    
    # 1. 简单Graph Agent
    print("\n1️⃣ 简单Graph Agent:")
    simple_agent = SimpleGraphAgent()
    result1 = simple_agent.run("系统温度监控数据")
    
    print("执行步骤:")
    for i, message in enumerate(result1["messages"], 1):
        print(f"  {i}. {message}")
    print(f"最终结果: {result1.get('result', 'N/A')}")
    
    # 2. LLM Graph Agent
    print("\n2️⃣ LLM增强Graph Agent:")
    llm_agent = LLMGraphAgent("mock")
    result2 = llm_agent.run("设备振动频率异常，需要诊断分析")
    
    print("执行步骤:")
    for i, message in enumerate(result2["messages"], 1):
        print(f"  {i}. {message}")
    print(f"LLM决策结果: {result2.get('result', 'N/A')[:100]}...")
    
    # 3. 条件Graph Agent
    print("\n3️⃣ 条件路由Graph Agent:")
    conditional_agent = ConditionalGraphAgent("mock")
    
    # 测试正常情况
    normal_data = {"temperature": 0.5, "pressure": 0.4, "vibration": 0.6}
    result3a = conditional_agent.run(normal_data)
    print(f"正常数据 {normal_data}:")
    print(f"  执行路径: {' -> '.join(result3a['steps_completed'])}")
    print(f"  诊断: {result3a.get('diagnosis', 'N/A')}")
    
    # 测试异常情况
    abnormal_data = {"temperature": 0.9, "pressure": 0.1, "vibration": 0.95}
    result3b = conditional_agent.run(abnormal_data)
    print(f"\\n异常数据 {abnormal_data}:")
    print(f"  执行路径: {' -> '.join(result3b['steps_completed'])}")
    print(f"  置信分数: {result3b.get('confidence_score', 0):.2f}")
    print(f"  行动计划: {result3b.get('action_plan', 'N/A')[:100]}...")
    
    return result1, result2, result3a, result3b


def compare_traditional_vs_graph():
    """对比传统Agent和Graph Agent"""
    print("\\n⚖️ 传统Agent vs Graph Agent对比")
    print("=" * 50)
    
    comparison_table = """
    | 特性 | 传统Agent | Graph Agent |
    |------|-----------|-------------|
    | 执行流程 | 线性（感知→思考→行动） | 图结构（灵活路由） |
    | 状态管理 | 简单变量 | 结构化状态 |
    | 决策复杂度 | 单步决策 | 多步条件决策 |
    | 并行处理 | 困难 | 原生支持 |
    | 流程可视化 | 隐含 | 明确的图结构 |
    | 扩展性 | 有限 | 高度模块化 |
    | 调试能力 | 困难 | 状态可追踪 |
    | 错误恢复 | 重新开始 | 节点级重试 |
    """
    
    print(comparison_table)
    
    print("\\n🎯 Graph Agent的优势:")
    print("✅ 明确的状态管理")
    print("✅ 灵活的条件路由") 
    print("✅ 可视化的工作流")
    print("✅ 模块化的节点设计")
    print("✅ 更好的错误处理")
    print("✅ 支持复杂的业务逻辑")
    
    print("\\n⚠️ 使用建议:")
    print("• 简单任务使用传统Agent")
    print("• 复杂工作流使用Graph Agent")  
    print("• 需要条件分支时选择Graph Agent")
    print("• 多步骤协作场景使用Graph Agent")


if __name__ == "__main__":
    # 运行演示
    demonstrate_graph_agents()
    compare_traditional_vs_graph()