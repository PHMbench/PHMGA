"""
LangGraph工作流模块

本模块展示如何使用LangGraph创建复杂的工作流，包括：
- 多节点工作流
- 条件分支和路由
- 并行处理
- 错误处理和重试
"""

from typing import Dict, Any, List, TypedDict, Annotated
import operator
import time

# 依赖检查和回退处理
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph模块可用")
except ImportError:
    print("⚠️ LangGraph模块不可用，将使用模拟实现")
    LANGGRAPH_AVAILABLE = False
    
    # 模拟LangGraph类
    class MockStateGraph:
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = []
        
        def add_node(self, name, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node, to_node):
            self.edges.append((from_node, to_node))
        
        def set_entry_point(self, node_name):
            self.entry_point = node_name
        
        def compile(self):
            return MockCompiledGraph(self)
    
    class MockCompiledGraph:
        def __init__(self, graph):
            self.graph = graph
        
        def invoke(self, input_data):
            # 简单的模拟执行
            return {
                "result": "模拟LangGraph执行结果",
                "steps": list(self.graph.nodes.keys()),
                "mock": True
            }
    
    StateGraph = MockStateGraph
    END = "END"


class WorkflowState(TypedDict):
    """工作流状态定义"""
    input_data: Dict[str, Any]
    processing_history: Annotated[List[str], operator.add]
    current_node: str
    results: Dict[str, Any]
    should_continue: bool


class AdvancedWorkflowState(TypedDict):
    """高级工作流状态"""
    sensor_readings: Dict[str, float]
    analysis_path: str
    processing_steps: Annotated[List[str], operator.add]
    parallel_results: Dict[str, Any]
    final_decision: str
    confidence_level: float


class PHMAnalysisWorkflow:
    """PHM分析工作流"""
    
    def __init__(self):
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        
        def preprocessing_node(state: WorkflowState) -> Dict[str, Any]:
            """数据预处理节点"""
            print("🔄 执行数据预处理...")
            
            # 模拟预处理逻辑
            raw_data = state["input_data"].get("sensor_data", {})
            processed_data = {k: v * 1.1 for k, v in raw_data.items() if isinstance(v, (int, float))}
            
            return {
                "processing_history": ["数据预处理完成"],
                "current_node": "preprocessing",
                "results": {"processed_data": processed_data}
            }
        
        def feature_extraction_node(state: WorkflowState) -> Dict[str, Any]:
            """特征提取节点"""
            print("🔍 执行特征提取...")
            
            processed_data = state["results"].get("processed_data", {})
            features = {
                "mean_value": sum(processed_data.values()) / len(processed_data) if processed_data else 0,
                "max_value": max(processed_data.values()) if processed_data else 0,
                "feature_count": len(processed_data)
            }
            
            return {
                "processing_history": ["特征提取完成"],
                "current_node": "feature_extraction",
                "results": {**state["results"], "features": features}
            }
        
        def classification_node(state: WorkflowState) -> Dict[str, Any]:
            """分类诊断节点"""
            print("🎯 执行故障诊断...")
            
            features = state["results"].get("features", {})
            mean_val = features.get("mean_value", 0)
            
            # 简单的分类逻辑
            if mean_val > 80:
                diagnosis = "严重故障"
                confidence = 0.9
            elif mean_val > 60:
                diagnosis = "轻微异常"
                confidence = 0.7
            else:
                diagnosis = "正常状态"
                confidence = 0.95
            
            return {
                "processing_history": [f"诊断完成: {diagnosis} (置信度: {confidence:.2f})"],
                "current_node": "classification",
                "results": {**state["results"], "diagnosis": diagnosis, "confidence": confidence},
                "should_continue": False
            }
        
        # 构建图
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("preprocess", preprocessing_node)
        workflow.add_node("extract_features", feature_extraction_node)
        workflow.add_node("classify", classification_node)
        
        # 定义执行顺序
        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "extract_features")
        workflow.add_edge("extract_features", "classify")
        workflow.add_edge("classify", END)
        
        return workflow.compile()
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行分析"""
        initial_state = {
            "input_data": input_data,
            "processing_history": ["工作流启动"],
            "current_node": "start",
            "results": {},
            "should_continue": True
        }
        
        result = self.workflow.invoke(initial_state)
        return result


class AdvancedPHMWorkflow:
    """高级PHM工作流，支持条件分支和并行处理"""
    
    def __init__(self):
        self.workflow = self._build_advanced_workflow()
    
    def _build_advanced_workflow(self) -> StateGraph:
        """构建高级工作流"""
        
        def data_validation_node(state: AdvancedWorkflowState) -> Dict[str, Any]:
            """数据验证节点"""
            readings = state["sensor_readings"]
            
            # 检查数据质量
            missing_sensors = []
            required_sensors = ["temperature", "vibration", "pressure"]
            
            for sensor in required_sensors:
                if sensor not in readings or readings[sensor] is None:
                    missing_sensors.append(sensor)
            
            if missing_sensors:
                analysis_path = "simple_analysis"
                step_msg = f"数据不完整（缺少: {missing_sensors}），使用简单分析"
            else:
                analysis_path = "full_analysis"
                step_msg = "数据完整，使用完整分析流程"
            
            return {
                "analysis_path": analysis_path,
                "processing_steps": [step_msg]
            }
        
        def simple_analysis_node(state: AdvancedWorkflowState) -> Dict[str, Any]:
            """简单分析节点"""
            readings = state["sensor_readings"]
            
            avg_value = sum(v for v in readings.values() if v is not None) / len(readings)
            
            if avg_value > 70:
                decision = "需要注意：传感器读数偏高"
                confidence = 0.6
            else:
                decision = "状态正常"
                confidence = 0.8
            
            return {
                "processing_steps": ["执行简单分析"],
                "final_decision": decision,
                "confidence_level": confidence
            }
        
        def parallel_analysis_node(state: AdvancedWorkflowState) -> Dict[str, Any]:
            """并行分析节点（模拟）"""
            readings = state["sensor_readings"]
            
            # 模拟并行处理的多个分析
            temp_analysis = {
                "result": "正常" if readings.get("temperature", 0) < 80 else "异常",
                "score": 0.9 if readings.get("temperature", 0) < 80 else 0.3
            }
            
            vibration_analysis = {
                "result": "正常" if readings.get("vibration", 0) < 5 else "异常",
                "score": 0.85 if readings.get("vibration", 0) < 5 else 0.2
            }
            
            pressure_analysis = {
                "result": "正常" if readings.get("pressure", 0) > 1.5 else "异常",
                "score": 0.9 if readings.get("pressure", 0) > 1.5 else 0.4
            }
            
            parallel_results = {
                "temperature": temp_analysis,
                "vibration": vibration_analysis,
                "pressure": pressure_analysis
            }
            
            return {
                "processing_steps": ["执行并行分析（温度、振动、压力）"],
                "parallel_results": parallel_results
            }
        
        def decision_fusion_node(state: AdvancedWorkflowState) -> Dict[str, Any]:
            """决策融合节点"""
            parallel_results = state["parallel_results"]
            
            total_score = 0
            abnormal_count = 0
            
            for sensor, analysis in parallel_results.items():
                total_score += analysis["score"]
                if analysis["result"] == "异常":
                    abnormal_count += 1
            
            avg_confidence = total_score / len(parallel_results)
            
            if abnormal_count >= 2:
                decision = "多个传感器异常，建议立即检修"
            elif abnormal_count == 1:
                decision = "单个传感器异常，建议密切监控"
            else:
                decision = "所有传感器正常"
            
            return {
                "processing_steps": [f"决策融合完成，异常传感器数量: {abnormal_count}"],
                "final_decision": decision,
                "confidence_level": avg_confidence
            }
        
        # 路由决策函数
        def route_analysis(state: AdvancedWorkflowState) -> str:
            """根据数据质量选择分析路径"""
            return state["analysis_path"]
        
        def should_fuse_results(state: AdvancedWorkflowState) -> str:
            """判断是否需要结果融合"""
            if state.get("parallel_results"):
                return "fusion"
            else:
                return "end"
        
        # 构建图
        workflow = StateGraph(AdvancedWorkflowState)
        
        # 添加节点
        workflow.add_node("validate", data_validation_node)
        workflow.add_node("simple_analysis", simple_analysis_node)
        workflow.add_node("parallel_analysis", parallel_analysis_node)
        workflow.add_node("fusion", decision_fusion_node)
        
        # 设置入口
        workflow.set_entry_point("validate")
        
        # 条件路由
        workflow.add_conditional_edges(
            "validate",
            route_analysis,
            {
                "simple_analysis": "simple_analysis",
                "full_analysis": "parallel_analysis"
            }
        )
        
        # 简单分析直接结束
        workflow.add_edge("simple_analysis", END)
        
        # 并行分析后进行融合
        workflow.add_conditional_edges(
            "parallel_analysis",
            should_fuse_results,
            {
                "fusion": "fusion",
                "end": END
            }
        )
        
        workflow.add_edge("fusion", END)
        
        return workflow.compile()
    
    def run_analysis(self, sensor_readings: Dict[str, float]) -> Dict[str, Any]:
        """运行高级分析"""
        initial_state = {
            "sensor_readings": sensor_readings,
            "analysis_path": "",
            "processing_steps": ["高级工作流启动"],
            "parallel_results": {},
            "final_decision": "",
            "confidence_level": 0.0
        }
        
        result = self.workflow.invoke(initial_state)
        return result


def demo_basic_workflow():
    """演示基础工作流"""
    print("🏭 基础PHM分析工作流演示")
    
    phm_workflow = PHMAnalysisWorkflow()
    
    test_data = {
        "sensor_data": {
            "temperature": 75.5,
            "vibration": 4.2,
            "pressure": 2.3
        },
        "timestamp": "2024-01-15 10:30:00"
    }
    
    print(f"输入数据: {test_data}")
    result = phm_workflow.analyze(test_data)
    
    print(f"\n✅ 工作流执行完成")
    print(f"执行历史: {result['processing_history']}")
    print(f"最终诊断: {result['results'].get('diagnosis', 'N/A')}")
    print(f"置信度: {result['results'].get('confidence', 0):.2f}")
    
    return result


def demo_advanced_workflow():
    """演示高级工作流"""
    print("\n🚀 高级PHM工作流演示")
    
    advanced_workflow = AdvancedPHMWorkflow()
    
    test_scenarios = [
        {
            "name": "完整数据 - 正常状态",
            "data": {"temperature": 70, "vibration": 3.0, "pressure": 2.2}
        },
        {
            "name": "完整数据 - 多项异常",
            "data": {"temperature": 85, "vibration": 6.5, "pressure": 1.0}
        },
        {
            "name": "不完整数据",
            "data": {"temperature": 75, "pressure": 2.1}  # 缺少振动数据
        }
    ]
    
    results = []
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"🧪 测试场景: {scenario['name']}")
        print(f"📊 输入数据: {scenario['data']}")
        
        result = advanced_workflow.run_analysis(scenario['data'])
        results.append(result)
        
        print(f"🔄 处理路径: {result['analysis_path']}")
        print(f"📝 处理步骤: {result['processing_steps']}")
        print(f"🎯 最终决策: {result['final_decision']}")
        print(f"📈 置信度: {result['confidence_level']:.2f}")
        
        if result.get('parallel_results'):
            print(f"🔄 并行分析结果:")
            for sensor, analysis in result['parallel_results'].items():
                print(f"  {sensor}: {analysis['result']} (得分: {analysis['score']:.2f})")
    
    return results


if __name__ == "__main__":
    demo_basic_workflow()
    demo_advanced_workflow()