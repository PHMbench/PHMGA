"""
ReAct模式实现模块

ReAct (Reasoning + Acting) 是一种强大的Agent模式，它结合了推理和行动，
让Agent能够在执行过程中进行思考、观察和调整。

本模块提供基于LangGraph的ReAct模式实现。
"""

from typing import Dict, Any, List, TypedDict, Annotated, Optional, Union
import operator
import time
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor


class ReActState(TypedDict):
    """ReAct模式的状态定义"""
    input: str  # 初始输入
    thought: str  # 当前思考
    action: str  # 要执行的行动
    action_input: str  # 行动输入
    observation: str  # 观察结果
    final_answer: str  # 最终答案
    steps: Annotated[List[Dict[str, str]], operator.add]  # 执行步骤历史
    iteration: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数


class ReActAgent:
    """
    基于LangGraph的ReAct Agent实现
    
    ReAct模式的核心循环：
    1. Thought (思考) - 分析当前情况
    2. Action (行动) - 选择和执行工具
    3. Observation (观察) - 观察行动结果
    4. 重复或结束
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[Tool],
        max_iterations: int = 10,
        verbose: bool = False
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_executor = ToolExecutor(tools)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.graph = self._build_react_graph()
        
        # 创建工具描述
        self.tool_descriptions = self._format_tools()
    
    def _format_tools(self) -> str:
        """格式化工具描述"""
        if not self.tools:
            return "没有可用的工具。"
        
        tool_strings = []
        for tool in self.tools.values():
            tool_strings.append(f"{tool.name}: {tool.description}")
        
        return "可用工具:\n" + "\n".join(tool_strings)
    
    def _build_react_graph(self) -> StateGraph:
        """构建ReAct图结构"""
        
        def think_node(state: ReActState) -> Dict[str, Any]:
            """思考节点"""
            input_text = state["input"]
            steps = state.get("steps", [])
            iteration = state.get("iteration", 0)
            
            # 构建思考提示
            if iteration == 0:
                # 第一次思考
                prompt = f"""
你是一个智能助手，需要回答用户的问题。你可以使用以下工具：

{self.tool_descriptions}

问题: {input_text}

请按照以下格式进行推理：
Thought: 你的思考过程
Action: 选择一个工具名称
Action Input: 工具的输入参数

如果你已经有足够信息回答问题，请输出：
Thought: 我现在可以回答这个问题
Final Answer: 你的最终答案
"""
            else:
                # 基于之前的步骤继续思考
                history = "\n".join([
                    f"Thought: {step.get('thought', '')}\n"
                    f"Action: {step.get('action', '')}\n"
                    f"Action Input: {step.get('action_input', '')}\n"
                    f"Observation: {step.get('observation', '')}"
                    for step in steps
                ])
                
                prompt = f"""
你是一个智能助手，正在回答用户的问题：{input_text}

以下是你之前的推理过程：
{history}

{self.tool_descriptions}

基于以上信息，继续你的推理。如果你已经有足够信息，请给出最终答案。

请按照以下格式继续：
Thought: 你的思考过程
Action: 选择一个工具名称 (如果需要)
Action Input: 工具的输入参数 (如果需要)

或者如果可以给出最终答案：
Thought: 我现在可以回答这个问题
Final Answer: 你的最终答案
"""
            
            try:
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # 解析响应
                thought, action, action_input, final_answer = self._parse_llm_response(content)
                
                if self.verbose:
                    print(f"🤔 Thought: {thought}")
                
                return {
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "final_answer": final_answer,
                    "iteration": iteration + 1
                }
                
            except Exception as e:
                return {
                    "thought": f"思考过程出现错误: {e}",
                    "action": "",
                    "action_input": "",
                    "final_answer": "",
                    "iteration": iteration + 1
                }
        
        def act_node(state: ReActState) -> Dict[str, Any]:
            """行动节点"""
            action = state["action"]
            action_input = state["action_input"]
            thought = state["thought"]
            
            if not action or action not in self.tools:
                observation = f"无效的工具: {action}. 可用工具: {list(self.tools.keys())}"
            else:
                try:
                    if self.verbose:
                        print(f"🎬 Action: {action}")
                        print(f"📝 Input: {action_input}")
                    
                    # 执行工具
                    tool = self.tools[action]
                    observation = tool.run(action_input)
                    
                    if self.verbose:
                        print(f"👁️ Observation: {observation}")
                        
                except Exception as e:
                    observation = f"工具执行错误: {e}"
            
            # 记录步骤
            step = {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation
            }
            
            return {
                "observation": observation,
                "steps": [step]
            }
        
        def should_continue(state: ReActState) -> str:
            """决定是否继续ReAct循环"""
            # 如果有最终答案，结束循环
            if state.get("final_answer"):
                return "end"
            
            # 如果达到最大迭代次数，强制结束
            if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
                return "end"
            
            # 如果没有行动，说明需要继续思考
            if not state.get("action"):
                return "think"
            
            # 否则执行行动
            return "act"
        
        def finalize_node(state: ReActState) -> Dict[str, Any]:
            """最终化节点"""
            final_answer = state.get("final_answer")
            
            if not final_answer:
                # 如果没有最终答案但循环结束了，生成一个答案
                steps = state.get("steps", [])
                if steps:
                    last_observation = steps[-1].get("observation", "")
                    final_answer = f"基于观察结果，我的回答是: {last_observation}"
                else:
                    final_answer = "抱歉，我无法回答这个问题。"
            
            if self.verbose:
                print(f"🎯 Final Answer: {final_answer}")
            
            return {
                "final_answer": final_answer
            }
        
        # 构建图
        workflow = StateGraph(ReActState)
        
        # 添加节点
        workflow.add_node("think", think_node)
        workflow.add_node("act", act_node)
        workflow.add_node("finalize", finalize_node)
        
        # 设置入口点
        workflow.set_entry_point("think")
        
        # 添加条件路由
        workflow.add_conditional_edges(
            "think",
            should_continue,
            {
                "act": "act",
                "end": "finalize",
                "think": "think"  # 继续思考
            }
        )
        
        workflow.add_conditional_edges(
            "act", 
            should_continue,
            {
                "think": "think",
                "end": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _parse_llm_response(self, content: str) -> tuple:
        """解析LLM响应"""
        thought = ""
        action = ""
        action_input = ""
        final_answer = ""
        
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                action_input = line[13:].strip()
            elif line.startswith("Final Answer:"):
                final_answer = line[13:].strip()
        
        return thought, action, action_input, final_answer
    
    def run(self, input_text: str) -> Dict[str, Any]:
        """运行ReAct Agent"""
        initial_state = {
            "input": input_text,
            "thought": "",
            "action": "",
            "action_input": "",
            "observation": "",
            "final_answer": "",
            "steps": [],
            "iteration": 0,
            "max_iterations": self.max_iterations
        }
        
        result = self.graph.invoke(initial_state)
        return result


# 预定义的工具示例
def create_phm_tools() -> List[Tool]:
    """创建PHM相关的工具"""
    
    def check_sensor_status(sensor_name: str) -> str:
        """检查传感器状态"""
        sensor_data = {
            "temperature": {"status": "normal", "value": 65.2, "unit": "°C"},
            "vibration": {"status": "warning", "value": 7.8, "unit": "mm/s"},
            "pressure": {"status": "normal", "value": 2.3, "unit": "bar"},
            "flow": {"status": "critical", "value": 0.2, "unit": "m³/h"}
        }
        
        if sensor_name.lower() in sensor_data:
            data = sensor_data[sensor_name.lower()]
            return f"{sensor_name}传感器状态: {data['status']}, 当前值: {data['value']}{data['unit']}"
        else:
            return f"未找到传感器: {sensor_name}。可用传感器: {list(sensor_data.keys())}"
    
    def get_maintenance_history(equipment_id: str) -> str:
        """获取设备维护历史"""
        history = {
            "pump001": "最后维护: 2024-01-15, 类型: 预防性维护, 下次计划: 2024-04-15",
            "motor002": "最后维护: 2024-01-20, 类型: 故障维修, 下次计划: 2024-03-20",
            "bearing003": "最后维护: 2023-12-10, 类型: 润滑保养, 下次计划: 2024-03-10"
        }
        
        return history.get(equipment_id, f"未找到设备 {equipment_id} 的维护记录")
    
    def calculate_remaining_life(equipment_id: str, current_condition: str) -> str:
        """计算剩余使用寿命"""
        conditions = {
            "good": "预估剩余寿命: 18-24个月",
            "fair": "预估剩余寿命: 6-12个月", 
            "poor": "预估剩余寿命: 1-3个月",
            "critical": "建议立即更换"
        }
        
        return conditions.get(current_condition.lower(), "无法评估，条件信息不足")
    
    def search_fault_database(symptom: str) -> str:
        """搜索故障数据库"""
        fault_db = {
            "高温": "可能原因: 润滑不足、轴承磨损、过载运行。建议: 检查润滑系统，测量轴承间隙。",
            "振动": "可能原因: 不平衡、对中不良、轴承故障。建议: 进行振动分析，检查安装精度。",
            "噪音": "可能原因: 齿轮磨损、轴承损坏、松动。建议: 声学检测，紧固检查。",
            "泄漏": "可能原因: 密封件老化、压力过高、安装不当。建议: 更换密封件，检查系统压力。"
        }
        
        for key, value in fault_db.items():
            if key in symptom:
                return value
        
        return "未找到匹配的故障模式。请提供更具体的症状描述。"
    
    # 创建工具列表
    tools = [
        Tool(
            name="check_sensor_status",
            description="检查指定传感器的状态和当前值。输入传感器名称(temperature/vibration/pressure/flow)",
            func=check_sensor_status
        ),
        Tool(
            name="get_maintenance_history", 
            description="获取设备的维护历史记录。输入设备ID(如pump001, motor002等)",
            func=get_maintenance_history
        ),
        Tool(
            name="calculate_remaining_life",
            description="根据当前状态计算设备剩余使用寿命。输入设备状态(good/fair/poor/critical)",
            func=calculate_remaining_life
        ),
        Tool(
            name="search_fault_database",
            description="在故障数据库中搜索相关信息。输入故障症状关键词",
            func=search_fault_database
        )
    ]
    
    return tools


def demo_react_agent():
    """演示ReAct Agent的使用"""
    from ..Part1_Foundations.modules.llm_providers_unified import create_llm
    
    print("🤖 ReAct Agent演示")
    print("=" * 50)
    
    # 创建LLM和工具
    llm = create_llm("mock", temperature=0.1)
    tools = create_phm_tools()
    
    # 创建ReAct Agent
    react_agent = ReActAgent(llm, tools, max_iterations=5, verbose=True)
    
    # 测试问题
    test_questions = [
        "pump001设备出现高温问题，请帮我诊断并给出建议",
        "检查所有传感器状态，判断系统是否正常",
        "motor002需要维护吗？请检查其维护历史"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"🔍 测试问题 {i}: {question}")
        print(f"{'='*60}")
        
        result = react_agent.run(question)
        
        print(f"\n📋 最终答案:")
        print(result["final_answer"])
        
        print(f"\n📊 统计信息:")
        print(f"  迭代次数: {result['iteration']}")
        print(f"  步骤数: {len(result['steps'])}")
        
        if result['steps']:
            print(f"\n🔄 执行步骤:")
            for j, step in enumerate(result['steps'], 1):
                print(f"  {j}. 思考: {step['thought'][:50]}...")
                print(f"     行动: {step['action']} -> {step['observation'][:50]}...")


if __name__ == "__main__":
    demo_react_agent()