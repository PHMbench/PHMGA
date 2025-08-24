"""
状态管理模块

本模块展示如何在Graph Agent中进行状态管理，包括：
- TypedDict状态定义
- Annotated类型的使用
- 状态更新和合并机制
"""

from typing import Dict, Any, List, TypedDict, Annotated
import operator
import time


class BasicState(TypedDict):
    """基础状态示例"""
    current_step: str
    messages: Annotated[List[str], operator.add]  # 自动累加列表
    sensor_data: Dict[str, float]
    analysis_results: Dict[str, Any]
    confidence: float


class PHMDiagnosisState(TypedDict):
    """PHM诊断状态定义"""
    equipment_id: str
    sensor_readings: Dict[str, float]
    analysis_history: Annotated[List[str], operator.add]
    current_diagnosis: str
    severity_level: str
    recommended_actions: List[str]
    processing_time: float


def demo_state_management():
    """演示状态管理基础"""
    print("🔄 状态管理演示")
    
    # 初始状态
    initial_state = {
        "current_step": "initialization",
        "messages": ["系统启动"],
        "sensor_data": {"temperature": 65.2, "vibration": 3.1},
        "analysis_results": {},
        "confidence": 0.0
    }
    
    print(f"初始状态: {initial_state}")
    
    # 状态更新
    update1 = {
        "current_step": "analysis",
        "messages": ["开始数据分析"],  # 会自动累加到现有messages
        "analysis_results": {"frequency_domain": "正常"}
    }
    
    # 模拟LangGraph的状态合并
    merged_state = simulate_state_merge(initial_state, update1)
    
    print(f"\n更新后状态:")
    for key, value in merged_state.items():
        print(f"  {key}: {value}")
    
    return merged_state


def simulate_state_merge(current_state: Dict, update: Dict) -> Dict:
    """模拟LangGraph的状态合并逻辑"""
    merged_state = current_state.copy()
    
    for key, value in update.items():
        if key == "messages":
            # Annotated[List, operator.add] 的行为：累加列表
            merged_state[key].extend(value)
        elif key == "analysis_results":
            # 字典类型：更新合并
            merged_state[key].update(value)
        else:
            # 其他类型：直接替换
            merged_state[key] = value
    
    return merged_state


class StateManager:
    """状态管理器"""
    
    def __init__(self):
        self.state_history = []
        self.current_state = {}
    
    def update_state(self, updates: Dict[str, Any]):
        """更新状态"""
        # 保存历史
        self.state_history.append(self.current_state.copy())
        
        # 更新当前状态
        self.current_state = simulate_state_merge(self.current_state, updates)
        
        return self.current_state
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """获取状态快照"""
        return {
            "current": self.current_state.copy(),
            "history_length": len(self.state_history),
            "timestamp": time.time()
        }
    
    def rollback_state(self, steps: int = 1) -> Dict[str, Any]:
        """回滚状态"""
        if len(self.state_history) >= steps:
            for _ in range(steps):
                self.current_state = self.state_history.pop()
        return self.current_state


def create_phm_diagnosis_state(equipment_id: str, sensor_data: Dict[str, float]) -> PHMDiagnosisState:
    """创建PHM诊断状态"""
    return {
        "equipment_id": equipment_id,
        "sensor_readings": sensor_data,
        "analysis_history": [f"开始诊断设备 {equipment_id}"],
        "current_diagnosis": "",
        "severity_level": "unknown",
        "recommended_actions": [],
        "processing_time": 0.0
    }


def demo_phm_state_workflow():
    """演示PHM状态工作流"""
    print("\n🏭 PHM状态工作流演示")
    
    # 创建初始状态
    state = create_phm_diagnosis_state(
        "PUMP-001", 
        {"temperature": 85.5, "vibration": 6.2, "pressure": 1.8}
    )
    
    print(f"初始PHM状态: {state['equipment_id']}")
    
    # 状态管理器
    manager = StateManager()
    manager.current_state = state
    
    # 模拟诊断流程
    steps = [
        {
            "analysis_history": ["数据预处理完成"],
            "current_diagnosis": "检测到高温和异常振动",
            "severity_level": "high"
        },
        {
            "analysis_history": ["深度分析完成"],
            "recommended_actions": ["立即停机", "检查轴承", "更换润滑油"],
            "processing_time": 2.35
        }
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"\n步骤 {i}: {list(step.keys())}")
        updated_state = manager.update_state(step)
        
    # 最终状态
    final_snapshot = manager.get_state_snapshot()
    final_state = final_snapshot["current"]
    
    print(f"\n✅ 诊断完成:")
    print(f"  设备: {final_state['equipment_id']}")
    print(f"  诊断: {final_state['current_diagnosis']}")
    print(f"  严重程度: {final_state['severity_level']}")
    print(f"  建议行动: {final_state['recommended_actions']}")
    print(f"  处理时间: {final_state['processing_time']:.2f}s")
    print(f"  分析历史: {len(final_state['analysis_history'])} 条记录")
    
    return final_state


if __name__ == "__main__":
    # 运行演示
    demo_state_management()
    demo_phm_state_workflow()