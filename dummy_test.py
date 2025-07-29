from __future__ import annotations
import uuid
import numpy as np
from typing import Dict

# 修正导入，确保所有需要的模型都被正确引入
from src.phm_outer_graph import build_outer_graph
from src.states.phm_states import PHMState, DAGState, InputData

def load_signal_data(path: str) -> Dict[str, np.ndarray]:
    """
    一个加载信号数据的辅助函数。
    为演示目的，生成包含四种不同状态信号的表格化数据（字典）。
    """
    print(f"Generating dummy signal data for 4 states (ignoring path: {path})")
    
    states = ["normal", "inner_race_fault", "outer_race_fault", "ball_fault"]
    signal_length = 8192
    num_channels = 1
    
    signals = {}
    for state in states:
        # 基础正弦波
        base_signal = np.sin(np.linspace(0, 100, signal_length))
        # 添加高斯噪声
        noise = np.random.randn(signal_length) * 0.5
        
        # 为不同故障类型添加特征
        if state == "inner_race_fault":
            impulses = np.zeros(signal_length)
            for i in range(10):
                impulses[np.random.randint(0, signal_length)] = 2.0
            signal = base_signal + noise + impulses
        elif state == "outer_race_fault":
            impulses = np.zeros(signal_length)
            for i in range(0, signal_length, signal_length // 15):
                impulses[i] = 1.5
            signal = base_signal + noise + impulses
        elif state == "ball_fault":
            impulses = np.zeros(signal_length)
            for i in range(5):
                impulses[np.random.randint(0, signal_length)] = 2.5
            signal = base_signal + noise + impulses
        else: # normal
            signal = base_signal + noise
            
        # 统一格式为 (B, L, C)，其中 B=1
        signals[state] = signal.reshape(1, signal_length, num_channels)
        
    return signals

def initialize_state(user_instruction: str, ref_signal_path: str, test_signal_path: str) -> PHMState:
    """
    根据初始输入，创建并初始化整个系统的状态（PHMState）。
    这对应您描述的 "START -> 接收 llm指令 | 参考信号 | 测试信号" 阶段。
    """
    # 1. 加载信号数据
    # 注意：load_signal_data 返回的是一个包含4种状态的字典
    all_ref_data = load_signal_data(ref_signal_path)
    all_test_data = load_signal_data(test_signal_path)

    ref_data_node = InputData(
        node_id="ref_signal_node",
        data=all_ref_data,
        parents=[],
        shape=next(iter(all_ref_data.values())).shape  # 获取第一个信号的形状
    )
    test_data_node = InputData(
        node_id="test_signal_node",
        data=all_test_data,
        parents=[],
        shape=next(iter(all_test_data.values())).shape  # 获取第一个信号的形状
    )
    # 从字典中选取一个作为代表，在实际应用中应有更明确的选择逻辑
    ref_data_array = next(iter(all_ref_data.values()))
    test_data_array = next(iter(all_test_data.values()))

    # 4. 创建 DAGState，这是状态的核心。
    #    我们为根节点定义清晰的ID，并用 InputData 对象填充初始节点。
    ref_root_id = "ref_root_node_01"
    test_root_id = "test_root_node_01"
    
    # 修正：在创建 InputData 时提供所有必需的字段
    # DAGState 节点存储实际的数据信息，不能再嵌套 InputData
    # 因此直接复用上面创建的 InputData 对象作为根节点
    initial_nodes = {
        ref_root_id: InputData(
            node_id=ref_root_id,
            data=all_ref_data,
            parents=[],
            shape=ref_data_array.shape,
        ),
        test_root_id: InputData(
            node_id=test_root_id,
            data=all_test_data,
            parents=[],
            shape=test_data_array.shape,
        ),
    }

    dag_state = DAGState(
        user_instruction=user_instruction,
        reference_root=ref_root_id,
        test_root=test_root_id,
        nodes=initial_nodes,
        leaves=[ref_root_id, test_root_id]
    )

    # 5. 创建完整的 PHMState 对象
    #    - dag_tracker 留空 (None)，让 PHMState.tracker() 方法按需创建
    #    - 其他字段根据其类型正确填充
    initial_state = PHMState(
        user_instruction=user_instruction,
        reference_signal=ref_data_node,
        test_signal=test_data_node,
        dag_state=dag_state,
        dag_tracker=None,
    )
    return initial_state

def main():
    """
    程序主入口，执行一次完整的故障诊断流程。
    """
    # 1. 定义初始输入
    user_instruction = "Analyze the bearing signal for potential faults by comparing its frequency spectrum and statistical features against the reference signal."
    ref_signal_path = "data/healthy_bearing.csv"
    test_signal_path = "data/faulty_bearing.csv"

    # 2. 初始化状态
    initial_phm_state = initialize_state(user_instruction, ref_signal_path, test_signal_path)

    # 3. 构建并编译外层图
    app = build_outer_graph()

    # 4. 启动图的运行
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n--- Starting PHM Analysis Workflow ---\n")
    final_state = None
    for event in app.stream(initial_phm_state, config=config):
        for node_name, state_update in event.items():
            print(f"--- Executing Node: {node_name} ---")
            print("...done.\n")
            final_state = state_update

    # 5. 获取最终结果（直接使用最后一次状态）
    print("--- Workflow Finished ---")
    if isinstance(final_state, dict):
        final_state = PHMState.model_validate(final_state)

    if final_state and final_state.final_report:
        print("\nFinal Report:")
        print(final_state.final_report)
    else:
        print("No final report was generated or an error occurred.")


if __name__ == "__main__":
    main()

