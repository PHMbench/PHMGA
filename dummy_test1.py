from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
import h5py
from typing import Dict
from dotenv import load_dotenv
import os

# 禁用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

load_dotenv()
# 修正导入，确保所有需要的模型都被正确引入
from src.phm_outer_graph import build_outer_graph
from src.states.phm_states import PHMState, DAGState, InputData

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

def load_signal_data(metadata_path: str, h5_path: str, ids_to_load: list[int]) -> Dict[str, np.ndarray]:
    """
    从真实的 metadata 和 HDF5 文件中加载信号数据。
    """
    print(f"Loading data for IDs: {ids_to_load}")
    
    try:
        metadata_df = pd.read_excel(metadata_path)
        h5_file = h5py.File(h5_path, 'r')
    except Exception as e:
        print(f"Error loading data files: {e}")
        return {}

    signals = {}
    for sample_id in ids_to_load:
        # 在元数据中查找ID
        sample_info = metadata_df[metadata_df['Id'] == sample_id]
        if sample_info.empty:
            print(f"Warning: ID {sample_id} not found in metadata.")
            continue

        label_desc = sample_info['Label_Description'].iloc[0]
        sample_length = int(sample_info['Sample_lenth'].iloc[0])
        num_channels = int(sample_info['Channel'].iloc[0])

        # 从HDF5文件加载信号数据
        try:
            signal_data = h5_file[str(sample_id)][()]
            # 移除多余的单维度
            signal_data = np.squeeze(signal_data)
            # 确保形状为 (B, L, C)
            if signal_data.shape == (sample_length, num_channels):
                 signals[label_desc] = signal_data.reshape(1, sample_length, num_channels)
            else:
                print(f"Warning: Shape mismatch for ID {sample_id}. Expected {(sample_length, num_channels)}, got {signal_data.shape}")

        except KeyError:
            print(f"Warning: ID {sample_id} not found in HDF5 file.")
    
    h5_file.close()
    return signals


def initialize_state(
    user_instruction: str, metadata_path: str, h5_path: str, ref_ids: list[int], test_ids: list[int]
) -> PHMState:
    """
    根据初始输入，创建并初始化整个系统的状态（PHMState）。
    为每个信号通道（如 "Health", "Ball_fault"）创建一个包含ref和test信号的初始节点。
    """
    ref_signals = load_signal_data(metadata_path, h5_path, ref_ids)
    test_signals = load_signal_data(metadata_path, h5_path, test_ids)

    if not ref_signals or not test_signals:
        raise ValueError("Failed to load reference or test signals.")

    nodes = {}
    leaves = []
    all_channels = sorted(list(test_signals.keys()))

    for channel_name in all_channels:
        ref_sig = ref_signals.get(channel_name)
        test_sig = test_signals.get(channel_name)

        if ref_sig is None or test_sig is None:
            print(f"Warning: Missing ref or test signal for channel {channel_name}")
            continue

        node_id = channel_name.replace(' ', '_')
        node = InputData(
            node_id=node_id,
            data={}, # data字段可以留空或用于其他元数据
            results={"ref": ref_sig, "tst": test_sig}, # 将ref和test信号存储在results中
            parents=[],
            shape=ref_sig.shape,
            meta={"channel": channel_name}
        )
        nodes[node_id] = node
        leaves.append(node_id)

    dag_state = DAGState(
        user_instruction=user_instruction,
        nodes=nodes,
        leaves=leaves,
        channels=all_channels
    )

    # PHMState中的reference_signal和test_signal字段需要一个代表性对象
    initial_state = PHMState(
        user_instruction=user_instruction,
        reference_signal=next(iter(nodes.values())),
        test_signal=next(iter(nodes.values())),
        dag_state=dag_state,
    )
    return initial_state


def main():
    """
    程序主入口，执行一次完整的故障诊断流程。
    """
    # 1. 定义初始输入
    user_instruction = "Analyze the bearing signals for potential faults. The reference set contains signals for 5 different states (health, ball, cage, inner, outer). The test set also contains signals for the same 5 states. The goal is to correctly classify each test signal by comparing it to the reference set."
    metadata_path = "/mnt/crucial/LQ/PHM-Vibench/metadata_6_11.xlsx"
    h5_path = "/mnt/crucial/LQ/PHM-Vibench/cache.h5"
    
    # 定义要测试的ID
    ref_ids = [
        47050,  # health/15hz_health_400s_1.csv
        47044,  # ball/15hz_ball10_400s_1.csv
        47047,  # cage/15hz_cage_400s_1.csv
        47053,  # inner/15hz_inner_400s_1.csv
        47056,  # outer/15hz_outer_400s_1.csv
    ]
    test_ids = [
        47051,  # health/15hz_health_400s_2.csv
        47045,  # ball/15hz_ball10_400s_2.csv
        47048,  # cage/15hz_cage_400s_2.csv
        47054,  # inner/15hz_inner_400s_2.csv
        47057,  # outer/15hz_outer_400s_2.csv
    ]


    # 2. 初始化状态
    initial_phm_state = initialize_state(
        user_instruction, metadata_path, h5_path, ref_ids, test_ids
    )

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
            # 打印出状态更新中的关键信息以供调试
            if state_update is None:
                print("Warning: Received None state update.")
                continue
            if 'final_report' in state_update and state_update['final_report']:
                 print("Final report generated.")
            if 'error_logs' in state_update and state_update['error_logs']:
                 print(f"Errors: {state_update['error_logs']}")
            print("...done.\n")
            final_state = state_update

    # 5. 获取最终结果
    generate_final_report(final_state)

def generate_final_report(final_state, save_dir="/home/lq/LQcode/2_project/PHMBench/PHMGA/save/"):
    print("--- Workflow Finished ---")
    if isinstance(final_state, dict) and final_state.get("final_report"):
        report = final_state["final_report"]
        print("\nFinal Report:")
        print(report)
        # 确保保存目录存在
        import os
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "final_report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {os.path.join(save_dir, 'final_report.md')}")
    else:
        print("No final report was generated or an error occurred.")
        if final_state and 'error_logs' in final_state and final_state['error_logs']:
            print("Errors during execution:", final_state['error_logs'])


if __name__ == "__main__":
    main()
