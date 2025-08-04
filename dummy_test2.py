from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
import h5py
from typing import Dict, Tuple
from dotenv import load_dotenv
import os
import pickle

# 禁用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

load_dotenv()

# 导入解耦后的两个图构建器
from src.phm_outer_graph import build_builder_graph, build_executor_graph
from src.states.phm_states import PHMState, DAGState, InputData

# --- Helper Functions (reused from dummy_test1.py) ---

def load_signal_data(metadata_path: str, h5_path: str, ids_to_load: list[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    从真实的 metadata 和 HDF5 文件中加载信号数据和标签。
    返回两个字典:
    1. signals: {'id': signal_array}
    2. labels: {'id': label}
    """
    print(f"Loading data for IDs: {ids_to_load}")
    
    try:
        metadata_df = pd.read_excel(metadata_path)
        h5_file = h5py.File(h5_path, 'r')
    except Exception as e:
        print(f"Error loading data files: {e}")
        return {}, {}

    signals = {}
    labels = {}
    for sample_id in ids_to_load:
        sample_info = metadata_df[metadata_df['Id'] == sample_id]
        if sample_info.empty:
            print(f"Warning: ID {sample_id} not found in metadata.")
            continue

        label = sample_info['Label'].iloc[0]
        sample_length = int(sample_info['Sample_lenth'].iloc[0])
        num_channels = int(sample_info['Channel'].iloc[0])

        try:
            signal_data = h5_file[str(sample_id)][()]
            signal_data = np.squeeze(signal_data)
            
            if signal_data.shape == (sample_length, num_channels):
                signals[str(sample_id)] = signal_data.reshape(1, sample_length, num_channels)
                labels[str(sample_id)] = label
            else:
                print(f"Warning: Shape mismatch for ID {sample_id}. Expected {(sample_length, num_channels)}, got {signal_data.shape}")

        except KeyError:
            print(f"Warning: ID {sample_id} not found in HDF5 file.")
    
    h5_file.close()
    return signals, labels


def initialize_state(
    user_instruction: str, metadata_path: str, h5_path: str, ref_ids: list[int], test_ids: list[int]
) -> PHMState:
    """
    根据初始输入，创建并初始化整个系统的状态（PHMState）。
    为每个物理信号通道创建一个初始节点，并将所有信号按通道分配。
    """
    ref_signals, ref_labels = load_signal_data(metadata_path, h5_path, ref_ids)
    test_signals, test_labels = load_signal_data(metadata_path, h5_path, test_ids)

    if not ref_signals or not test_signals:
        raise ValueError("Failed to load reference or test signals.")

    all_labels = {**ref_labels, **test_labels}
    
    # --- 确定通道数 ---
    # 从第一个加载的信号中推断出通道数
    first_sig_array = next(iter(ref_signals.values()))
    num_channels = first_sig_array.shape[2] # Shape is (B, L, C)
    channel_names = [f"ch{i+1}" for i in range(num_channels)]
    
    nodes = {}
    leaves = []
    
    for i, channel_name in enumerate(channel_names):
        # 为当前通道提取所有信号
        channel_ref_signals = {sig_id: sig[:, :, i:i+1] for sig_id, sig in ref_signals.items()}
        channel_test_signals = {sig_id: sig[:, :, i:i+1] for sig_id, sig in test_signals.items()}
        
        first_sig_shape = next(iter(channel_ref_signals.values())).shape

        node = InputData(
            node_id=channel_name,
            data={},
            results={"ref": channel_ref_signals, "tst": channel_test_signals},
            parents=[],
            shape=first_sig_shape,
            meta={
                "channel": channel_name,
                "labels": all_labels # 所有标签信息都附加到每个通道节点
            }
        )
        nodes[channel_name] = node
        leaves.append(channel_name)

    if not nodes:
        raise ValueError("No valid nodes could be created from the provided data.")

    # dag_state.channels should be the physical channel names for the planner
    dag_state = DAGState(
        user_instruction=user_instruction,
        nodes=nodes,
        leaves=leaves,
        channels=channel_names # Use physical channel names
    )

    initial_state = PHMState(
        user_instruction=user_instruction,
        reference_signal=next(iter(nodes.values())),
        test_signal=next(iter(nodes.values())),
        dag_state=dag_state,
    )
    return initial_state


def generate_final_report(final_state, save_dir="/home/lq/LQcode/2_project/PHMBench/PHMGA/save/"):
    """
    保存最终的报告。
    """
    print("\n--- Workflow Finished ---")
    if isinstance(final_state, dict) and final_state.get("final_report"):
        report = final_state["final_report"]
        print("\nFinal Report:")
        print(report)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "final_report_test2.md"), "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {os.path.join(save_dir, 'final_report_test2.md')}")
    else:
        print("No final report was generated or an error occurred.")
        if final_state and 'dag_state' in final_state and final_state['dag_state'].error_log:
            print("Errors during execution:", final_state['dag_state'].error_log)

# --- Main Execution Logic ---

def main():
    """
    程序主入口，分两步执行故障诊断流程：
    1. 运行DAG构建图，生成计算流程。
    2. 运行DAG执行图，根据生成的流程进行分析和报告。
    """
    # 1. 定义初始输入
    user_instruction = "Analyze the bearing signals for potential faults. The reference set contains signals for 5 different states (health, ball, cage, inner, outer). The test set also contains signals for the same 5 states. The goal is to correctly classify each test signal by comparing it to the reference set."
    metadata_path = "/mnt/crucial/LQ/PHM-Vibench/metadata_6_11.xlsx"
    h5_path = "/mnt/crucial/LQ/PHM-Vibench/cache.h5"
    ########################################################
    # # ref_ids = [47050, 47044, 47047, 47053, 47056]
    # # test_ids = [47051, 47045, 47048, 47054, 47057]
    # ref_ids = [47050]
    # test_ids = [47051]  # 只测试一个信号以简化调试
    
    # # 2. 初始化状态
    # initial_phm_state = initialize_state(
    #     user_instruction, metadata_path, h5_path, ref_ids, test_ids
    # )

    # # --- Part 1: 运行DAG构建图 ---
    # print("\n--- [Part 1] Starting DAG Builder Workflow ---")
    # builder_app = build_builder_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # # Initialize built_state with a copy of the initial state
    # built_state = initial_phm_state.model_copy(deep=True)
    # for event in builder_app.stream(initial_phm_state, config=config):
    #     for node_name, state_update in event.items():
    #         print(f"--- Builder Node Executed: {node_name} ---")
    #         # Merge the updates into the current state
    #         if state_update is not None:
    #             for key, value in state_update.items():
    #                 setattr(built_state, key, value)

    # print("\n--- [Part 1] DAG Builder Workflow Finished ---")
    # if not built_state:
    #     print("Builder workflow failed to produce a final state.")
    #     return
        
    # print(f"Final leaves of the built DAG: {built_state.dag_state.leaves}")
    # print(f"Total nodes in DAG: {len(built_state.dag_state.nodes)}")
    # print(f"Errors during build: {built_state.dag_state.error_log}")
#####################################################################
    # def save_state(state, filepath):
    #     """
    #     Save a state object to disk using pickle.
        
    #     Args:
    #         state: The state object to save
    #         filepath: Path where the state will be saved
        
    #     Returns:
    #         bool: True if save was successful, False otherwise
    #     """
    #     try:
    #         print(f"\n--- Saving state to {filepath} ---")
    #         os.makedirs(os.path.dirname(filepath), exist_ok=True)
    #         with open(filepath, "wb") as f:
    #             pickle.dump(state, f)
    #         print("...done.")
    #         return True
    #     except Exception as e:
    #         print(f"Error saving state: {e}")
    #         return False

    def load_state(filepath):
        """
        Load a state object from disk using pickle.
        
        Args:
            filepath: Path where the state is saved
        
        Returns:
            The loaded state object or None if loading failed
        """
        try:
            print(f"\n--- Loading state from {filepath} ---")
            with open(filepath, "rb") as f:
                state = pickle.load(f)
            print("...done.")
            print(f"Successfully loaded state with {len(state.dag_state.nodes)} nodes.")
            return state
        except Exception as e:
            print(f"Error loading state: {e}")
            return None

    # --- Save the built state ---
    state_save_path = "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/built_state.pkl"
    # save_state(built_state, state_save_path)

    # --- Load the built state ---
    loaded_state = load_state(state_save_path)
    if loaded_state is None:
        print("Failed to load state. Exiting.")
        exit(1)


    # --- Part 2: 运行DAG执行图 ---
    print("\n--- [Part 2] Starting DAG Executor Workflow ---")
    executor_app = build_executor_graph()
    
    # Initialize final_state with a copy of the loaded state
    final_state = loaded_state.model_copy(deep=True)
    for event in executor_app.stream(loaded_state, config=config):
        for node_name, state_update in event.items():
            print(f"--- Executor Node Executed: {node_name} ---")
            # Merge the updates into the final state
            if state_update is not None:
                for key, value in state_update.items():
                    setattr(final_state, key, value)

    # 3. 生成最终报告
    generate_final_report(final_state)


if __name__ == "__main__":
    main()
