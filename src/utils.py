from __future__ import annotations
from typing import Any, Dict, List, Tuple
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from src.states.phm_states import PHMState, DAGState, InputData
import pandas as pd
import h5py
import numpy as np
import os
import pickle
import uuid

# Load environment variables from .env file first
from dotenv import load_dotenv
load_dotenv()

# Configure LangSmith settings using unified state management
try:
    from src.states.phm_states import get_unified_state
    unified_state = get_unified_state()

    # Apply system configuration from unified state
    os.environ["LANGCHAIN_TRACING_V2"] = str(unified_state.get('system.langchain_tracing', False)).lower()
    os.environ["LANGCHAIN_ENDPOINT"] = unified_state.get('system.langchain_endpoint', "")
    os.environ["LANGCHAIN_API_KEY"] = unified_state.get('system.langchain_api_key', "")
    os.environ["LANGCHAIN_PROJECT"] = unified_state.get('system.langchain_project', "")
except ImportError:
    # Fallback to legacy configuration
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_ENDPOINT"] = ""
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = ""

# 导入解耦后的两个图构建器
from src.phm_outer_graph import build_builder_graph, build_executor_graph

def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segment_string' (the marker to insert).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


def get_citations(response, resolved_urls_map):
    """
    Extracts and formats citation information from a Gemini model's response.

    This function processes the grounding metadata provided in the response to
    construct a list of citation objects. Each citation object includes the
    start and end indices of the text segment it refers to, and a string
    containing formatted markdown links to the supporting web chunks.

    Args:
        response: The response object from the Gemini model, expected to have
                  a structure including `candidates[0].grounding_metadata`.
                  It also relies on a `resolved_map` being available in its
                  scope to map chunk URIs to resolved URLs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text. Defaults to 0
                                     if not specified.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list[str]): A list of individual markdown-formatted
                                        links for each grounding chunk.
              - "segment_string" (str): A concatenated string of all markdown-
                                        formatted links for the citation.
              Returns an empty list if no valid candidates or grounding supports
              are found, or if essential data is missing.
    """
    citations = []

    # Ensure response and necessary nested structures are present
    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}

        # Ensure segment information is present
        if not hasattr(support, "segment") or support.segment is None:
            continue  # Skip this support if segment info is missing

        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        # Ensure end_index is present to form a valid segment
        if support.segment.end_index is None:
            continue  # Skip if end_index is missing, as it's crucial

        # Add 1 to end_index to make it an exclusive end for slicing/range purposes
        # (assuming the API provides an inclusive end_index)
        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    citation["segments"].append(
                        {
                            "label": chunk.web.title.split(".")[:-1][0],
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    # Handle cases where chunk, web, uri, or resolved_map might be problematic
                    # For simplicity, we'll just skip adding this particular segment link
                    # In a production system, you might want to log this.
                    pass
        citations.append(citation)
    return citations


def dag_to_llm_payload(state: PHMState, max_nodes: int = 40) -> str:
    """Return a JSON string representing the latest portion of the DAG.

    Parameters
    ----------
    state : PHMState
        State whose internal DAG should be exported.
    max_nodes : int, optional
        Maximum number of nodes to include from the tail of the DAG.

    Returns
    -------
    str
        JSON payload for use in LLM prompts.
    """
    return state.tracker().export_json(max_nodes=max_nodes)


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
    user_instruction: str,
    metadata_path: str,
    h5_path: str,
    ref_ids: list[int],
    test_ids: list[int],
    case_name: str,
    fs: float | None = None,
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

        meta = {
            "channel": channel_name,
            "labels": all_labels,  # 所有标签信息都附加到每个通道节点
        }
        if fs is not None:
            meta["fs"] = fs

        node = InputData(
            node_id=channel_name,
            data={},
            results={"ref": channel_ref_signals, "tst": channel_test_signals},
            parents=[],
            shape=first_sig_shape,
            meta=meta,
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

    return PHMState(
        case_name=case_name,
        user_instruction=user_instruction,
        reference_signal=next(iter(nodes.values())),
        test_signal=next(iter(nodes.values())),
        dag_state=dag_state,
        fs=fs,
    )


def generate_final_report(final_state, report_path: str):
    """
    保存最终的报告。
    """
    print("\n--- Workflow Finished ---")
    if isinstance(final_state, dict) and final_state.get("final_report"):
        report = final_state["final_report"]
        print("\nFinal Report:")
        print(report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {report_path}")
    else:
        print("No final report was generated or an error occurred.")
        if final_state and 'dag_state' in final_state and final_state['dag_state'].error_log:
            print("Errors during execution:", final_state['dag_state'].error_log)


def save_state(state, filepath: str):
    """
    使用pickle将状态对象保存到磁盘。
    """
    try:
        print(f"\n--- Saving state to {filepath} ---")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print("...done.")
        return True
    except Exception as e:
        print(f"Error saving state: {e}")
        return False

def load_state(filepath: str):
    """
    使用pickle从磁盘加载状态对象。
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

