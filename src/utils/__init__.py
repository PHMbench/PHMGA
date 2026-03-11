from __future__ import annotations
from typing import Any, Dict, List, Tuple
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from src.config import resolve_data_selection
from src.states.phm_states import (
    CANONICAL_SPLITS,
    PHMState,
    DAGState,
    InputData,
    ProcessedData,
    get_split_results,
    normalize_result_splits,
)
from src.tools.signal_processing_schemas import get_operator, MultiVariableOp
import numpy as np
import os
import pickle
import uuid
import hashlib
import json
import networkx as nx

# 禁用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

# Load environment variables from `.env` (best-effort; do not override existing env).
try:  # pragma: no cover
    from dotenv import load_dotenv

    repo_env = os.path.join(os.getcwd(), ".env")
    load_dotenv(dotenv_path=repo_env, override=False)
except Exception:
    pass

# 导入解耦后的两个图构建器
# from src.phm_outer_graph import build_builder_graph, build_executor_graph

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


def _get_results(node: InputData | ProcessedData) -> Dict[str, Any]:
    """Safely extract the results dictionary from a node."""
    return node.results or {}





def _execute_multi_variable_op(
    op: Any, parent_ids: List[str], nodes: Dict[str, InputData | ProcessedData]
) -> Dict[str, Any]:
    """Execute a MultiVariableOp on all parents and return canonical split outputs."""
    split_outputs: Dict[str, Any] = {}
    for split in CANONICAL_SPLITS:
        parent_split = {pid: get_split_results(_get_results(nodes[pid]), split) for pid in parent_ids}
        parent_split = {k: v for k, v in parent_split.items() if isinstance(v, dict)}
        if not parent_split:
            split_outputs[split] = {}
            continue
        signal_keys = list(next(iter(parent_split.values())).keys())
        split_results = {}
        for key in signal_keys:
            single_input = {
                pid: data.get(key) for pid, data in parent_split.items() if data.get(key) is not None
            }
            if len(single_input) == len(parent_ids):
                split_results[key] = op.execute(single_input)
        split_outputs[split] = split_results
    return normalize_result_splits(split_outputs)


def _execute_single_variable_op(
    op: Any, parent_id: str, nodes: Dict[str, InputData | ProcessedData]
) -> Dict[str, Any]:
    """Execute a single-variable operator on one parent node."""
    parent_results = _get_results(nodes[parent_id])
    split_outputs: Dict[str, Any] = {}
    for split in CANONICAL_SPLITS:
        split_in = get_split_results(parent_results, split)
        if isinstance(split_in, dict):
            split_outputs[split] = {key: op.execute(val) for key, val in split_in.items()} if split_in else {}
        else:
            split_outputs[split] = op.execute(split_in) if split_in is not None else {}
    return normalize_result_splits(split_outputs)





def load_signal_data(
    metadata_path: str, h5_path: str, ids_to_load: list[int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Any]:
    """
    从真实的 metadata 和 HDF5 文件中加载信号数据和标签。
    返回两个字典:
    1. signals: {'id': signal_array}
    2. labels: {'id': label}
    """
    print(f"Loading data for IDs: {ids_to_load}")
    
    try:
        import pandas as pd
        metadata_df = pd.read_excel(metadata_path)
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return {}, {}, None

    signals = {}
    labels = {}
    try:
        import h5py
        with h5py.File(h5_path, "r") as h5_file:
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
    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        return {}, {}, None

    return signals, labels, metadata_df


def apply_windowing(signals: Dict[str, np.ndarray], labels: Dict[str, str], window_size: int = 4096, overlap: int = 128) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Apply windowing to the signals with specified window size and overlap.
    Returns a new dictionary of windowed signals and corresponding labels.
    """
    windowed_signals = {}
    windowed_labels = {}
    
    for sig_id, sig_array in signals.items():
        B, L, C = sig_array.shape
        step = window_size - overlap
        num_windows = max(1, (L - overlap) // step)

        for w in range(1, num_windows):
            start = w * step
            end = start + window_size
            if end > L:
                break  # Avoid going out of bounds
            
            windowed_sig = sig_array[:, start:end, :]
            new_id = f"{sig_id}_w{w+1}"
            windowed_signals[new_id] = windowed_sig
            windowed_labels[new_id] = labels[sig_id]
    
    return windowed_signals, windowed_labels

def initialize_state(
    user_instruction: str,
    metadata_path: str,
    h5_path: str,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    case_name: str,
    use_window: bool = True,
    *,
    allow_test_labels_for_reporting: bool = False,
    train_backend: str = "shallow",
    model_config_path: str | None = None,
    save_dir: str | None = None,
    data_cfg: Dict[str, Any] | None = None,
) -> PHMState:
    """
    根据初始输入，创建并初始化整个系统的状态（PHMState）。
    为每个物理信号通道创建一个初始节点，并将所有信号按通道分配。
    """
    train_signals, train_labels, train_metadata = load_signal_data(metadata_path, h5_path, train_ids)
    val_signals, val_labels, _val_metadata = load_signal_data(metadata_path, h5_path, val_ids) if list(val_ids or []) else ({}, {}, {})
    test_signals, test_labels, test_metadata = load_signal_data(metadata_path, h5_path, test_ids)

    if use_window:
        # Apply windowing to the signals
        train_signals, train_labels = apply_windowing(train_signals, train_labels)
        if val_signals:
            val_signals, val_labels = apply_windowing(val_signals, val_labels)
        test_signals, test_labels = apply_windowing(test_signals, test_labels)

    if not train_signals:
        raise ValueError("Failed to load training signals.")

    # --- 确定通道数 ---
    # 从第一个加载的信号中推断出通道数
    first_sig_array = next(iter(train_signals.values()))
    num_channels = first_sig_array.shape[2] # Shape is (B, L, C)
    channel_names = [f"ch{i+1}" for i in range(num_channels)]
    
    nodes = {}
    leaves = []
    
    for i, channel_name in enumerate(channel_names):
        # 为当前通道提取所有信号
        channel_train_signals = {sig_id: sig[:, :, i:i+1] for sig_id, sig in train_signals.items()}
        channel_val_signals = {sig_id: sig[:, :, i:i+1] for sig_id, sig in val_signals.items()} if val_signals else {}
        channel_test_signals = {sig_id: sig[:, :, i:i+1] for sig_id, sig in test_signals.items()}
        
        first_sig_shape = next(iter(channel_train_signals.values())).shape

        node = InputData(
            node_id=channel_name,
            data={},
            results=normalize_result_splits(
                {
                    "train": channel_train_signals,
                    "val": channel_val_signals,
                    "test": channel_test_signals,
                }
            ),
            parents=[],
            shape=first_sig_shape,
            meta={
                "channel": channel_name,
                "labels_train": train_labels,
                "labels_val": val_labels,
                "labels_test": test_labels,
                "fs": train_metadata['Sample_rate'].iloc[0]  # 采样频率
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

    return PHMState(
        case_name=case_name,
        user_instruction=user_instruction,
        reference_signal=next(iter(nodes.values())),
        test_signal=next(iter(nodes.values())),
        dag_state=dag_state,
        labels_train=train_labels,
        labels_val=val_labels,
        labels_test=test_labels,
        allow_test_labels_for_reporting=allow_test_labels_for_reporting,
        train_backend=train_backend,
        model_config_path=model_config_path,
        save_dir=save_dir,
        data_cfg=dict(data_cfg or {}),
    )


def initialize_state_vibench(
    *,
    user_instruction: str,
    case_name: str,
    data_cfg: Dict[str, Any],
    allow_test_labels_for_reporting: bool = False,
    train_backend: str = "tspn",
    model_config_path: str | None = None,
    save_dir: str | None = None,
    max_preview_samples: int = 4,
) -> PHMState:
    """Initialize PHMState from PHM-Vibench data_factory (preview-only roots).

    - Roots keep only a small preview for planning/execution debugging.
    - Real training must use `state.data_cfg` and re-load via data_factory in the trainer.
    """
    from src.utils.data_factory_wrapper import PHMVibenchDataFactory

    built = PHMVibenchDataFactory(data_cfg).build()

    def _take_preview(loader) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        out: Dict[str, Any] = {}
        labs: Dict[str, Any] = {}
        for batch in loader:
            x = batch["x"]  # (B,L,C)
            y = batch["y"]  # (B,)
            ids = list(batch.get("file_id") or [])
            for i, sid in enumerate(ids):
                k = str(sid)
                if k in out:
                    continue
                out[k] = x[i : i + 1].detach().cpu().numpy()  # (1,L,C)
                labs[k] = str(int(y[i].detach().cpu().item()))
                if len(out) >= int(max_preview_samples):
                    return out, labs
        return out, labs

    train_preview, labels_train = _take_preview(built.train_loader)
    val_preview, labels_val = _take_preview(built.val_loader)
    test_preview, labels_test = _take_preview(built.test_loader)
    if not train_preview:
        raise ValueError("vibench preview is empty (train split). Check data_cfg.")

    first_arr = next(iter(train_preview.values()))
    if getattr(first_arr, "shape", None) is None or len(first_arr.shape) != 3:
        raise ValueError("Expected preview arrays with shape (1,L,C).")
    _, L, C = first_arr.shape

    channel_names = [f"ch{i+1}" for i in range(int(C))]
    nodes: Dict[str, Any] = {}
    leaves: List[str] = []
    fs_hz = data_cfg.get("fs_hz")

    for i, ch in enumerate(channel_names):
        ch_train = {sid: arr[:, :, i : i + 1] for sid, arr in train_preview.items()}
        ch_val = {sid: arr[:, :, i : i + 1] for sid, arr in val_preview.items()}
        ch_test = {sid: arr[:, :, i : i + 1] for sid, arr in test_preview.items()}
        node = InputData(
            node_id=ch,
            data={},
            results=normalize_result_splits({"train": ch_train, "val": ch_val, "test": ch_test}),
            parents=[],
            shape=(1, int(L), 1),
            meta={
                "channel": ch,
                "labels_train": labels_train,
                "labels_val": labels_val,
                "labels_test": labels_test,
                "fs": fs_hz,
            },
        )
        nodes[ch] = node
        leaves.append(ch)

    dag_state = DAGState(user_instruction=user_instruction, nodes=nodes, leaves=leaves, channels=channel_names)

    return PHMState(
        case_name=case_name,
        user_instruction=user_instruction,
        reference_signal=next(iter(nodes.values())),
        test_signal=next(iter(nodes.values())),
        dag_state=dag_state,
        labels_train=labels_train,
        labels_val=labels_val,
        labels_test=labels_test,
        allow_test_labels_for_reporting=allow_test_labels_for_reporting,
        train_backend=train_backend,
        model_config_path=model_config_path,
        save_dir=save_dir,
        data_cfg=data_cfg,
    )


def generate_final_report(final_state, report_path: str):
    """
    保存最终的报告。
    """
    print("\n--- Workflow Finished ---")
    if final_state.final_report:
        report = final_state.final_report
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


def _resolve_source_mode_for_state(state: PHMState | None, source_mode: str | None = None) -> str:
    source = str(source_mode or "").strip().lower()
    if source in {"fixed_ids", "vibench"}:
        return source
    data_cfg = dict(getattr(state, "data_cfg", {}) or {})
    source = str(data_cfg.get("source_mode") or "").strip().lower()
    if source in {"fixed_ids", "vibench"}:
        return source
    backend = str(data_cfg.get("backend") or "").strip().lower()
    if backend == "vibench":
        return "vibench"
    return "fixed_ids"


def _resolve_state_save_mode(save_mode: str | None, source_mode: str | None, state: PHMState | None) -> str:
    requested = str(save_mode or "auto").strip().lower() or "auto"
    if requested not in {"auto", "full", "minimal"}:
        requested = "auto"
    if requested != "auto":
        return requested
    resolved_source = _resolve_source_mode_for_state(state, source_mode)
    return "minimal" if resolved_source == "vibench" else "full"


def _sum_numpy_nbytes(obj: Any, seen: set[int]) -> int:
    if obj is None:
        return 0
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        total = 0
        for key, value in obj.items():
            total += _sum_numpy_nbytes(key, seen)
            total += _sum_numpy_nbytes(value, seen)
        return total
    if isinstance(obj, (list, tuple, set)):
        return sum(_sum_numpy_nbytes(item, seen) for item in obj)
    return 0


def _estimate_state_numpy_bytes(state: PHMState) -> int:
    dag_state = getattr(state, "dag_state", None)
    nodes = dict(getattr(dag_state, "nodes", {}) or {})
    seen: set[int] = set()
    total = 0
    for node in nodes.values():
        total += _sum_numpy_nbytes(getattr(node, "data", None), seen)
        total += _sum_numpy_nbytes(getattr(node, "results", None), seen)
    total += _sum_numpy_nbytes(getattr(state, "datasets", None), seen)
    return int(total)


def _build_minimal_state_snapshot(state: PHMState) -> PHMState:
    snapshot = state.model_copy(deep=True)
    dag_state = getattr(snapshot, "dag_state", None)
    nodes = dict(getattr(dag_state, "nodes", {}) or {})
    for node in nodes.values():
        if hasattr(node, "data"):
            node.data = {}
        if hasattr(node, "results"):
            node.results = {}
    for root_name in ("reference_signal", "test_signal"):
        root = getattr(snapshot, root_name, None)
        if root is not None:
            if hasattr(root, "data"):
                root.data = {}
            if hasattr(root, "results"):
                root.results = {}
    return snapshot


def _normalize_state_split_contract(state: PHMState) -> PHMState:
    dag_state = getattr(state, "dag_state", None)
    nodes = dict(getattr(dag_state, "nodes", {}) or {})
    for node in nodes.values():
        if hasattr(node, "results"):
            node.results = normalize_result_splits(getattr(node, "results", None))
        meta = dict(getattr(node, "meta", {}) or {})
        if "labels_train" not in meta and "labels_ref" in meta:
            meta["labels_train"] = meta.pop("labels_ref")
        if "labels_test" not in meta and "labels_tst" in meta:
            meta["labels_test"] = meta.pop("labels_tst")
        meta.setdefault("labels_val", {})
        if hasattr(node, "meta"):
            node.meta = meta
    if not getattr(state, "labels_train", None):
        state.labels_train = dict(getattr(state, "labels_ref", {}) or {})
    if not getattr(state, "labels_test", None):
        state.labels_test = dict(getattr(state, "labels_tst", {}) or {})
    if getattr(state, "labels_val", None) is None:
        state.labels_val = {}
    state.labels_ref = dict(state.labels_train)
    state.labels_tst = dict(state.labels_test)
    return state


def _missing_root_train_channels(state: PHMState) -> List[str]:
    missing: List[str] = []
    channels = list(getattr(state.dag_state, "channels", []) or [])
    for ch in channels:
        node = state.dag_state.nodes.get(ch)
        if not isinstance(node, InputData):
            missing.append(str(ch))
            continue
        train = get_split_results(node.results or {}, "train")
        if not isinstance(train, dict) or not train:
            missing.append(str(ch))
    return missing


def save_state(state, filepath: str, *, save_mode: str = "auto", source_mode: str | None = None):
    """
    使用pickle将状态对象保存到磁盘。
    """
    try:
        resolved_source_mode = _resolve_source_mode_for_state(state, source_mode)
        effective_mode = _resolve_state_save_mode(save_mode, resolved_source_mode, state)
        state_to_save = state if effective_mode == "full" else _build_minimal_state_snapshot(state)
        bytes_before = _estimate_state_numpy_bytes(state)
        bytes_after = _estimate_state_numpy_bytes(state_to_save)

        print(
            f"\n--- Saving state to {filepath} "
            f"(requested={save_mode}, effective={effective_mode}, source_mode={resolved_source_mode}) ---"
        )
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as f:
            pickle.dump(state_to_save, f)
        digest = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        with open(f"{abs_path}.sha256", "w", encoding="utf-8") as f:
            f.write(digest.hexdigest())
        bytes_reduced = max(0, int(bytes_before) - int(bytes_after))
        reduction_ratio = (float(bytes_reduced) / float(bytes_before)) if int(bytes_before) > 0 else 0.0
        meta = {
            "save_mode_requested": str(save_mode or "auto"),
            "effective_mode": effective_mode,
            "source_mode": resolved_source_mode,
            "nodes_count": len(getattr(state.dag_state, "nodes", {}) or {}),
            "channels_count": len(getattr(state.dag_state, "channels", []) or []),
            "numpy_bytes_before": int(bytes_before),
            "numpy_bytes_after": int(bytes_after),
            "numpy_bytes_reduced": int(bytes_reduced),
            "numpy_reduction_ratio": float(reduction_ratio),
            "pickle_bytes": int(os.path.getsize(abs_path)),
        }
        with open(f"{abs_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
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
        abs_path = os.path.abspath(filepath)
        print(f"\n--- Loading state from {abs_path} ---")
        sidecar = f"{abs_path}.sha256"
        allow_unverified = str(os.getenv("PHM_ALLOW_UNVERIFIED_STATE", "")).strip().lower() in {"1", "true", "yes", "y"}
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                expected = f.read().strip()
            digest = hashlib.sha256()
            with open(abs_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise ValueError("State checksum verification failed.")
        elif not allow_unverified:
            raise ValueError(
                f"State checksum sidecar missing: {sidecar}. "
                "Set PHM_ALLOW_UNVERIFIED_STATE=1 to bypass for local debugging."
            )

        with open(abs_path, "rb") as f:
            state = pickle.load(f)
        state = _normalize_state_split_contract(state)
        print("...done.")
        print(f"Successfully loaded state with {len(state.dag_state.nodes)} nodes.")
        source = _resolve_source_mode_for_state(state, None)
        if source == "fixed_ids":
            missing = _missing_root_train_channels(state)
            if missing:
                print(
                    "Warning: loaded state is missing root InputData.results['train'] for "
                    f"channels={missing}. fixed_ids flow expects full state arrays. "
                    "If this is a minimal snapshot, set data.state_save_mode=full and rebuild."
                )
        return state
    except Exception as e:
        print(f"Error loading state: {e}")
        return None


def get_dag_depth(dag_state: "DAGState") -> int:
    """
    计算DAG的最大深度。
    深度定义为最长路径上的节点数。
    """
    if not dag_state.nodes:
        return 0
    
    # 1. 从 state 中的节点和父子关系构建一个 networkx 有向图
    G = nx.DiGraph()
    for node_id, node in dag_state.nodes.items():
        G.add_node(node_id)
        # 确保 parents 属性是一个列表
        parents = node.parents if isinstance(node.parents, list) else [node.parents]
        for parent_id in parents:
            if parent_id:  # 根节点的 parent 可能为空列表
                G.add_edge(parent_id, node_id)

    # 2. 检查图是否是无环的（DAG的基本要求）
    if not nx.is_directed_acyclic_graph(G):
        print("Warning: Cycle detected in the DAG. Depth calculation is not possible.")
        return -1  # 返回-1表示错误状态

    # 如果图中没有节点，深度为0
    if not G.nodes:
        return 0
        
    # 如果有节点但没有边（所有节点都是根节点），深度为1
    if not G.edges:
        return 1

    # 3. 使用 networkx 计算最长路径的长度（边数）
    #    深度（节点数）= 边数 + 1
    try:
        # nx.dag_longest_path_length 在整个DAG（可能不连通）中找到最长路径
        longest_path_edges = nx.dag_longest_path_length(G)
        return longest_path_edges + 1
    except nx.NetworkXError:
        # 这是一个备用逻辑，以防万一（例如，在空图上调用，尽管已经检查过）
        return 1 if G.nodes else 0


if __name__ == "__main__":
    import numpy as np

    # 构建一个简单的 DAG: 输入信号 -> 求均值
    train = np.arange(10, dtype=float).reshape(1, 10, 1)
    test = np.arange(10, 20, dtype=float).reshape(1, 10, 1)

    in_node = InputData(
        node_id="ch1",
        parents=[],
        shape=train.shape,
        data={},
        results={"train": train, "val": {}, "test": test},
        meta={"fs": 1},
    )

    mean_node = ProcessedData(
        node_id="n_mean",
        parents=["ch1"],
        shape=(1, 1),
        source_signal_id="ch1",
        method="mean",
        meta={"tool": "mean", "params": {}, "parent": "ch1"},
    )

    dag = DAGState(
        user_instruction="demo",
        channels=["ch1"],
        nodes={"ch1": in_node, "n_mean": mean_node},
        leaves=["n_mean"],
    )

    state = PHMState(
        case_name="demo",
        user_instruction="demo",
        reference_signal=in_node,
        test_signal=in_node,
        dag_state=dag,
        fs=1,
    )

    new_train = {"ch1": train * 2}
    new_test = {"ch1": test * 2}
    updated = run_dag_on_new_data(state, new_train, new_test)
    print("Mean results:", updated.dag_state.nodes["n_mean"].results)
