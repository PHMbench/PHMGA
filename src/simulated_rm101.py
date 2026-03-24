from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

from .config import load_case_config
from .evaluation.full_dag_ml import summarize_dag_state
from .rm101_metadata import get_channel_aliases
from .states.phm_states import DAGState, InputData, PHMState, ProcessedData
from .tools import MultiVariableOp, get_operator


RM101_FS = 12800.0
SIMULATED_MODEL_TAGS = (
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
)

_MODEL_ORDER = list(SIMULATED_MODEL_TAGS)
_SPECTRAL_OR_ORDER_OPS = {
    "fft",
    "psd",
    "stft",
    "order_track_resample",
    "tsa_cycle_average",
}
_FEATURE_OPS = {
    "rms",
    "kurtosis",
    "crest_factor",
    "spectral_centroid",
    "order_band_energy",
    "sideband_ratio",
}
_EXPECTED_SUMMARY = {
    "google/gemini-2.0-flash-001": {"depth": 3, "node_count": 20},
    "google/gemini-2.5-flash": {"depth": 4, "node_count": 28},
    "google/gemini-2.5-pro": {"depth": 5, "node_count": 52},
}


@dataclass(frozen=True)
class SimulatedRunInfo:
    model_tag: str
    run_name: str
    paper_label: str
    run_type: str = "simulated_variant"


def get_simulated_run_info(model_tag: str, *, run_suffix: str = "simulated_paper_v3") -> SimulatedRunInfo:
    if model_tag not in SIMULATED_MODEL_TAGS:
        raise ValueError(f"Unsupported simulated model tag: {model_tag}")
    labels = {
        "google/gemini-2.0-flash-001": "Gemini 2.0 Flash (simulated)",
        "google/gemini-2.5-flash": "Gemini 2.5 Flash (simulated)",
        "google/gemini-2.5-pro": "Gemini 2.5 Pro (simulated)",
    }
    slug = model_tag.replace("/", "__")
    return SimulatedRunInfo(
        model_tag=model_tag,
        run_name=f"openrouter__{slug}__{run_suffix}",
        paper_label=labels[model_tag],
    )


def _key_phase_signal(length: int, period: int = 256) -> np.ndarray:
    signal = np.zeros((length,), dtype=float)
    signal[::period] = 1.0
    for offset in range(1, 4):
        signal[offset::period] = np.maximum(signal[offset::period], 0.25 / offset)
    return signal


def _vibration_signal(
    time: np.ndarray,
    *,
    base_hz: float,
    side_hz: float,
    impulse_period: int,
    amplitude: float,
) -> np.ndarray:
    wave = amplitude * np.sin(2.0 * np.pi * base_hz * time)
    wave += 0.35 * np.sin(2.0 * np.pi * side_hz * time + 0.4)
    envelope = 1.0 + 0.25 * np.sin(2.0 * np.pi * 3.2 * time)
    wave = wave * envelope
    impulses = np.zeros_like(wave)
    impulses[::impulse_period] = 1.0
    kernel = np.exp(-np.linspace(0.0, 3.0, 32))
    wave += 0.18 * np.convolve(impulses, kernel, mode="same")
    return wave


def _build_root_results(length: int = 4096) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    time = np.arange(length, dtype=float) / RM101_FS
    pulse = _key_phase_signal(length)
    speed = pulse + 0.05 * np.sin(2.0 * np.pi * 2.0 * time)
    torque = 1.0 + 0.15 * np.sin(2.0 * np.pi * 1.25 * time) + 0.05 * np.cos(2.0 * np.pi * 0.75 * time)

    roots_ref: Dict[str, np.ndarray] = {}
    roots_tst: Dict[str, np.ndarray] = {}
    channels = {
        "ch1": speed,
        "ch2": torque,
        "ch3": _vibration_signal(time, base_hz=28.0, side_hz=56.0, impulse_period=220, amplitude=0.6),
        "ch4": _vibration_signal(time, base_hz=31.0, side_hz=63.0, impulse_period=240, amplitude=0.55),
        "ch5": _vibration_signal(time, base_hz=34.0, side_hz=68.0, impulse_period=260, amplitude=0.5),
        "ch6": _vibration_signal(time, base_hz=92.0, side_hz=184.0, impulse_period=128, amplitude=0.9),
        "ch7": _vibration_signal(time, base_hz=105.0, side_hz=210.0, impulse_period=112, amplitude=1.0),
        "ch8": _vibration_signal(time, base_hz=118.0, side_hz=236.0, impulse_period=96, amplitude=1.1),
    }
    for channel_id, series in channels.items():
        ref = np.asarray(series, dtype=float).reshape(1, -1, 1)
        tst = np.asarray(series * (1.02 if channel_id.startswith("ch6") else 0.98), dtype=float).reshape(1, -1, 1)
        roots_ref[channel_id] = ref
        roots_tst[channel_id] = tst
    return {
        "ref": {"ref_dummy": roots_ref},
        "tst": {"tst_dummy": roots_tst},
    }


def _base_runtime_config(model_tag: str) -> Dict[str, Any]:
    run_info = get_simulated_run_info(model_tag)
    return {
        "llm": {
            "provider": "openrouter",
            "model": model_tag,
        },
        "run_metadata": {
            "run_type": run_info.run_type,
            "paper_label": run_info.paper_label,
        },
    }


def _build_base_state(case_name: str, model_tag: str) -> PHMState:
    channel_aliases = get_channel_aliases(dataset_name="RM101", dataset_id=101, channel_count=8)
    root_results = _build_root_results()
    labels = {"ref_dummy": 0, "tst_dummy": 0}
    nodes: Dict[str, InputData] = {}
    for channel_id in [f"ch{i}" for i in range(1, 9)]:
        alias = channel_aliases.get(channel_id, channel_id)
        node = InputData(
            node_id=channel_id,
            parents=[],
            shape=(1, 4096, 1),
            data={},
            results={
                "ref": {"ref_dummy": root_results["ref"]["ref_dummy"][channel_id]},
                "tst": {"tst_dummy": root_results["tst"]["tst_dummy"][channel_id]},
            },
            metadata={},
            meta={
                "channel": channel_id,
                "channel_alias": alias,
                "labels": labels,
                "fs": RM101_FS,
            },
        )
        nodes[channel_id] = node
    dag = DAGState(
        user_instruction="offline simulated RM101 variable-speed planner",
        channels=[f"ch{i}" for i in range(1, 9)],
        nodes=nodes,
        leaves=list(nodes),
    )
    return PHMState(
        case_name=case_name,
        user_instruction="offline simulated RM101 variable-speed planner",
        reference_signal=nodes["ch6"],
        test_signal=nodes["ch6"],
        dag_state=dag,
        fs=RM101_FS,
        runtime_config=_base_runtime_config(model_tag),
    )


def _node_op_name(node: Any) -> str:
    return str(node.meta.get("tool") or node.meta.get("method") or getattr(node, "method", "") or "").strip()


def _execute_results(
    state: PHMState,
    *,
    parent_ids: List[str],
    op_name: str,
    params: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, np.ndarray]], tuple[int, ...]]:
    op_cls = get_operator(op_name)
    parent_value: str | List[str] = parent_ids[0] if len(parent_ids) == 1 else list(parent_ids)
    op = op_cls(parent=parent_value, **params)
    outputs: Dict[str, Dict[str, np.ndarray]] = {"ref": {}, "tst": {}}
    inferred_shape: tuple[int, ...] | None = None
    split_names = ("ref", "tst")
    if issubclass(op_cls, MultiVariableOp):
        for split_name in split_names:
            parent_maps = {
                parent_id: dict((state.dag_state.nodes[parent_id].results or {}).get(split_name) or {})
                for parent_id in parent_ids
            }
            common_ids = set.intersection(*(set(values.keys()) for values in parent_maps.values()))
            for sample_id in sorted(common_ids):
                payload = {parent_id: parent_maps[parent_id][sample_id] for parent_id in parent_ids}
                value = np.asarray(op.execute(payload), dtype=float)
                outputs[split_name][sample_id] = value
                if inferred_shape is None:
                    inferred_shape = tuple(value.shape)
    else:
        parent_id = parent_ids[0]
        for split_name in split_names:
            parent_map = dict((state.dag_state.nodes[parent_id].results or {}).get(split_name) or {})
            for sample_id, array in parent_map.items():
                value = np.asarray(op.execute(array), dtype=float)
                outputs[split_name][sample_id] = value
                if inferred_shape is None:
                    inferred_shape = tuple(value.shape)
    if inferred_shape is None:
        raise RuntimeError(f"Failed to infer shape for simulated node using op={op_name}.")
    return outputs, inferred_shape


def _channel_metadata(state: PHMState, parent_ids: List[str]) -> tuple[str, str]:
    first = state.dag_state.nodes[parent_ids[0]]
    if len(parent_ids) == 1:
        return str(first.meta.get("channel", parent_ids[0])), str(first.meta.get("channel_alias", parent_ids[0]))
    aliases = [str(state.dag_state.nodes[parent_id].meta.get("channel_alias", parent_id)) for parent_id in parent_ids]
    return "multi", "+".join(aliases)


def _add_processed_node(
    state: PHMState,
    *,
    node_id: str,
    parents: List[str],
    op_name: str,
    params: Dict[str, Any] | None = None,
) -> str:
    params = dict(params or {})
    outputs, shape = _execute_results(state, parent_ids=parents, op_name=op_name, params=params)
    channel_name, channel_alias = _channel_metadata(state, parents)
    node = ProcessedData(
        node_id=node_id,
        parents=list(parents),
        source_signal_id=str(parents[0]),
        method=op_name,
        results=outputs,
        meta={
            "tool": op_name,
            "method": op_name,
            "params": params,
            "parent": parents[0] if len(parents) == 1 else list(parents),
            "channel": channel_name,
            "channel_alias": channel_alias,
            "input_aliases": [str(state.dag_state.nodes[parent_id].meta.get("channel_alias", parent_id)) for parent_id in parents],
        },
        shape=shape,
    )
    state.tracker().add_node(node)
    return node_id


def _refresh_terminal_processed_leaves(state: PHMState) -> None:
    graph = state.tracker().g
    leaves = []
    for node_id in nx.topological_sort(graph):
        node = state.dag_state.nodes[node_id]
        if node.stage == "input":
            continue
        if graph.out_degree(node_id) == 0:
            leaves.append(node_id)
    state.dag_state.leaves = leaves
    state._tracker_instance = None


def _template_gemini_20(state: PHMState) -> None:
    _add_processed_node(state, node_id="fft_06_ch6", parents=["ch6"], op_name="fft")
    _add_processed_node(state, node_id="psd_06_ch6", parents=["ch6"], op_name="psd", params={"fs": RM101_FS, "nperseg": 256})
    _add_processed_node(state, node_id="fft_07_ch7", parents=["ch7"], op_name="fft")
    _add_processed_node(state, node_id="psd_07_ch7", parents=["ch7"], op_name="psd", params={"fs": RM101_FS, "nperseg": 256})
    _add_processed_node(state, node_id="fft_08_ch8", parents=["ch8"], op_name="fft")
    _add_processed_node(state, node_id="psd_08_ch8", parents=["ch8"], op_name="psd", params={"fs": RM101_FS, "nperseg": 256})

    _add_processed_node(state, node_id="rms_01_fft_06_ch6", parents=["fft_06_ch6"], op_name="rms")
    _add_processed_node(state, node_id="crest_factor_02_psd_06_ch6", parents=["psd_06_ch6"], op_name="crest_factor")
    _add_processed_node(state, node_id="kurtosis_03_fft_07_ch7", parents=["fft_07_ch7"], op_name="kurtosis")
    _add_processed_node(state, node_id="spectral_centroid_04_psd_07_ch7", parents=["psd_07_ch7"], op_name="spectral_centroid", params={"fs": RM101_FS})
    _add_processed_node(state, node_id="rms_05_fft_08_ch8", parents=["fft_08_ch8"], op_name="rms")
    _add_processed_node(state, node_id="kurtosis_06_psd_08_ch8", parents=["psd_08_ch8"], op_name="kurtosis")


def _template_gemini_25_flash(state: PHMState) -> None:
    _add_processed_node(
        state,
        node_id="order_track_06_ch6",
        parents=["ch6", "ch1"],
        op_name="order_track_resample",
        params={"points_per_rev": 256, "revolutions": 4},
    )
    _add_processed_node(
        state,
        node_id="order_track_07_ch7",
        parents=["ch7", "ch1"],
        op_name="order_track_resample",
        params={"points_per_rev": 256, "revolutions": 4},
    )
    _add_processed_node(state, node_id="fft_06_ch6", parents=["ch6"], op_name="fft")
    _add_processed_node(state, node_id="psd_07_ch7", parents=["ch7"], op_name="psd", params={"fs": RM101_FS, "nperseg": 256})
    _add_processed_node(state, node_id="fft_08_ch8", parents=["ch8"], op_name="fft")
    _add_processed_node(state, node_id="psd_08_ch8", parents=["ch8"], op_name="psd", params={"fs": RM101_FS, "nperseg": 256})
    _add_processed_node(state, node_id="coherence_58_ch5_ch8", parents=["ch5", "ch8"], op_name="coherence", params={"fs": RM101_FS, "nperseg": 256})
    _add_processed_node(state, node_id="coherence_47_ch4_ch7", parents=["ch4", "ch7"], op_name="coherence", params={"fs": RM101_FS, "nperseg": 256})

    _add_processed_node(state, node_id="order_band_energy_01_order_track_06_ch6", parents=["order_track_06_ch6"], op_name="order_band_energy")
    _add_processed_node(state, node_id="sideband_ratio_02_order_track_06_ch6", parents=["order_track_06_ch6"], op_name="sideband_ratio")
    _add_processed_node(state, node_id="order_band_energy_03_order_track_07_ch7", parents=["order_track_07_ch7"], op_name="order_band_energy")
    _add_processed_node(state, node_id="sideband_ratio_04_order_track_07_ch7", parents=["order_track_07_ch7"], op_name="sideband_ratio")
    _add_processed_node(state, node_id="rms_05_fft_06_ch6", parents=["fft_06_ch6"], op_name="rms")
    _add_processed_node(state, node_id="kurtosis_06_psd_07_ch7", parents=["psd_07_ch7"], op_name="kurtosis")
    _add_processed_node(state, node_id="spectral_centroid_07_fft_08_ch8", parents=["fft_08_ch8"], op_name="spectral_centroid", params={"fs": RM101_FS})
    _add_processed_node(state, node_id="crest_factor_08_psd_08_ch8", parents=["psd_08_ch8"], op_name="crest_factor")
    _add_processed_node(state, node_id="rms_09_coherence_58_ch5_ch8", parents=["coherence_58_ch5_ch8"], op_name="rms")
    _add_processed_node(state, node_id="kurtosis_10_coherence_47_ch4_ch7", parents=["coherence_47_ch4_ch7"], op_name="kurtosis")
    _add_processed_node(
        state,
        node_id="torque_normalize_11_order_band_energy_01_order_track_06_ch6",
        parents=["order_band_energy_01_order_track_06_ch6", "ch2"],
        op_name="torque_normalize",
        params={"mode": "abs_mean"},
    )
    _add_processed_node(
        state,
        node_id="torque_normalize_12_sideband_ratio_04_order_track_07_ch7",
        parents=["sideband_ratio_04_order_track_07_ch7", "ch2"],
        op_name="torque_normalize",
        params={"mode": "abs_mean"},
    )


def _template_gemini_25_pro(state: PHMState) -> None:
    for channel in ("ch6", "ch7", "ch8"):
        suffix = f"{int(channel.split('ch', 1)[1]):02d}"
        _add_processed_node(
            state,
            node_id=f"order_track_{suffix}_{channel}",
            parents=[channel, "ch1"],
            op_name="order_track_resample",
            params={"points_per_rev": 256, "revolutions": 4},
        )
        _add_processed_node(
            state,
            node_id=f"tsa_{suffix}_{channel}",
            parents=[channel, "ch1"],
            op_name="tsa_cycle_average",
            params={"points_per_rev": 256, "revolutions": 4},
        )
        _add_processed_node(
            state,
            node_id=f"stft_{suffix}_{channel}",
            parents=[channel],
            op_name="stft",
            params={"fs": RM101_FS, "nperseg": 256, "noverlap": 128},
        )
    _add_processed_node(state, node_id="fft_06_ch6", parents=["ch6"], op_name="fft")
    _add_processed_node(state, node_id="psd_07_ch7", parents=["ch7"], op_name="psd", params={"fs": RM101_FS, "nperseg": 256})
    _add_processed_node(state, node_id="fft_08_ch8", parents=["ch8"], op_name="fft")
    _add_processed_node(state, node_id="coherence_58_ch5_ch8", parents=["ch5", "ch8"], op_name="coherence", params={"fs": RM101_FS, "nperseg": 256})
    _add_processed_node(state, node_id="coherence_47_ch4_ch7", parents=["ch4", "ch7"], op_name="coherence", params={"fs": RM101_FS, "nperseg": 256})

    _add_processed_node(state, node_id="fft_tsa_06_tsa_06_ch6", parents=["tsa_06_ch6"], op_name="fft")
    _add_processed_node(state, node_id="psd_tsa_07_tsa_07_ch7", parents=["tsa_07_ch7"], op_name="psd", params={"fs": RM101_FS, "nperseg": 128})
    _add_processed_node(state, node_id="fft_tsa_08_tsa_08_ch8", parents=["tsa_08_ch8"], op_name="fft")
    _add_processed_node(state, node_id="patch_01_fft_tsa_06_tsa_06_ch6", parents=["fft_tsa_06_tsa_06_ch6"], op_name="patch", params={"patch_size": 32, "stride": 32})
    _add_processed_node(state, node_id="patch_02_psd_tsa_07_tsa_07_ch7", parents=["psd_tsa_07_tsa_07_ch7"], op_name="patch", params={"patch_size": 32, "stride": 32})
    _add_processed_node(state, node_id="patch_03_fft_08_ch8", parents=["fft_08_ch8"], op_name="patch", params={"patch_size": 32, "stride": 32})

    _add_processed_node(state, node_id="order_band_energy_01_order_track_06_ch6", parents=["order_track_06_ch6"], op_name="order_band_energy")
    _add_processed_node(state, node_id="sideband_ratio_02_order_track_06_ch6", parents=["order_track_06_ch6"], op_name="sideband_ratio")
    _add_processed_node(state, node_id="order_band_energy_03_order_track_07_ch7", parents=["order_track_07_ch7"], op_name="order_band_energy")
    _add_processed_node(state, node_id="sideband_ratio_04_order_track_07_ch7", parents=["order_track_07_ch7"], op_name="sideband_ratio")
    _add_processed_node(state, node_id="order_band_energy_05_order_track_08_ch8", parents=["order_track_08_ch8"], op_name="order_band_energy")
    _add_processed_node(state, node_id="sideband_ratio_06_order_track_08_ch8", parents=["order_track_08_ch8"], op_name="sideband_ratio")
    _add_processed_node(state, node_id="crest_factor_07_fft_tsa_06_tsa_06_ch6", parents=["fft_tsa_06_tsa_06_ch6"], op_name="crest_factor")
    _add_processed_node(state, node_id="kurtosis_08_psd_tsa_07_tsa_07_ch7", parents=["psd_tsa_07_tsa_07_ch7"], op_name="kurtosis")
    _add_processed_node(state, node_id="spectral_centroid_09_fft_tsa_08_tsa_08_ch8", parents=["fft_tsa_08_tsa_08_ch8"], op_name="spectral_centroid", params={"fs": RM101_FS})
    _add_processed_node(state, node_id="rms_10_stft_06_ch6", parents=["stft_06_ch6"], op_name="rms")
    _add_processed_node(state, node_id="kurtosis_11_stft_07_ch7", parents=["stft_07_ch7"], op_name="kurtosis")
    _add_processed_node(state, node_id="crest_factor_12_stft_08_ch8", parents=["stft_08_ch8"], op_name="crest_factor")
    _add_processed_node(state, node_id="rms_13_patch_01_fft_tsa_06_tsa_06_ch6", parents=["patch_01_fft_tsa_06_tsa_06_ch6"], op_name="rms")
    _add_processed_node(state, node_id="kurtosis_14_patch_02_psd_tsa_07_tsa_07_ch7", parents=["patch_02_psd_tsa_07_tsa_07_ch7"], op_name="kurtosis")
    _add_processed_node(state, node_id="crest_factor_15_patch_03_fft_08_ch8", parents=["patch_03_fft_08_ch8"], op_name="crest_factor")
    _add_processed_node(state, node_id="spectral_centroid_16_fft_06_ch6", parents=["fft_06_ch6"], op_name="spectral_centroid", params={"fs": RM101_FS})
    _add_processed_node(state, node_id="kurtosis_17_psd_07_ch7", parents=["psd_07_ch7"], op_name="kurtosis")
    _add_processed_node(state, node_id="crest_factor_18_fft_08_ch8", parents=["fft_08_ch8"], op_name="crest_factor")
    _add_processed_node(state, node_id="rms_19_coherence_58_ch5_ch8", parents=["coherence_58_ch5_ch8"], op_name="rms")
    _add_processed_node(state, node_id="kurtosis_20_coherence_47_ch4_ch7", parents=["coherence_47_ch4_ch7"], op_name="kurtosis")
    _add_processed_node(
        state,
        node_id="torque_normalize_21_order_band_energy_05_order_track_08_ch8",
        parents=["order_band_energy_05_order_track_08_ch8", "ch2"],
        op_name="torque_normalize",
        params={"mode": "abs_mean"},
    )
    _add_processed_node(
        state,
        node_id="torque_normalize_22_sideband_ratio_06_order_track_08_ch8",
        parents=["sideband_ratio_06_order_track_08_ch8", "ch2"],
        op_name="torque_normalize",
        params={"mode": "abs_mean"},
    )
    _add_processed_node(
        state,
        node_id="torque_normalize_23_rms_19_coherence_58_ch5_ch8",
        parents=["rms_19_coherence_58_ch5_ch8", "ch2"],
        op_name="torque_normalize",
        params={"mode": "abs_mean"},
    )
    _add_processed_node(
        state,
        node_id="torque_normalize_24_rms_10_stft_06_ch6",
        parents=["rms_10_stft_06_ch6", "ch2"],
        op_name="torque_normalize",
        params={"mode": "abs_mean"},
    )


def apply_simulated_template(state: PHMState, model_tag: str) -> PHMState:
    if model_tag not in SIMULATED_MODEL_TAGS:
        raise ValueError(f"Unsupported simulated model tag: {model_tag}")
    working = state.model_copy(deep=True)
    working.runtime_config = _base_runtime_config(model_tag)
    if model_tag == "google/gemini-2.0-flash-001":
        _template_gemini_20(working)
    elif model_tag == "google/gemini-2.5-flash":
        _template_gemini_25_flash(working)
    else:
        _template_gemini_25_pro(working)
    _refresh_terminal_processed_leaves(working)
    return working


def _has_complete_strong_path(state: PHMState) -> bool:
    graph = state.tracker().g
    roots = [node_id for node_id, node in state.dag_state.nodes.items() if node.stage == "input"]
    for leaf_id in state.dag_state.leaves:
        if leaf_id not in graph:
            continue
        for root in roots:
            if root not in graph or not nx.has_path(graph, root, leaf_id):
                continue
            for path in nx.all_simple_paths(graph, root, leaf_id):
                seen_transform = False
                for node_id in path[1:]:
                    op_name = _node_op_name(state.dag_state.nodes[node_id])
                    if op_name in _SPECTRAL_OR_ORDER_OPS:
                        seen_transform = True
                    if seen_transform and op_name in _FEATURE_OPS:
                        return True
    return False


def validate_simulated_state(state: PHMState, model_tag: str) -> Dict[str, Any]:
    if model_tag not in _EXPECTED_SUMMARY:
        raise ValueError(f"Unsupported simulated model tag: {model_tag}")
    summary = summarize_dag_state(state)
    expected = _EXPECTED_SUMMARY[model_tag]
    if summary["depth"] != expected["depth"]:
        raise ValueError(f"{model_tag} depth mismatch: expected {expected['depth']}, got {summary['depth']}")
    if summary["node_count"] != expected["node_count"]:
        raise ValueError(f"{model_tag} node_count mismatch: expected {expected['node_count']}, got {summary['node_count']}")
    ops = set(summary.get("unique_ops") or [])
    if not _has_complete_strong_path(state):
        raise ValueError(f"{model_tag} missing strong path input -> spectral/order transform -> feature_stat.")
    if model_tag == "google/gemini-2.5-pro":
        required = {"patch", "stft", "order_track_resample", "tsa_cycle_average"}
        missing = sorted(required - ops)
        if missing:
            raise ValueError(f"{model_tag} missing required ops: {', '.join(missing)}")
    return summary


def validate_complexity_ladder(states: Dict[str, PHMState]) -> Dict[str, Dict[str, Any]]:
    summaries = {model_tag: validate_simulated_state(states[model_tag], model_tag) for model_tag in _MODEL_ORDER}
    previous_depth = -1
    previous_node_count = -1
    previous_ops: set[str] = set()
    for model_tag in _MODEL_ORDER:
        summary = summaries[model_tag]
        depth = int(summary["depth"])
        node_count = int(summary["node_count"])
        ops = set(summary.get("unique_ops") or [])
        if depth <= previous_depth:
            raise ValueError(f"Complexity ladder violated for {model_tag}: depth did not increase.")
        if node_count <= previous_node_count:
            raise ValueError(f"Complexity ladder violated for {model_tag}: node_count did not increase.")
        if previous_ops and not previous_ops.issubset(ops):
            raise ValueError(f"Complexity ladder violated for {model_tag}: unique_ops did not expand monotonically.")
        previous_depth = depth
        previous_node_count = node_count
        previous_ops = ops
    return summaries


def build_simulated_state(*, case_name: str, model_tag: str) -> tuple[PHMState, Dict[str, Any]]:
    case_config = load_case_config(case_name)
    state = _build_base_state(case_name, model_tag)
    state = apply_simulated_template(state, model_tag)
    validate_simulated_state(state, model_tag)
    return state, case_config


def save_simulated_state(state: PHMState, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(state, handle)
    return target


def write_simulated_summary(state: PHMState, path: str | Path) -> Path:
    payload = summarize_dag_state(state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
