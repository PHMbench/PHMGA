from __future__ import annotations

import json
from typing import Any, Dict, List

from src.configuration import Configuration
from src.model import get_llm
from src.states.phm_states import DAGState, InputData, PHMState, ProcessedData


def _infer_L_and_fs(state: PHMState) -> tuple[int | None, float | None]:
    channels = list(getattr(state.dag_state, "channels", []) or [])
    if not channels:
        return None, None
    root = state.dag_state.nodes.get(channels[0])
    if not isinstance(root, InputData):
        return None, None
    ref = (root.results or {}).get("ref") or {}
    if isinstance(ref, dict) and ref:
        first = next(iter(ref.values()))
        try:
            L = int(getattr(first, "shape", (0, 0, 0))[1])
        except Exception:
            L = None
    else:
        L = None
    fs = root.meta.get("fs")
    try:
        fs_f = float(fs) if fs is not None else None
    except Exception:
        fs_f = None
    return L, fs_f


def _coerce_chain(item: Any) -> List[str]:
    if isinstance(item, str):
        return [item]
    if isinstance(item, list):
        return [str(x) for x in item if str(x).strip()]
    return []


def dag_init_agent(
    state: PHMState,
    *,
    max_ops_per_channel: int = 2,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Initialize a minimal processed DAG using an LLM.

    This is a lightweight bridge between the outer loop (LLM) and the inner loop (TSPN):
    - input: only channel roots exist in the DAG (leaves are channel ids)
    - output: adds ProcessedData nodes (method tokens like 'fft', 'hilbert', 'wavefilter')
      and updates `dag_state.leaves` to those new nodes.

    The resulting DAG is then used by `tspn_bootstrap_agent` to create a TSPN config.
    """
    channels = list(getattr(state.dag_state, "channels", []) or [])
    if not channels:
        return {}

    # If the DAG already contains processed nodes, do nothing.
    if any(isinstance(n, ProcessedData) for n in (state.dag_state.nodes or {}).values()):
        return {}

    L, fs = _infer_L_and_fs(state)

    llm = get_llm(Configuration.from_runnable_config(None), temperature=temperature)
    prompt = {
        "role": "system",
        "content": (
            "You are a signal-processing expert. "
            "Choose a minimal per-channel preprocessing chain to help fault diagnosis. "
            "You must respond with STRICT JSON only."
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps(
            {
                "task": "Initialize a minimal processed DAG (per channel).",
                "constraints": {
                    "max_ops_per_channel": int(max_ops_per_channel),
                    "allowed_ops": ["identity", "fft", "hilbert", "wavefilter"],
                    "prefer_ops": ["fft", "wavefilter"],
                    "notes": "Use lowercase method names. Do not invent new ops.",
                },
                "data_hints": {"channels": channels, "L": L, "fs_hz": fs},
                "output_schema": {
                    "channels": [
                        {"channel_id": "ch1", "ops": ["fft", "wavefilter"]},
                    ]
                },
            },
            ensure_ascii=False,
        ),
    }

    chain_by_channel: Dict[str, List[str]] = {}
    try:
        resp = llm.invoke([prompt, user])
        raw = getattr(resp, "content", resp)
        data = json.loads(raw)
        for item in (data.get("channels") or []):
            cid = str(item.get("channel_id") or "").strip()
            if cid and cid in channels:
                chain = _coerce_chain(item.get("ops"))
                chain_by_channel[cid] = chain[: int(max_ops_per_channel)]
    except Exception:
        chain_by_channel = {}

    # Fallback: always create one FFT node per channel.
    for ch in channels:
        chain_by_channel.setdefault(ch, ["fft"])

    new_dag: DAGState = state.dag_state.model_copy(deep=True)
    nodes = dict(new_dag.nodes or {})
    leaves: List[str] = []

    for ch in channels:
        parent = ch
        chain = chain_by_channel.get(ch) or ["fft"]
        for i, method in enumerate(chain, start=1):
            m = str(method).strip().lower()
            if m not in {"identity", "fft", "hilbert", "wavefilter"}:
                continue
            nid = f"init_{i:02d}_{m}_{ch}"
            # Shape contract for bootstrap: keep (1,L,1) at each node.
            shape = (1, int(L or 0), 1) if L else (1, 0, 1)
            nodes[nid] = ProcessedData(
                node_id=nid,
                parents=[parent],
                shape=shape,
                source_signal_id=ch,
                method=m,
                results=None,
                meta={"tool": "dag_init_agent", "channel": ch},
            )
            parent = nid
        leaves.append(parent)

    new_dag.nodes = nodes
    new_dag.leaves = leaves
    return {"dag_state": new_dag}

