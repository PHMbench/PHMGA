from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any

import numpy as np

from phm_core import PHMState, InputData, ProcessedData
from src.model import get_llm
from src.tools.signal_processing_schemas import get_operator


DATA_DIR = os.environ.get("PHM_DATA_DIR", "/mnt/data")
MAX_STEPS = 100


def _resolve_params(llm, params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve templated parameters using an LLM if needed."""
    resolved = {}
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
            prompt = v.strip("{}")
            try:
                resp = llm.invoke(prompt)
                resolved[k] = float(resp.content)
            except Exception:
                resolved[k] = v
        else:
            resolved[k] = v
    return resolved


def _get_results(node: InputData | ProcessedData) -> Dict[str, Any]:
    if isinstance(node, InputData):
        return node.results
    if isinstance(node, ProcessedData):
        return node.results
    return {}


def execute_agent(state: PHMState) -> Dict[str, Any]:
    executed_steps = 0
    llm = get_llm(None)
    tracker = state.tracker()

    for idx, step in enumerate(state.detailed_plan[:MAX_STEPS], start=1):
        op_name = step.get("op_name")
        params = step.get("params", {})
        parent_id = params.get("parent")

        if parent_id not in state.dag_state.nodes:
            state.dag_state.error_log.append(f"Missing parent: {parent_id}")
            break

        try:
            params = _resolve_params(llm, params)
            op_cls = get_operator(op_name)
            op = op_cls(**{k: v for k, v in params.items() if k != "parent"}, parent=parent_id)

            parent_node = state.dag_state.nodes[parent_id]
            parent_results = _get_results(parent_node)

            ref_in = parent_results.get("ref")
            tst_in = parent_results.get("tst")

            out_ref = op.execute(ref_in) if ref_in is not None else None
            out_tst = op.execute(tst_in) if tst_in is not None else None

            channel = parent_id.split("_")[-1]
            op_abbr = op_name[:3]
            new_id = f"{op_abbr}_{idx:02d}_{channel}"
            kind = "both"
            if out_ref is not None and out_tst is None:
                kind = "ref"
            elif out_ref is None and out_tst is not None:
                kind = "tst"

            save_dir = os.path.join(DATA_DIR, new_id)
            os.makedirs(save_dir, exist_ok=True)
            ref_path = os.path.join(save_dir, "ref.npy")
            tst_path = os.path.join(save_dir, "tst.npy")
            if out_ref is not None:
                np.save(ref_path, out_ref)
            if out_tst is not None:
                np.save(tst_path, out_tst)

            shape = (
                np.array(out_tst if out_tst is not None else out_ref).shape
                if (out_ref is not None or out_tst is not None)
                else (0,)
            )
            node = ProcessedData(
                node_id=new_id,
                parents=[parent_id],
                source_signal_id=parent_id,
                method=op_name,
                processed_data=out_tst if out_tst is not None else out_ref,
                results={"ref": out_ref, "tst": out_tst},
                meta={
                    "tool": op_name,
                    "params": params,
                    "parent": parent_id,
                    "channel": channel,
                    "kind": kind,
                    "method": op_name,
                    "saved": {"ref_path": ref_path, "tst_path": tst_path},
                },
                shape=shape,
            )
            tracker.add_node(node)

            executed_steps += 1
        except Exception as exc:
            state.dag_state.error_log.append(f"{parent_id}: {exc}")
            break

    if len(state.detailed_plan) > MAX_STEPS:
        state.dag_state.error_log.append("MAX_STEPS_EXCEEDED")

    png_path = os.path.join(DATA_DIR, f"dag_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    tracker.write_png(png_path)
    state.dag_state.graph_path = png_path + ".png"

    return {"executed_steps": executed_steps, "dag_png_path": state.dag_state.graph_path}


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    from langchain_community.chat_models import FakeListChatModel
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from phm_core import PHMState, DAGState, InputData

    os.environ["FAKE_LLM"] = "true"
    from src import model

    model._FAKE_LLM = FakeListChatModel(responses=["0"])

    instruction = "轴承故障诊断"
    sig1 = np.ones((1, 4, 1))
    sig2 = np.full((1, 4, 1), 2)
    ch1 = InputData(node_id="ch1", data={"signal": sig1}, results={"ref": sig1, "tst": sig1}, parents=[], shape=sig1.shape)
    ch2 = InputData(node_id="ch2", data={"signal": sig2}, results={"ref": sig2, "tst": sig2}, parents=[], shape=sig2.shape)
    dag = DAGState(user_instruction=instruction, channels=["ch1", "ch2"], nodes={"ch1": ch1, "ch2": ch2}, leaves=["ch1", "ch2"])
    state = PHMState(
        user_instruction=instruction,
        reference_signal=ch1,
        test_signal=ch2,
        dag_state=dag,
        detailed_plan=[
            {"op_name": "mean", "params": {"parent": "ch1"}},
            {"op_name": "mean", "params": {"parent": "ch2"}},
        ],
    )
    print({"before": list(state.dag_state.nodes.keys())})
    result = execute_agent(state)
    print({"after": list(state.dag_state.nodes.keys())})
    print(result)

