from __future__ import annotations

import json
import os
import importlib.util
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.model.explainable.operator_catalog import is_contract_allowed, resolve_operator_contract
from src.prompts.plan_prompt import PLANNER_PROMPT
from src.states.phm_states import PHMState
from src.tools.signal_processing_schemas import OP_REGISTRY, get_operator
from src.utils import get_dag_depth
from src.utils.logging_setup import get_current_logger, log_event, timed

# 1. 定义期望的输出结构
class Step(BaseModel):
    """A single step in the processing plan."""

    parent: str = Field(
        ...,
        description="The ID of the parent node to which this operation should be applied."
    )
    op_name: str = Field(
        ...,
        description="The name of the operator to use, must be one of the provided tools.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="The parameters for the operator."
    )


class Plan(BaseModel):
    """The detailed processing plan."""

    plan: List[Step] = Field(
        ..., description="A list of processing steps to execute in sequence."
    )


def _dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _op_enabled_by_dependencies(op_name: str) -> bool:
    dependency_map = {
        "approximate_entropy": "nolds",
        "permutation_entropy": "antropy",
        "power_to_db": "librosa",
        "mel_spectrogram": "librosa",
        "vqt": "librosa",
        "patch": "skimage",
        "wavelet_transform": "pywt",
        "denoise_wavelet": "pywt",
        "vmd": "vmdpy",
        "emd": "emd",
    }
    dep = dependency_map.get(str(op_name or "").strip().lower())
    if not dep:
        return True
    return _dependency_available(dep)


_RM101_PRIORITY_OPS = {
    "filter",
    "hilbert_envelope",
    "fft",
    "stft",
    "band_power",
    "cross_correlation",
    "spectral_kurtosis",
    "spectral_centroid",
}
_RM101_DEPRIORITIZED_OPS = {
    "approximate_entropy",
    "permutation_entropy",
}


def _tool_priority(op_name: str, dataset_name: str | None) -> int:
    dataset = str(dataset_name or "").strip()
    if dataset == "RM_101_THU_GEARBOX":
        if op_name in _RM101_PRIORITY_OPS:
            return 0
        if op_name in _RM101_DEPRIORITIZED_OPS:
            return 2
    return 1


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_tools_description(
    dataset_name: str | None = None,
    *,
    operator_contract: str | None = None,
    enforce_tspn_closed_world: bool = True,
) -> str:
    contract = resolve_operator_contract(operator_contract)
    tool_descriptions: List[str] = []
    for op in sorted(
        OP_REGISTRY.values(),
        key=lambda item: (_tool_priority(str(getattr(item, "op_name", "") or "").strip(), dataset_name), str(getattr(item, "op_name", "") or "").strip()),
    ):
        op_name = str(getattr(op, "op_name", "") or "").strip()
        if not op_name:
            continue
        if not _op_enabled_by_dependencies(op_name):
            continue
        if enforce_tspn_closed_world and not is_contract_allowed(op_name, contract):
            continue
        schema = op.model_json_schema()
        description = schema.get("description", "No description available.")
        priority = _tool_priority(op_name, dataset_name)
        priority_text = "high" if priority == 0 else ("low" if priority == 2 else "normal")
        tool_descriptions.append(
            f"- op_name: {op_name}\n  priority: {priority_text}\n  description: {description}\n"
        )
    return "\n---\n".join(tool_descriptions)


def _dataset_policy(state: PHMState) -> tuple[str, str, int]:
    data_cfg = dict(getattr(state, "data_cfg", {}) or {})
    dataset_name = str(data_cfg.get("dataset_name") or "")
    max_ops = int(data_cfg.get("max_ops_per_iteration") or 0)
    if dataset_name == "RM_101_THU_GEARBOX":
        if max_ops <= 0:
            max_ops = 8
        hint = (
            "Prioritize gearbox-oriented operators: filter/hilbert_envelope/fft/stft/band_power/"
            "cross_correlation/spectral_kurtosis. Avoid unstable entropy-heavy branches unless required."
        )
    else:
        if max_ops <= 0:
            max_ops = 0
        hint = "Prefer diverse but valid operators; always use available tools and meaningful fs/band parameters."
    return dataset_name, hint, max_ops


_PLAN_OP_ALIASES: Dict[str, str] = {
    "spectral_entropy": "spectral_flatness",
    "hilbert": "hilbert_envelope",
    "zscore": "normalize",
    "bandpass": "filter",
    "welch": "psd",
}


def _sanitize_plan_steps(
    raw_steps: List[Dict[str, Any]],
    *,
    operator_contract: str | None = None,
    enforce_tspn_closed_world: bool = True,
) -> tuple[List[Dict[str, Any]], List[str]]:
    contract = resolve_operator_contract(operator_contract)
    sanitized: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for idx, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            warnings.append(f"Drop non-dict step at index={idx}.")
            continue
        op_name_raw = str(step.get("op_name") or "").strip()
        if not op_name_raw:
            warnings.append(f"Drop step index={idx}: missing op_name.")
            continue

        op_name = op_name_raw.lower()
        if op_name not in OP_REGISTRY:
            alias = _PLAN_OP_ALIASES.get(op_name)
            if alias and alias in OP_REGISTRY:
                step = dict(step)
                step["op_name"] = alias
                op_name = alias
                warnings.append(f"Map unknown op '{op_name_raw}' -> '{alias}' at index={idx}.")
            else:
                warnings.append(f"Drop unknown op '{op_name_raw}' at index={idx}.")
                continue

        if not _op_enabled_by_dependencies(op_name):
            warnings.append(f"Drop op '{op_name}' at index={idx}: missing runtime dependency.")
            continue
        if enforce_tspn_closed_world and not is_contract_allowed(op_name, contract):
            warnings.append(
                f"Drop op '{op_name}' at index={idx}: contract_violation ({contract.name})."
            )
            continue

        sanitized.append(step)
    return sanitized, warnings


def _normalize_llm_plan_payload(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        return {"plan": content}

    text = str(content or "").strip()
    if not text:
        return {"plan": []}

    if "```json" in text:
        text = text.split("```json", 1)[1].strip()
    if "```" in text:
        text = text.split("```", 1)[0].strip()

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return {"plan": parsed}
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Unsupported LLM plan payload type: {type(parsed).__name__}")


def plan_agent(state: PHMState) -> dict:
    """Call LLM to generate a detailed processing plan using structured output."""

    logger = get_current_logger()
    llm = get_llm()
    
    data_cfg = dict(getattr(state, "data_cfg", {}) or {})
    operator_contract = str(data_cfg.get("operator_contract") or "rm101_closed_v1").strip().lower() or "rm101_closed_v1"
    enforce_closed_world = _parse_bool(data_cfg.get("enforce_tspn_closed_world"), default=True)

    # --- MODIFIED: Generate a concise, dependency-aware tool description ---
    dataset_name, dataset_hint, max_ops_per_iteration = _dataset_policy(state)
    tools_description = _build_tools_description(
        dataset_name,
        operator_contract=operator_contract,
        enforce_tspn_closed_world=enforce_closed_world,
    )

    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    
    # 不再使用 .with_structured_output()，而是手动解析
    chain = prompt | llm

    try:
        # --- MODIFIED: Create a lightweight topology-only representation of the DAG ---
        dag_topology = {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parents": node.parents,
                    "stage": node.stage,
                    "method": getattr(node, 'method', None),
                    "shape": node.shape
                }
                for node in state.dag_state.nodes.values()
            ],
            # "leaves": state.dag_state.leaves, # Optional: include leaves if needed
        }
        dag_json = json.dumps(dag_topology, indent=2)
        
        reflection = state.reflection_history

        llm_input = {
            "instruction": state.user_instruction,
            "dag_json": dag_json, # Pass the lightweight topology
            "tools": tools_description,
            "reflection": json.dumps(reflection, indent=2),
            "min_depth": state.min_depth,
            "min_width": state.min_width,
            "max_depth": state.max_depth,
            "current_depth": get_dag_depth(state.dag_state),
            "dataset_name": dataset_name,
            "dataset_hint": dataset_hint,
            "max_ops_per_iteration": max_ops_per_iteration,
            "operator_contract": operator_contract,
            "enforce_tspn_closed_world": enforce_closed_world,
        }
        with timed(logger, event="llm_call", phase="builder", node="plan", message="plan_agent LLM invoke"):
            log_event(
                logger,
                level="INFO",
                event="llm.request",
                phase="builder",
                node="plan",
                message="Sending plan prompt to LLM.",
                payload={
                    "provider": os.getenv("LLM_PROVIDER"),
                    "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
                    "prompt": PLANNER_PROMPT,
                    "inputs": llm_input,
                },
            )
            resp = chain.invoke(llm_input)
            log_event(
                logger,
                level="INFO",
                event="llm.response",
                phase="builder",
                node="plan",
                message="Received plan response from LLM.",
                payload={"response": getattr(resp, "content", "")},
            )
        
        # 1. Parse and normalize payload into {"plan": [...]}
        plan_dict = _normalize_llm_plan_payload(getattr(resp, "content", ""))

        # 2. 手动预处理（例如，处理空的 params）
        raw_steps = (plan_dict.get("plan", []) if isinstance(plan_dict, dict) else [])
        for step_data in raw_steps:
            if not isinstance(step_data, dict):
                continue
            # Backward-compat: some older prompts put parent inside params.
            if "parent" not in step_data:
                params = step_data.get("params")
                if isinstance(params, dict) and "parent" in params:
                    step_data["parent"] = params.pop("parent")

            if step_data.get("params") in ("", None):
                step_data["params"] = {}

        sanitized_steps, sanitize_warnings = _sanitize_plan_steps(
            [s for s in raw_steps if isinstance(s, dict)],
            operator_contract=operator_contract,
            enforce_tspn_closed_world=enforce_closed_world,
        )
        if max_ops_per_iteration > 0 and len(sanitized_steps) > max_ops_per_iteration:
            dropped = len(sanitized_steps) - max_ops_per_iteration
            sanitized_steps = sanitized_steps[:max_ops_per_iteration]
            sanitize_warnings.append(
                f"Trim planner steps to max_ops_per_iteration={max_ops_per_iteration}; dropped={dropped}."
            )
        if sanitize_warnings:
            existing_logs = list(state.error_logs)
            capped = sanitize_warnings[:10]
            if len(sanitize_warnings) > 10:
                capped.append(f"... and {len(sanitize_warnings) - 10} more")
            state.error_logs = existing_logs + [f"Planner sanitize: {msg}" for msg in capped]
            log_event(
                logger,
                level="WARNING",
                event="plan.sanitize",
                phase="builder",
                node="plan",
                message="Planner output sanitized.",
                payload={"warnings": sanitize_warnings[:20], "n_warnings": len(sanitize_warnings)},
            )

        plan_dict["plan"] = sanitized_steps

        # 3. 使用 Plan.model_validate() 验证和转换
        plan_obj = Plan.model_validate(plan_dict)
        detailed_plan = [step.model_dump() for step in plan_obj.plan]

        # --- Inject sampling frequency if required ---
        fs = getattr(state, "fs", None)
        if fs is None:
            fs = getattr(state.reference_signal, "meta", {}).get("fs")

        if fs is not None:
            for step in detailed_plan:
                try:
                    op_cls = get_operator(step["op_name"])
                except KeyError:
                    continue
                if "fs" in op_cls.model_fields and "fs" not in step["params"]:
                    step["params"]["fs"] = fs

    except Exception as e:
        # 捕获 LLM 调用、解析或验证中可能出现的错误
        detailed_plan = []
        error_logs = state.error_logs + [f"Planner error: {e}"]
        state.error_logs = error_logs
        log_event(
            logger,
            level="ERROR",
            event="plan.error",
            phase="builder",
            node="plan",
            message=f"Planner failed: {e}",
            payload={"error_logs_count": len(error_logs)},
        )

    return {"detailed_plan": detailed_plan}


def run_test_with_fake_llm(state: PHMState):
    """使用 FakeLLM 测试 plan_agent。"""
    print("\n--- Testing with Fake LLM ---")
    
    initial_leaves = state.dag_state.leaves
    
    # 使用 FakeLLM 来模拟一个可预测的输出
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(
        responses=[
            json.dumps({
                "plan": [
                    {"parent": leaf, "op_name": "fft", "params": {}} for leaf in initial_leaves
                ]
            })
        ]
    )

    result = plan_agent(state)
    
    print("\n--- Fake LLM Plan Agent Output ---")
    print(json.dumps(result, indent=2))
    print("----------------------------------\n")

    # --- 输出验证 ---
    assert "detailed_plan" in result
    plan = result["detailed_plan"]
    assert isinstance(plan, list)
    assert len(plan) == len(initial_leaves)
    for i, step in enumerate(plan):
        assert "parent" in step
        assert "op_name" in step
        assert "params" in step
        assert step["parent"] == initial_leaves[i]
    
    print("✅ Fake LLM Plan Agent test passed!")

def run_test_with_real_llm(state: PHMState):
    """使用真实的 LLM 测试 plan_agent。"""
    print("\n--- Testing with Real LLM ---")
    
    initial_leaves = state.dag_state.leaves

    # 禁用 FakeLLM
    os.environ["FAKE_LLM"] = "false" 
    from src import model
    import importlib
    importlib.reload(model)

    # 调用 plan_agent
    real_result = plan_agent(state)

    print("\n--- Real LLM Plan Agent Output ---")
    print(json.dumps(real_result, indent=2))
    print("----------------------------------\n")

    # --- 对真实 LLM 的输出进行验证 ---
    assert "detailed_plan" in real_result
    real_plan = real_result["detailed_plan"]
    assert isinstance(real_plan, list)
    assert len(real_plan) > 0 
    for step in real_plan:
        assert "parent" in step
        assert "op_name" in step
        assert "params" in step
        assert step["parent"] in initial_leaves

    print("✅ Real LLM Plan Agent test passed!")


if __name__ == "__main__":
    raise SystemExit(
        "This module is not intended to be executed as a script. "
        "Use pytest (tests/test_plan_agent.py) or run the workflow via `python main.py case1 --config ...`."
    )

    dag = DAGState(
        user_instruction=instruction, 
        channels=channels, 
        nodes=initial_nodes, 
        leaves=initial_leaves
    )
    
    state = PHMState(
        user_instruction=instruction, 
        reference_signal=initial_nodes["ch1"], 
        test_signal=initial_nodes["ch1"], 
        dag_state=dag
    )
    
    print("--- Initial State for Plan Agent ---")
    print(f"Instruction: {state.user_instruction}")
    print(f"Leaf Nodes (Channels): {state.dag_state.leaves}")
    print("------------------------------------")

    # 依次运行测试
    # run_test_with_fake_llm(state)
    run_test_with_real_llm(state)
