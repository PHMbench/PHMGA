from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.builder_quality import evaluate_builder_richness, reflection_fallback
from src.model import get_llm
from src.configuration import Configuration
from src.prompts.reflect_prompt import REFLECT_PROMPT
from src.states.phm_states import PHMState
from src.utils import get_dag_depth

VALID_DECISIONS = {"finish", "need_patch", "need_replan", "halt"}




def reflect_agent(
    *,
    instruction: Optional[str] = None,
    stage: Optional[str] = None,
    dag_blueprint: Optional[Dict[str, Any]] = None,
    issues_summary: Optional[str] = None,
    state: "PHMState", # 添加 state 以访问 DAG 信息
) -> Dict[str, str]:
    """Quality check the DAG and return a decision with reason."""
    if instruction is None or stage is None or dag_blueprint is None:
        return {"decision": "halt", "reason": "INVALID_INPUT"}

    depth = get_dag_depth(state.dag_state)
    contextual_issues = issues_summary or ""
    if not contextual_issues:
        contextual_issues = f"Execution was successful. The current DAG has a depth of {depth}."
    else:
        contextual_issues = f"{issues_summary}\nAdditionally, the current DAG has a depth of {depth}."

    runtime_config = state.runtime_config or {"llm": Configuration.from_runnable_config(None).model_dump()}
    llm = get_llm(runtime_config)
    if getattr(llm, "mode", "") == "offline_stub":
        decision, reason = reflection_fallback(state, extra_reason=contextual_issues if issues_summary else "")
    else:
        prompt = REFLECT_PROMPT.format(
            instruction=instruction,
            stage=stage,
            dag_blueprint=json.dumps(dag_blueprint, ensure_ascii=False),
            issues_summary=contextual_issues,
            min_depth=state.min_depth,
            min_width=state.min_width,
            max_depth=state.max_depth,
            current_depth=depth,
        )
        repair_prompt = (
            "Return only a JSON object with `decision` and `reason`.\n\n" + prompt
        )
        try:
            data = llm.generate_json(prompt, repair_prompt=repair_prompt)
            candidate_decision = data.get("decision", "halt")
            candidate_reason = str(data.get("reason", "") or "")
            if candidate_decision not in VALID_DECISIONS:
                candidate_decision = "halt"
                candidate_reason = "INVALID_DECISION"
            quality = evaluate_builder_richness(state)
            if quality["passes"] and depth >= state.min_depth and candidate_decision not in {"need_patch", "need_replan"}:
                decision = "finish"
                reason = candidate_reason or quality["reason"]
            elif candidate_decision == "need_replan":
                decision = "need_replan"
                reason = candidate_reason or quality["reason"]
            else:
                decision, reason = reflection_fallback(state, extra_reason=candidate_reason)
        except Exception as exc:  # pragma: no cover - defensive
            decision, reason = reflection_fallback(state, extra_reason=f"PROVIDER_FAILURE: {exc}")
    return {"decision": decision, "reason": reason}


def reflect_agent_node(state: PHMState, *, stage: str) -> Dict[str, Any]:
    """Adapter for the outer graph using :class:`PHMState`."""
    try:
        dag_blueprint = json.loads(state.tracker().export_json())
    except Exception:
        dag_blueprint = {}
    issues = "\n".join(state.dag_state.error_log)
    result = reflect_agent(
        instruction=state.user_instruction,
        stage=stage,
        dag_blueprint=dag_blueprint,
        issues_summary=issues or None,
        state=state, # 传递整个 state
    )
    needs_revision = result["decision"] != "finish"
    history = state.reflection_history + [result["reason"]]
    return {
        "needs_revision": needs_revision,
        "reflection_history": history,
        "last_reflection_decision": result["decision"],
    }


if __name__ == "__main__":
    import os
    import sys
    from langchain_community.chat_models import FakeListChatModel

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    os.environ["FAKE_LLM"] = "true"
    from src import model
    from phm_core import PHMState, DAGState, InputData

    model._FAKE_LLM = FakeListChatModel(responses=['{"decision": "finish", "reason": "ok"}'])

    instruction = "轴承故障诊断"
    ch1 = InputData(node_id="ch1", data={}, parents=[], shape=(0,))
    ch2 = InputData(node_id="ch2", data={}, parents=[], shape=(0,))
    dag = DAGState(user_instruction=instruction, channels=["ch1", "ch2"], nodes={"ch1": ch1, "ch2": ch2}, leaves=["ch1", "ch2"])
    state = PHMState(user_instruction=instruction, reference_signal=ch1, test_signal=ch2, dag_state=dag)
    print({"before": state.model_dump(exclude={"reference_signal", "test_signal"})})
    out = reflect_agent_node(state, stage="POST_PLAN")
    print({"after": out})
