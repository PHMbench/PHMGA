from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.model import get_llm
from src.configuration import Configuration
from src.prompts.reflect_prompt import REFLECT_PROMPT
from src.states.phm_states import PHMState
from src.utils import get_dag_depth
from src.utils.logging_setup import get_current_logger, log_event, timed

VALID_DECISIONS = {"finish", "need_patch", "need_replan", "halt"}


def _debug_enabled() -> bool:
    return os.getenv("PHM_DEBUG_REFLECT", "").strip().lower() in {"1", "true", "yes", "y"}


def reflect_agent(
    *,
    instruction: Optional[str] = None,
    stage: Optional[str] = None,
    dag_blueprint: Optional[Dict[str, Any]] = None,
    issues_summary: Optional[str] = None,
    state: "PHMState" | None = None,  # Optional for backward compatibility in offline tests
) -> Dict[str, str]:
    """Quality check the DAG and return a decision with reason."""
    logger = get_current_logger()
    if _debug_enabled():
        print("\n--- Reflect Agent Inputs ---")
        print(f"Stage: {stage}")
        print(f"Issues Summary: '{issues_summary}'")
        print("--------------------------\n")

    if instruction is None or stage is None or dag_blueprint is None:
        return {"decision": "halt", "reason": "INVALID_INPUT"}

    # 1. 计算DAG的深度，作为LLM决策的上下文之一
    depth = get_dag_depth(state.dag_state) if state is not None else 0
    if _debug_enabled():
        print(f"\n--- Current DAG Depth for Reflection: {depth} ---\n")

    # 2. 准备给LLM的上下文，包括深度信息
    # 即使没有错误，也把深度信息加进去，让LLM判断是否需要继续迭代
    contextual_issues = issues_summary or ""
    if not contextual_issues:
        contextual_issues = f"Execution was successful. The current DAG has a depth of {depth}."
    else:
        contextual_issues = f"{issues_summary}\nAdditionally, the current DAG has a depth of {depth}."


    # 3. 总是调用LLM进行反思，而不是使用硬编码规则
    # LLM将基于指令、阶段、DAG结构和深度等信息，做出更全面的决策
    llm = get_llm(Configuration.from_runnable_config(None))
    prompt = ChatPromptTemplate.from_template(REFLECT_PROMPT)
    chain = prompt | llm
    llm_input = {
        "instruction": instruction,
        "stage": stage,
        "dag_blueprint": json.dumps(dag_blueprint, ensure_ascii=False),
        "issues_summary": contextual_issues, # 使用包含深度信息的上下文
        "min_depth": state.min_depth if state is not None else 0,
        "min_width": state.min_width if state is not None else 0,
        "max_depth": state.max_depth if state is not None else 999,
        "current_depth": get_dag_depth(state.dag_state) if state is not None else depth,
    }
    with timed(logger, event="llm_call", phase="builder", node="reflect", message="reflect_agent LLM invoke"):
        log_event(
            logger,
            level="INFO",
            event="llm.request",
            phase="builder",
            node="reflect",
            message="Sending reflect prompt to LLM.",
            payload={
                "provider": os.getenv("LLM_PROVIDER"),
                "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
                "prompt": REFLECT_PROMPT,
                "inputs": llm_input,
            },
        )
        resp = chain.invoke(llm_input)
        log_event(
            logger,
            level="INFO",
            event="llm.response",
            phase="builder",
            node="reflect",
            message="Received reflect response from LLM.",
            payload={"response": getattr(resp, "content", "")},
        )
    if _debug_enabled():
        print("\n--- Reflect Agent LLM Response ---")

        # From the LLM response, extract the JSON string and remove Markdown code fences.
        json_str_dbg = resp.content
        if "```json" in json_str_dbg:
            json_str_dbg = json_str_dbg.split("```json")[1].strip()
        if "```" in json_str_dbg:
            json_str_dbg = json_str_dbg.split("```")[0].strip()

        try:
            parsed_json = json.loads(json_str_dbg)
            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(resp.content)
        print("---------------------------------\n")

    # From the LLM response, extract the JSON string and remove Markdown code fences.
    json_str = resp.content
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].strip()
    if "```" in json_str:
        json_str = json_str.split("```")[0].strip()

    try:
        # 使用清理后的字符串进行解析
        data = json.loads(json_str)
        decision = data.get("decision", "halt")
        reason = data.get("reason", "")
        if decision not in VALID_DECISIONS:
            decision = "halt"
            reason = "INVALID_DECISION"
    except Exception as exc:  # pragma: no cover - defensive
        decision = "halt"
        reason = f"PARSE_ERROR: {exc}"
        log_event(
            logger,
            level="ERROR",
            event="reflect.parse_error",
            phase="builder",
            node="reflect",
            message=reason,
        )
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
    return {"needs_revision": needs_revision, "reflection_history": history}


if __name__ == "__main__":
    raise SystemExit(
        "This module is not intended to be executed as a script. "
        "Use pytest (tests/test_reflect_agent.py) or run the workflow via `python main.py case1 --config ...`."
    )
