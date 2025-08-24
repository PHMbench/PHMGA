from __future__ import annotations

import json
from typing import Any, Dict, Optional
import networkx as nx

from langchain_core.prompts import ChatPromptTemplate

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
    # --- 诊断性打印 ---
    print("\n--- Reflect Agent Inputs ---")
    print(f"Stage: {stage}")
    print(f"Issues Summary: '{issues_summary}'")
    print("--------------------------\n")
    # --- 结束诊断 ---

    if instruction is None or stage is None or dag_blueprint is None:
        return {"decision": "halt", "reason": "INVALID_INPUT"}

    # 1. 计算DAG的深度，作为LLM决策的上下文之一
    depth = get_dag_depth(state.dag_state)
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
    resp = chain.invoke(
        {
            "instruction": instruction,
            "stage": stage,
            "dag_blueprint": json.dumps(dag_blueprint, ensure_ascii=False),
            "issues_summary": contextual_issues, # 使用包含深度信息的上下文
            "min_depth": state.min_depth,
            "min_width": state.min_width,
            "max_depth": state.max_depth,
            "current_depth": get_dag_depth(state.dag_state)
        }
    )
    # 漂亮地打印出LLM的响应以供调试
    print("\n--- Reflect Agent LLM Response ---")

    # 从LLM响应中提取JSON字符串，移除Markdown代码块
    json_str = resp.content
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].strip()
    if "```" in json_str:
        json_str = json_str.split("```")[0].strip()

    try:
        # 假设响应内容是JSON字符串
        parsed_json = json.loads(json_str)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        # 如果不是JSON，则按原样打印原始响应
        print(resp.content)
    print("---------------------------------\n")
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
    return {"decision": decision, "reason": reason}


def reflect_agent_node(state: PHMState, *, stage: str) -> None:
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
