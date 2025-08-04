from __future__ import annotations

import json
import json
from typing import Any, Dict, Optional
import networkx as nx

from langchain_core.prompts import ChatPromptTemplate

from src.model import get_llm
from src.configuration import Configuration
from src.prompts.reflect_prompt import REFLECT_PROMPT
from phm_core import PHMState


VALID_DECISIONS = {"proceed", "need_patch", "need_replan", "halt"}

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
        }
    )
    # 漂亮地打印出LLM的响应以供调试
    print("\n--- Reflect Agent LLM Response ---")
    try:
        # 假设响应内容是JSON字符串
        parsed_json = json.loads(resp.content)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        # 如果不是JSON，则按原样打印
        print(resp.content)
    print("---------------------------------\n")
    try:
        data = json.loads(resp.content)
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
    needs_revision = result["decision"] != "proceed"
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

    model._FAKE_LLM = FakeListChatModel(responses=['{"decision": "proceed", "reason": "ok"}'])

    instruction = "轴承故障诊断"
    ch1 = InputData(node_id="ch1", data={}, parents=[], shape=(0,))
    ch2 = InputData(node_id="ch2", data={}, parents=[], shape=(0,))
    dag = DAGState(user_instruction=instruction, channels=["ch1", "ch2"], nodes={"ch1": ch1, "ch2": ch2}, leaves=["ch1", "ch2"])
    state = PHMState(user_instruction=instruction, reference_signal=ch1, test_signal=ch2, dag_state=dag)
    print({"before": state.model_dump(exclude={"reference_signal", "test_signal"})})
    out = reflect_agent_node(state, stage="POST_PLAN")
    print({"after": out})
