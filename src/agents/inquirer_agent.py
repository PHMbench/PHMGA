import json
from langchain_core.prompts import ChatPromptTemplate
from ..model import get_llm
from ..configuration import Configuration
from ..states.phm_states import PHMState, ProcessedData
from ..prompts.inquirer_prompt import INQUIRER_PROMPT
from ..tools.decision_schemas import OP_REGISTRY as DECISION_OPS


def generate_dag_summary(state: PHMState) -> str:
    lines = []
    for node_id, node in state.dag_state.nodes.items():
        if isinstance(node, ProcessedData):
            lines.append(f"- id:{node_id}, method:{node.method}, parents:{node.parents}")
    return "\n".join(lines) if lines else "None"


def inquirer_agent(state: PHMState) -> dict:
    """Design a comparison and decision plan based on the existing DAG."""
    llm = get_llm(Configuration())
    tools_schemas = json.dumps([cls.model_json_schema() for cls in DECISION_OPS.values()], ensure_ascii=False)
    summary = generate_dag_summary(state)
    prompt = ChatPromptTemplate.from_template(INQUIRER_PROMPT)
    chain = prompt | llm
    resp = chain.invoke({"instruction": state.user_instruction, "dag_summary": summary, "tools": tools_schemas})
    try:
        plan_json = json.loads(resp.content)
        plan = plan_json.get("plan", plan_json)
    except Exception as e:
        state.error_logs.append(f"Inquirer parse error: {e}")
        plan = []
    return {"detailed_plan": plan}


if __name__ == "__main__":
    import numpy as np
    from ..states.phm_states import DAGState, InputData

    ref = InputData(node_id="ref", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    test = InputData(node_id="test", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    dag = DAGState(user_instruction="demo", reference_root="ref", test_root="test")
    state = PHMState(
        user_instruction="diagnose",
        reference_signal=ref,
        test_signal=test,
        dag_state=dag,
    )
    out = inquirer_agent(state)
    print(out)
