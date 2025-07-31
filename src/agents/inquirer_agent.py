
import json
from langchain_core.prompts import ChatPromptTemplate
from ..model import get_llm
from ..states.phm_states import PHMState, ProcessedData
from ..tools.comparator_tool import compare_processed_nodes
from ..schemas.insight_schema import AnalysisInsight
from ..prompts.inquirer_prompt import INQUIRER_PROMPT


def generate_dag_summary(state: PHMState) -> str:
    """Creates a concise summary of the DAG for the LLM."""
    summary_lines = []
    for node_id, node in state.dag_state.nodes.items():
        if isinstance(node, ProcessedData):
            # Example line: "- node_id: proc_123abcf, method: fft_analysis, parents: [ref_root_node_01]"
            summary_lines.append(
                f"- node_id: {node.node_id}, method: {getattr(node, 'method', 'N/A')}, parents: {node.parents}"
            )
    return "\n".join(summary_lines) if summary_lines else "No processed nodes available."

def inquirer_agent(state: PHMState) -> dict:
    """
    Intelligently inspects the final DAG and calls an LLM to decide which nodes to compare,
    then executes those comparisons to generate insights.
    """
    print("\n--- Inquirer Agent: Inspecting DAG to generate insights... ---")
    llm = get_llm()
    
    # Bind the comparison tool to the LLM
    try:
        llm_with_tools = llm.bind_tools([compare_processed_nodes])
    except NotImplementedError:
        print("Warning: LLM does not support tool binding. Inquirer will have limited function.")
        return {"insights": []}

    dag_summary = generate_dag_summary(state)
    
    prompt = ChatPromptTemplate.from_template(INQUIRER_PROMPT)
    
    chain = prompt | llm_with_tools
    
    ai_msg = chain.invoke({
        "instruction": state.user_instruction,
        "dag_summary": dag_summary
    })

    insights: list[AnalysisInsight] = []
    if not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls:
        print("--- Inquirer Agent: LLM did not suggest any comparisons. ---")
        return {"insights": []}

    print(f"--- Inquirer Agent: LLM suggested {len(ai_msg.tool_calls)} comparison(s). Executing... ---")
    for call in ai_msg.tool_calls:
        if call.get("name") == compare_processed_nodes.__name__:
            try:
                args = call.get("args", {})
                # The tool needs the full state to access node data
                args["state"] = state
                insight = compare_processed_nodes(**args)
                insights.append(insight)
                print(f"  - Comparison successful for: {args.get('reference_node_id')} vs {args.get('test_node_id')}")
            except Exception as e:
                print(f"  - Error executing comparison for args {args}: {e}")

    return {"insights": insights}
