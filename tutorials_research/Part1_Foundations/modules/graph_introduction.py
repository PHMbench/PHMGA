"""
LangGraph Introduction for Research Applications

Basic introduction to LangGraph concepts using research scenarios.
Focuses on simple, practical patterns for academic workflows.
"""

from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict
from dataclasses import dataclass
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class WorkflowStage(Enum):
    """Stages in a research workflow"""
    INPUT = "input"
    ANALYZE = "analyze"
    PROCESS = "process"
    VALIDATE = "validate"
    OUTPUT = "output"


@dataclass
class ResearchTask:
    """Represents a research task to be processed"""
    task_id: str
    content: str
    task_type: str
    priority: int = 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SimpleResearchState(TypedDict):
    """
    Simple state for research workflows.
    
    This demonstrates basic LangGraph state management concepts
    without overwhelming complexity.
    """
    # Core content
    task: ResearchTask
    current_stage: str
    
    # Processing results
    analysis_result: str
    processed_content: str
    validation_status: str
    final_output: str
    
    # Metadata
    processing_history: List[Dict[str, Any]]
    error_messages: List[str]


class CodeConversionState(TypedDict):
    """
    State for code-to-LaTeX conversion workflow.
    
    Demonstrates a more specific research application.
    """
    # Input
    original_code: str
    language: str
    conversion_type: str
    
    # Processing stages
    parsed_code: Dict[str, Any]
    latex_output: str
    validation_result: Dict[str, Any]
    
    # Results
    final_latex: str
    confidence_score: float
    metadata: Dict[str, Any]


def create_simple_research_workflow() -> StateGraph:
    """
    Create a basic research workflow using LangGraph.
    
    Workflow: Input → Analyze → Process → Validate → Output
    
    Returns:
        Compiled LangGraph workflow
    """
    
    def input_node(state: SimpleResearchState) -> Dict[str, Any]:
        """Initial processing of research task"""
        task = state["task"]
        
        print(f"📥 INPUT: Processing task '{task.task_id}'")
        print(f"   Content: {task.content[:100]}...")
        
        # Update processing history
        history_entry = {
            "stage": "input",
            "timestamp": "now",  # In real use, use datetime
            "details": f"Started processing {task.task_type} task"
        }
        
        return {
            "current_stage": "input_complete",
            "processing_history": state.get("processing_history", []) + [history_entry]
        }
    
    def analyze_node(state: SimpleResearchState) -> Dict[str, Any]:
        """Analyze the research task"""
        task = state["task"]
        
        print(f"🔍 ANALYZE: Analyzing task type '{task.task_type}'")
        
        # Simple analysis logic
        if task.task_type == "code_conversion":
            analysis = f"Code conversion task detected. Language: {task.metadata.get('language', 'unknown')}"
        elif task.task_type == "literature_review":
            analysis = "Literature review task detected. Preparing for search and summarization."
        else:
            analysis = f"General {task.task_type} task. Applying standard processing."
        
        history_entry = {
            "stage": "analyze",
            "timestamp": "now",
            "details": analysis
        }
        
        return {
            "current_stage": "analyze_complete",
            "analysis_result": analysis,
            "processing_history": state["processing_history"] + [history_entry]
        }
    
    def process_node(state: SimpleResearchState) -> Dict[str, Any]:
        """Main processing logic"""
        task = state["task"]
        analysis = state["analysis_result"]
        
        print(f"⚙️ PROCESS: Executing main processing for {task.task_type}")
        
        # Simulate processing based on task type
        if task.task_type == "code_conversion":
            latex_content = task.content.replace('np.', '\\\\').replace('**', '^')
            processed = f"LaTeX conversion: ${latex_content}$"
        elif task.task_type == "literature_review":
            processed = f"Literature summary: Key findings from analysis of '{task.content}'"
        else:
            processed = f"Processed result: {task.content.upper()}"
        
        history_entry = {
            "stage": "process",
            "timestamp": "now",
            "details": f"Applied {task.task_type} processing logic"
        }
        
        return {
            "current_stage": "process_complete",
            "processed_content": processed,
            "processing_history": state["processing_history"] + [history_entry]
        }
    
    def validate_node(state: SimpleResearchState) -> Dict[str, Any]:
        """Validate the processed results"""
        processed = state["processed_content"]
        
        print(f"✅ VALIDATE: Checking output quality")
        
        # Simple validation logic
        if len(processed) < 10:
            validation = "Warning: Output seems too short"
        elif "error" in processed.lower():
            validation = "Error: Processing may have failed"
        else:
            validation = "Validation passed: Output appears correct"
        
        history_entry = {
            "stage": "validate",
            "timestamp": "now",
            "details": validation
        }
        
        return {
            "current_stage": "validate_complete",
            "validation_status": validation,
            "processing_history": state["processing_history"] + [history_entry]
        }
    
    def output_node(state: SimpleResearchState) -> Dict[str, Any]:
        """Finalize and format output"""
        processed = state["processed_content"]
        validation = state["validation_status"]
        
        print(f"📤 OUTPUT: Finalizing results")
        
        # Create final output
        final_output = f"RESEARCH RESULT:\\n{processed}\\n\\nValidation: {validation}"
        
        history_entry = {
            "stage": "output",
            "timestamp": "now",
            "details": "Workflow completed successfully"
        }
        
        return {
            "current_stage": "complete",
            "final_output": final_output,
            "processing_history": state["processing_history"] + [history_entry]
        }
    
    # Build the graph
    builder = StateGraph(SimpleResearchState)
    
    # Add nodes
    builder.add_node("input", input_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("process", process_node)
    builder.add_node("validate", validate_node)
    builder.add_node("output", output_node)
    
    # Define the flow
    builder.add_edge(START, "input")
    builder.add_edge("input", "analyze")
    builder.add_edge("analyze", "process")
    builder.add_edge("process", "validate")
    builder.add_edge("validate", "output")
    builder.add_edge("output", END)
    
    # Compile the graph
    return builder.compile()


def create_code_conversion_workflow(converter_agent) -> StateGraph:
    """
    Create a specialized workflow for code-to-LaTeX conversion.
    
    This demonstrates a more sophisticated LangGraph pattern
    with conditional logic and specialized processing.
    
    Args:
        converter_agent: CodeToLatexAgent instance
        
    Returns:
        Compiled LangGraph workflow for code conversion
    """
    
    def parse_code(state: CodeConversionState) -> Dict[str, Any]:
        """Parse and analyze the input code"""
        code = state["original_code"]
        language = state["language"]
        
        print(f"🔍 PARSE: Analyzing {language} code")
        
        # Simple parsing simulation
        parsed_info = {
            "functions": [],
            "variables": [],
            "complexity": "simple"
        }
        
        # Detect functions
        if "def " in code or "function" in code:
            parsed_info["functions"] = ["detected_function"]
            parsed_info["complexity"] = "moderate"
        
        # Detect mathematical operations
        if any(op in code for op in ["**", "sqrt", "sin", "cos", "exp"]):
            parsed_info["complexity"] = "mathematical"
        
        return {
            "parsed_code": parsed_info
        }
    
    def convert_code(state: CodeConversionState) -> Dict[str, Any]:
        """Convert code to LaTeX using the agent"""
        code = state["original_code"]
        conversion_type = state["conversion_type"]
        
        print(f"🔄 CONVERT: Converting to {conversion_type} LaTeX")
        
        # Use the converter agent
        result = converter_agent.convert_expression(code)
        
        return {
            "latex_output": result.output,
            "confidence_score": result.confidence,
            "metadata": result.metadata
        }
    
    def validate_latex(state: CodeConversionState) -> Dict[str, Any]:
        """Validate the LaTeX output"""
        latex = state["latex_output"]
        
        print("✅ VALIDATE: Checking LaTeX formatting")
        
        validation = {
            "has_math_delimiters": "$" in latex or "\\\\(" in latex,
            "has_latex_commands": "\\\\" in latex,
            "length_reasonable": 10 <= len(latex) <= 1000,
            "confidence": state["confidence_score"]
        }
        
        validation["overall_valid"] = all([
            validation["has_math_delimiters"],
            validation["has_latex_commands"],
            validation["length_reasonable"],
            validation["confidence"] > 0.5
        ])
        
        return {
            "validation_result": validation
        }
    
    def finalize_output(state: CodeConversionState) -> Dict[str, Any]:
        """Finalize the conversion output"""
        latex = state["latex_output"]
        validation = state["validation_result"]
        
        print("📋 FINALIZE: Preparing final output")
        
        if validation["overall_valid"]:
            final_latex = latex
        else:
            # Fallback to simple formatting if validation fails
            code = state["original_code"]
            final_latex = f"\\texttt{{{code}}}"  # Format as code if conversion failed
        
        metadata = {
            "original_code": state["original_code"],
            "conversion_successful": validation["overall_valid"],
            "confidence": state["confidence_score"],
            "validation_details": validation
        }
        
        return {
            "final_latex": final_latex,
            "metadata": metadata
        }
    
    # Build the conversion workflow
    builder = StateGraph(CodeConversionState)
    
    # Add nodes
    builder.add_node("parse", parse_code)
    builder.add_node("convert", convert_code)
    builder.add_node("validate", validate_latex)
    builder.add_node("finalize", finalize_output)
    
    # Define the flow
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "convert")
    builder.add_edge("convert", "validate")
    builder.add_edge("validate", "finalize")
    builder.add_edge("finalize", END)
    
    return builder.compile()


def demonstrate_basic_workflow():
    """Demonstrate basic LangGraph workflow concepts"""
    
    print("🎓 LANGGRAPH BASICS FOR RESEARCH")
    print("=" * 50)
    
    print("\n🔄 Core Concepts:")
    print("1. STATE: Data that flows between processing steps")
    print("2. NODES: Individual processing functions") 
    print("3. EDGES: Connections between nodes")
    print("4. GRAPH: Complete workflow definition")
    
    print("\n📊 Research Workflow Pattern:")
    print("   INPUT → ANALYZE → PROCESS → VALIDATE → OUTPUT")
    
    print("\n🎯 Benefits for Research:")
    print("• Reproducible workflows")
    print("• Clear processing steps")
    print("• Error handling and validation")
    print("• Easy to modify and extend")
    
    print("\n💡 When to Use LangGraph:")
    print("• Multi-step research processes")
    print("• Need for workflow visualization")
    print("• Complex decision logic")
    print("• Collaboration between different processing steps")


def run_example_workflow():
    """Run a simple example workflow"""
    
    print("\\n🧪 RUNNING EXAMPLE WORKFLOW")
    print("=" * 40)
    
    # Create the workflow
    workflow = create_simple_research_workflow()
    
    # Create a sample task
    sample_task = ResearchTask(
        task_id="example_001",
        content="np.sqrt(x**2 + y**2)",
        task_type="code_conversion",
        metadata={"language": "python"}
    )
    
    # Initial state
    initial_state = SimpleResearchState(
        task=sample_task,
        current_stage="starting",
        analysis_result="",
        processed_content="",
        validation_status="",
        final_output="",
        processing_history=[],
        error_messages=[]
    )
    
    try:
        # Run the workflow
        result = workflow.invoke(initial_state)
        
        print("\\n📋 WORKFLOW RESULTS:")
        print(f"Final Stage: {result['current_stage']}")
        print(f"Output: {result['final_output']}")
        print(f"Steps Completed: {len(result['processing_history'])}")
        
        print("\\n📚 Processing History:")
        for i, step in enumerate(result['processing_history']):
            print(f"  {i+1}. {step['stage'].upper()}: {step['details']}")
    
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        print("💡 Make sure you have the required dependencies installed")


if __name__ == "__main__":
    demonstrate_basic_workflow()
    
    print("\\n" + "="*50)
    print("🧪 EXAMPLE WORKFLOW EXECUTION")
    print("="*50)
    
    run_example_workflow()
    
    print("\\n" + "="*50)
    print("🔗 NEXT STEPS")
    print("="*50)
    print("To use with actual LLM agent:")
    print("""
from llm_providers import create_research_llm
from code_to_latex import create_research_converter
from graph_introduction import create_code_conversion_workflow

# Set up LLM and converter
llm = create_research_llm('google')
converter = create_research_converter(llm)

# Create and run workflow
workflow = create_code_conversion_workflow(converter)
result = workflow.invoke({
    "original_code": "np.sqrt(x**2 + y**2)",
    "language": "python",
    "conversion_type": "inline"
})

print(f"Final LaTeX: {result['final_latex']}")
""")