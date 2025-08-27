"""
Multi-Provider Research Graph - Reference Architecture

Complete LangGraph workflow adapted from Google's reference implementation
for multi-provider LLM compatibility while maintaining production patterns.
"""

import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Add path for Part 1 LLM providers
sys.path.append('../Part1_Foundations/modules')
from llm_providers import create_research_llm, LLMProvider

from state_schemas import OverallState, QueryGenerationState, WebSearchState, ReflectionState
from configuration import Configuration
from tools_and_schemas import SearchQueryList, Reflection

# Import function-based nodes
from query_generator import generate_query, get_research_topic
from web_searcher import web_research
from reflection_agent import reflection
from prompts import get_current_date, format_answer_prompt


# Core LangGraph nodes (reference pattern)
def continue_to_web_research(state: QueryGenerationState):
    """
    Send the search queries to web research nodes (reference pattern).
    
    Creates Send operations to spawn multiple parallel web research nodes,
    one for each generated query.
    """
    return [
        Send("web_research", {"search_query": search_query["query"], "id": str(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def evaluate_research(state: ReflectionState, config: RunnableConfig = None) -> str:
    """
    Route research flow based on reflection results (reference pattern).
    
    Controls the research loop by deciding whether to continue gathering information
    or finalize the summary based on reflection analysis and iteration limits.
    
    Args:
        state: Current reflection state with sufficiency analysis
        config: Configuration for runnable, including max_research_loops
        
    Returns:
        String indicating next node ("web_research" or "finalize_answer")
        or Send list for follow-up queries
    """
    from configuration import Configuration
    
    # Get configuration (with defaults for tutorial)
    if config:
        try:
            configurable = Configuration.from_runnable_config(config)
        except:
            configurable = Configuration()
    else:
        configurable = Configuration()
    
    max_research_loops = configurable.max_research_loops
    
    # Check if research is sufficient or max loops reached
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        # Generate Send operations for follow-up queries
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": str(state["number_of_ran_queries"] + int(idx)),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig = None) -> OverallState:
    """
    Finalize the research summary with comprehensive answer synthesis.
    
    Creates the final output by synthesizing all research findings into a
    well-structured answer with proper citations and source references.
    
    Args:
        state: Current graph state with all research results
        config: Configuration for the runnable
        
    Returns:
        Dictionary with final answer and processed sources
    """
    from configuration import Configuration
    
    # Get configuration
    if config:
        try:
            configurable = Configuration.from_runnable_config(config)
        except:
            configurable = Configuration()
    else:
        configurable = Configuration()
    
    # Create LLM for final synthesis
    try:
        llm = create_research_llm(
            provider_name="auto",
            temperature=0.0,  # Low temperature for consistent synthesis
            fast_mode=False
        )
    except Exception as e:
        print(f"⚠️ Failed to create LLM for synthesis: {e}")
        # Create fallback answer
        research_topic = get_research_topic(state["messages"])
        fallback_content = f"Research completed for: {research_topic}\n\nTotal sources gathered: {len(state.get('sources_gathered', []))}"
        
        return {
            "messages": [AIMessage(content=fallback_content)],
            "sources_gathered": _process_unique_sources(state)
        }
    
    # Format the synthesis prompt
    research_topic = get_research_topic(state["messages"])
    summaries = state.get("web_research_result", [])
    
    formatted_prompt = format_answer_prompt(
        research_topic=research_topic,
        summaries=summaries,
        style="comprehensive"
    )
    
    try:
        # Generate final comprehensive answer
        result = llm.invoke(formatted_prompt)
        
        # Process and deduplicate sources
        unique_sources = _process_unique_sources(state)
        
        # Replace short URLs with original URLs in the answer
        processed_content = _process_source_urls(result.content, unique_sources)
        
        return {
            "messages": [AIMessage(content=processed_content)],
            "sources_gathered": unique_sources
        }
        
    except Exception as e:
        print(f"⚠️ Answer synthesis failed: {e}")
        
        # Fallback synthesis
        research_topic = get_research_topic(state["messages"])
        summaries = state.get("web_research_result", [])
        
        fallback_content = f"# Research Summary: {research_topic}\n\n"
        if summaries:
            for i, summary in enumerate(summaries[:3], 1):
                fallback_content += f"## Finding {i}\n{summary[:300]}...\n\n"
        else:
            fallback_content += "Research completed but synthesis failed."
        
        return {
            "messages": [AIMessage(content=fallback_content)],
            "sources_gathered": _process_unique_sources(state)
        }


# Helper functions for research workflow
def _process_unique_sources(state: OverallState) -> List[Dict[str, Any]]:
    """Process and deduplicate sources from research state"""
    
    sources = state.get("sources_gathered", [])
    unique_sources = []
    seen_urls = set()
    
    for source in sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)
    
    return unique_sources[:20]  # Limit to top 20 sources


def _process_source_urls(content: str, sources: List[Dict[str, Any]]) -> str:
    """Replace short URLs with original URLs in content"""
    
    processed_content = content
    
    for source in sources:
        short_url = source.get("short_url", "")
        original_url = source.get("url", "")
        
        if short_url and original_url and short_url in processed_content:
            processed_content = processed_content.replace(short_url, original_url)
    
    return processed_content


# Create the research graph (reference pattern)
def create_research_graph() -> StateGraph:
    """
    Create the complete research workflow graph.
    
    Follows Google's reference architecture patterns with multi-provider support.
    Uses function-based nodes and proper Send/conditional routing.
    
    Returns:
        Compiled StateGraph ready for research execution
    """
    
    # Create graph builder
    builder = StateGraph(OverallState, config_schema=Configuration)
    
    # Add all nodes (function-based, reference pattern)
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)
    
    # Set entry point
    builder.add_edge(START, "generate_query")
    
    # Add conditional edge for parallel web research
    builder.add_conditional_edges(
        "generate_query", continue_to_web_research, ["web_research"]
    )
    
    # Web research flows to reflection
    builder.add_edge("web_research", "reflection")
    
    # Conditional routing based on reflection results
    builder.add_conditional_edges(
        "reflection", evaluate_research, ["web_research", "finalize_answer"]
    )
    
    # Final answer ends the workflow
    builder.add_edge("finalize_answer", END)
    
    # Compile the graph
    return builder.compile(name="multi-provider-research-agent")


# Research execution functions
def conduct_research(research_question: str, config: Optional[Configuration] = None) -> Dict[str, Any]:
    """
    Conduct complete research workflow for a given question.
    
    Args:
        research_question: The research question to investigate
        config: Optional research configuration
        
    Returns:
        Complete research results with final answer and metadata
    """
    
    print(f"🔬 Starting multi-provider research workflow")
    print(f"Question: '{research_question}'")
    
    # Create or use provided configuration
    if config is None:
        config = Configuration()
    
    # Create the graph
    graph = create_research_graph()
    
    # Initialize state
    initial_state = OverallState(
        messages=[HumanMessage(content=research_question)],
        search_query=[],
        web_research_result=[],
        sources_gathered=[],
        initial_search_query_count=config.number_of_initial_queries,
        max_research_loops=config.max_research_loops,
        research_loop_count=0,
        reasoning_model=config.answer_model
    )
    
    # Execute research workflow
    try:
        print("🚀 Executing research workflow...")
        
        # Run the graph with configuration
        final_state = graph.invoke(
            initial_state,
            config={"configurable": config.model_dump()}
        )
        
        # Extract results
        messages = final_state.get("messages", [])
        final_answer = messages[-1].content if messages else "No answer generated"
        sources = final_state.get("sources_gathered", [])
        
        print(f"✅ Research completed successfully")
        print(f"📊 Total sources: {len(sources)}")
        print(f"📝 Answer length: {len(final_answer)} characters")
        
        return {
            "success": True,
            "research_question": research_question,
            "final_answer": final_answer,
            "sources": sources,
            "total_sources": len(sources),
            "research_loops": final_state.get("research_loop_count", 0),
            "configuration": config.get_summary()
        }
        
    except Exception as e:
        print(f"❌ Research workflow failed: {e}")
        
        return {
            "success": False,
            "research_question": research_question,
            "error": str(e),
            "configuration": config.get_summary()
        }


# Legacy compatibility class
class ResearchWorkflowGraph:
    """
    Legacy compatibility wrapper for existing tutorial code.
    Wraps the function-based graph architecture.
    """
    
    def __init__(self, llm, config=None):
        self.llm = llm  # Keep for compatibility but not used
        self.config = config if isinstance(config, Configuration) else Configuration()
        self.graph = create_research_graph()
    
    def conduct_research(self, research_question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Legacy compatibility method"""
        
        # Use the function-based approach
        result = conduct_research(research_question, self.config)
        
        # Add session_id if provided
        if session_id:
            result["session_id"] = session_id
        
        # Convert to legacy format
        if result["success"]:
            legacy_result = {
                "success": True,
                "final_answer": result["final_answer"],
                "research_question": result["research_question"],
                "iterations_completed": result["research_loops"],
                "total_sources_found": result["total_sources"],
                "confidence_score": 0.8,  # Default confidence
                "research_summary": {
                    "total_iterations": result["research_loops"],
                    "total_sources": result["total_sources"],
                    "execution_time": 0.0
                },
                "sources": result["sources"]
            }
        else:
            legacy_result = {
                "success": False,
                "error": result["error"],
                "research_question": result["research_question"]
            }
        
        return legacy_result


# Factory function for workflow creation
def create_research_workflow(llm=None, config: Optional[Configuration] = None) -> ResearchWorkflowGraph:
    """
    Factory function for creating research workflow (legacy compatibility).
    
    Args:
        llm: LLM instance (kept for compatibility, auto-created internally)
        config: Optional research configuration
        
    Returns:
        ResearchWorkflowGraph instance
    """
    
    if config is None:
        config = Configuration()
    
    return ResearchWorkflowGraph(llm, config)


if __name__ == "__main__":
    print("🔬 MULTI-PROVIDER RESEARCH GRAPH")
    print("=" * 35)
    
    print("\n✅ Reference Architecture Features:")
    features = [
        "Function-based LangGraph nodes (Google reference pattern)",
        "Multi-provider LLM support via Part 1 integration",
        "Parallel web research with Send operations",
        "Conditional routing based on reflection analysis",
        "Production-tested workflow patterns",
        "Legacy compatibility for existing tutorial code"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    print("\n🔄 Workflow Architecture:")
    workflow_steps = [
        "START → generate_query: Create diverse search queries",
        "generate_query → web_research: Parallel search execution",
        "web_research → reflection: Analyze research completeness",
        "reflection → [continue|finalize]: Route based on sufficiency",
        "finalize_answer → END: Synthesize comprehensive answer"
    ]
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print("\n📊 Multi-Provider Benefits:")
    benefits = [
        "Works with any LLM provider from Part 1 (Google, OpenAI, etc.)",
        "Intelligent search strategy selection (native vs simulated)",
        "Graceful fallbacks when APIs unavailable",
        "Provider-specific prompt optimizations",
        "Cost-effective with fast/slow model selection"
    ]
    
    for benefit in benefits:
        print(f"   • {benefit}")
    
    print("\n🔧 Usage Examples:")
    print("""
    # Function-based (recommended)
    result = conduct_research("What are recent advances in AI?")
    
    # With custom configuration
    config = Configuration.for_academic_research()
    result = conduct_research("Research question", config)
    
    # Legacy compatibility
    workflow = create_research_workflow()
    result = workflow.conduct_research("Research question")
    """)
    
    print("\n✅ Production-ready multi-provider research system loaded!")