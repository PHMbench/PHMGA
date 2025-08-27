"""
Research Graph Workflow

Complete LangGraph workflow for reflective research using multi-provider LLM support.
Adapted from Gemini example to work with any LLM provider from Part 1.
"""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from state_schemas import (
    OverallResearchState, 
    ResearchConfiguration,
    create_initial_research_state,
    validate_research_state
)
from query_generator import ResearchQueryGenerator
from web_searcher import WebSearchExecutor
from reflection_agent import ResearchReflectionAgent


class ResearchWorkflowGraph:
    """
    Complete research workflow using reflection-based iterative improvement.
    
    Integrates query generation, web search, reflection, and synthesis
    into a comprehensive research system.
    """
    
    def __init__(self, llm, config: Optional[ResearchConfiguration] = None):
        self.llm = llm
        self.config = config or ResearchConfiguration()
        
        # Initialize specialized components
        self.query_generator = ResearchQueryGenerator(llm, config)
        self.web_searcher = WebSearchExecutor(config)
        self.reflection_agent = ResearchReflectionAgent(llm, config)
        
        # Build the graph
        self.graph = self._build_research_graph()
    
    def _build_research_graph(self) -> StateGraph:
        """Build the complete research workflow graph"""
        
        # Create the graph
        builder = StateGraph(OverallResearchState)
        
        # Add nodes
        builder.add_node("generate_queries", self._generate_initial_queries)
        builder.add_node("execute_search", self._execute_web_search)
        builder.add_node("reflect_on_research", self._reflect_on_completeness)
        builder.add_node("generate_follow_ups", self._generate_follow_up_queries)
        builder.add_node("execute_follow_up_search", self._execute_follow_up_search)
        builder.add_node("synthesize_answer", self._synthesize_final_answer)
        
        # Define workflow
        builder.add_edge(START, "generate_queries")
        builder.add_edge("generate_queries", "execute_search")
        builder.add_edge("execute_search", "reflect_on_research")
        
        # Conditional edge for research loops
        builder.add_conditional_edges(
            "reflect_on_research",
            self._decide_next_step,
            {
                "continue_research": "generate_follow_ups",
                "synthesize": "synthesize_answer"
            }
        )
        
        builder.add_edge("generate_follow_ups", "execute_follow_up_search")
        builder.add_edge("execute_follow_up_search", "reflect_on_research")
        builder.add_edge("synthesize_answer", END)
        
        return builder.compile()
    
    def conduct_research(self, research_question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct complete research workflow for a given question.
        
        Args:
            research_question: The research question to investigate
            session_id: Optional session identifier
            
        Returns:
            Complete research results with final answer and metadata
        """
        
        print(f"🔬 Starting comprehensive research workflow")
        print(f"Question: '{research_question}'")
        
        # Initialize state
        if session_id is None:
            session_id = f"research_{uuid.uuid4().hex[:8]}"
        
        initial_state = create_initial_research_state(
            research_question=research_question,
            config=self.config,
            session_id=session_id
        )
        
        # Validate initial state
        validation_errors = validate_research_state(initial_state)
        if validation_errors:
            print(f"⚠️ State validation errors: {validation_errors}")
        
        # Execute research workflow
        start_time = time.time()
        try:
            final_state = self.graph.invoke(initial_state)
            execution_time = time.time() - start_time
            
            # Update final execution time
            final_state["total_execution_time"] = execution_time
            
            # Prepare results
            results = self._prepare_final_results(final_state)
            
            print(f"✅ Research completed in {execution_time:.1f}s")
            print(f"📊 Total iterations: {final_state['research_loop_count']}")
            print(f"📚 Total sources: {len(final_state['sources_gathered'])}")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Research workflow failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "session_id": session_id
            }
    
    def _generate_initial_queries(self, state: OverallResearchState) -> Dict[str, Any]:
        """Generate initial search queries from research question"""
        
        print("🎯 Generating initial search queries...")
        
        try:
            query_result = self.query_generator.generate_initial_queries(
                state["research_question"]
            )
            
            print(f"   Generated {len(query_result.queries)} queries")
            for i, query in enumerate(query_result.queries, 1):
                print(f"   {i}. {query}")
            
            return {
                "search_queries": query_result.queries,
                "current_query_batch": query_result.queries.copy(),
                "messages": [AIMessage(content=f"Generated {len(query_result.queries)} initial search queries")]
            }
            
        except Exception as e:
            print(f"   ❌ Query generation failed: {e}")
            fallback_queries = [state["research_question"]]
            
            return {
                "search_queries": fallback_queries,
                "current_query_batch": fallback_queries,
                "messages": [AIMessage(content=f"Using fallback query due to error: {e}")]
            }
    
    def _execute_web_search(self, state: OverallResearchState) -> Dict[str, Any]:
        """Execute web search for current query batch"""
        
        queries = state.get("current_query_batch", [])
        print(f"🔍 Executing web search for {len(queries)} queries...")
        
        try:
            search_results = self.web_searcher.execute_parallel_search(queries)
            
            # Extract findings and sources
            all_findings = []
            all_sources = []
            
            for query, sources in search_results.items():
                for source in sources:
                    # Create finding from source
                    finding = f"From {source.source_type} source: {source.title} - {source.snippet[:200]}..."
                    all_findings.append(finding)
                    all_sources.append(source)
            
            print(f"   Found {len(all_sources)} sources across all queries")
            
            # Get search statistics
            search_stats = self.web_searcher.get_search_statistics(search_results)
            print(f"   Average relevance: {search_stats['average_relevance']}")
            print(f"   Source types: {search_stats['source_types']}")
            
            return {
                "web_research_results": all_findings,
                "sources_gathered": all_sources,
                "all_findings": all_findings
            }
            
        except Exception as e:
            print(f"   ❌ Web search failed: {e}")
            return {
                "web_research_results": [f"Search failed: {e}"],
                "sources_gathered": [],
                "all_findings": [f"Search error: {e}"]
            }
    
    def _reflect_on_completeness(self, state: OverallResearchState) -> Dict[str, Any]:
        """Reflect on research completeness and identify gaps"""
        
        current_iteration = state.get("research_loop_count", 0)
        print(f"🤔 Reflecting on research completeness (iteration {current_iteration + 1})...")
        
        try:
            reflection_result = self.reflection_agent.analyze_research_completeness(
                original_question=state["research_question"],
                research_findings=state.get("all_findings", []),
                sources=state.get("sources_gathered", []),
                current_iteration=current_iteration + 1
            )
            
            print(f"   Sufficient: {reflection_result.is_sufficient}")
            print(f"   Confidence: {reflection_result.confidence_score:.2f}")
            if not reflection_result.is_sufficient:
                print(f"   Gaps identified: {len(reflection_result.missing_aspects)}")
            
            return {
                "research_loop_count": current_iteration + 1,
                "reflection_history": [reflection_result],
                "knowledge_coverage_score": reflection_result.confidence_score
            }
            
        except Exception as e:
            print(f"   ❌ Reflection failed: {e}")
            
            # Fallback reflection
            from state_schemas import ReflectionResult
            fallback_reflection = ReflectionResult(
                is_sufficient=True,  # Assume sufficient to avoid infinite loops
                knowledge_gap=f"Reflection failed: {e}",
                follow_up_queries=[],
                confidence_score=0.5,
                missing_aspects=[]
            )
            
            return {
                "research_loop_count": current_iteration + 1,
                "reflection_history": [fallback_reflection],
                "knowledge_coverage_score": 0.5
            }
    
    def _decide_next_step(self, state: OverallResearchState) -> str:
        """Decide whether to continue research or synthesize answer"""
        
        # Get latest reflection
        reflection_history = state.get("reflection_history", [])
        if not reflection_history:
            return "synthesize"
        
        latest_reflection = reflection_history[-1]
        current_iteration = state.get("research_loop_count", 0)
        max_iterations = state.get("max_research_loops", self.config.max_research_loops)
        
        # Continue if not sufficient and within iteration limits
        if not latest_reflection.is_sufficient and current_iteration < max_iterations:
            print(f"   📋 Continuing research (iteration {current_iteration}/{max_iterations})")
            return "continue_research"
        else:
            if current_iteration >= max_iterations:
                print(f"   🏁 Max iterations reached ({max_iterations})")
            else:
                print(f"   ✅ Research deemed sufficient")
            return "synthesize"
    
    def _generate_follow_up_queries(self, state: OverallResearchState) -> Dict[str, Any]:
        """Generate follow-up queries based on reflection"""
        
        print("🎯 Generating follow-up queries...")
        
        try:
            # Get latest reflection
            reflection_history = state.get("reflection_history", [])
            if not reflection_history:
                return {"current_query_batch": []}
            
            latest_reflection = reflection_history[-1]
            
            # Generate follow-up queries
            follow_up_queries = self.query_generator.generate_follow_up_queries(
                original_question=state["research_question"],
                current_findings=state.get("all_findings", []),
                knowledge_gaps=latest_reflection.missing_aspects
            )
            
            print(f"   Generated {len(follow_up_queries)} follow-up queries:")
            for i, query in enumerate(follow_up_queries, 1):
                print(f"   {i}. {query}")
            
            return {
                "current_query_batch": follow_up_queries,
                "search_queries": follow_up_queries  # Add to overall query list
            }
            
        except Exception as e:
            print(f"   ❌ Follow-up generation failed: {e}")
            return {"current_query_batch": []}
    
    def _execute_follow_up_search(self, state: OverallResearchState) -> Dict[str, Any]:
        """Execute follow-up search queries"""
        
        # Reuse the main search function
        return self._execute_web_search(state)
    
    def _synthesize_final_answer(self, state: OverallResearchState) -> Dict[str, Any]:
        """Synthesize final comprehensive answer"""
        
        print("📝 Synthesizing final answer...")
        
        try:
            # Prepare research summary
            research_summary = self._prepare_research_summary(state)
            
            # Generate final answer using LLM
            synthesis_prompt = f"""Based on comprehensive research, provide a detailed answer to this question:

RESEARCH QUESTION: "{state['research_question']}"

RESEARCH FINDINGS:
{research_summary}

Please provide a comprehensive answer that:
1. Directly addresses the research question
2. Synthesizes information from multiple sources
3. Highlights key findings and insights
4. Mentions any limitations or areas needing further research
5. Is well-structured and academic in tone

Format your response as a clear, comprehensive answer suitable for academic or professional use."""

            response = self.llm.invoke(synthesis_prompt)
            final_answer = response.content
            
            print(f"   ✅ Generated final answer ({len(final_answer)} characters)")
            
            # Prepare citation information
            citations = self._prepare_citations(state)
            
            return {
                "messages": [AIMessage(content=final_answer)],
                "research_depth_score": self._calculate_research_depth(state),
                "source_diversity_score": self._calculate_source_diversity(state)
            }
            
        except Exception as e:
            print(f"   ❌ Synthesis failed: {e}")
            
            # Fallback synthesis
            findings = state.get("all_findings", [])
            fallback_answer = f"Research on '{state['research_question']}' found {len(findings)} key findings. However, synthesis failed due to: {e}"
            
            return {
                "messages": [AIMessage(content=fallback_answer)],
                "research_depth_score": 0.5,
                "source_diversity_score": 0.5
            }
    
    def _prepare_research_summary(self, state: OverallResearchState) -> str:
        """Prepare research summary for final synthesis"""
        
        findings = state.get("all_findings", [])
        sources = state.get("sources_gathered", [])
        
        if not findings:
            return "No research findings available."
        
        # Group findings by source type
        academic_findings = []
        technical_findings = []
        other_findings = []
        
        source_map = {source.url: source for source in sources}
        
        for finding in findings[:15]:  # Limit for LLM context
            # Try to determine source type from finding
            if "academic" in finding.lower():
                academic_findings.append(finding)
            elif "technical" in finding.lower() or "github" in finding.lower():
                technical_findings.append(finding)
            else:
                other_findings.append(finding)
        
        # Format summary
        summary_parts = []
        
        if academic_findings:
            summary_parts.append("ACADEMIC SOURCES:")
            for i, finding in enumerate(academic_findings[:5], 1):
                summary_parts.append(f"{i}. {finding[:300]}...")
        
        if technical_findings:
            summary_parts.append("\\nTECHNICAL SOURCES:")
            for i, finding in enumerate(technical_findings[:3], 1):
                summary_parts.append(f"{i}. {finding[:300]}...")
        
        if other_findings:
            summary_parts.append("\\nOTHER SOURCES:")
            for i, finding in enumerate(other_findings[:5], 1):
                summary_parts.append(f"{i}. {finding[:300]}...")
        
        return "\\n".join(summary_parts)
    
    def _prepare_citations(self, state: OverallResearchState) -> List[str]:
        """Prepare citation list from sources"""
        
        sources = state.get("sources_gathered", [])
        citations = []
        
        for source in sources[:10]:  # Limit citations
            if source.title and source.url:
                citation = f"{source.title}. Retrieved from {source.url}"
                if source.publication_date:
                    citation += f" ({source.publication_date[:4]})"
                citations.append(citation)
        
        return citations
    
    def _calculate_research_depth(self, state: OverallResearchState) -> float:
        """Calculate research depth score"""
        
        findings = state.get("all_findings", [])
        iterations = state.get("research_loop_count", 0)
        
        if not findings:
            return 0.0
        
        # Base score from number of findings
        finding_score = min(len(findings) / 20, 1.0)  # Normalize to 20 findings
        
        # Iteration score (more iterations = deeper research)
        iteration_score = min(iterations / 3, 1.0)  # Normalize to 3 iterations
        
        # Average scores
        return (finding_score + iteration_score) / 2
    
    def _calculate_source_diversity(self, state: OverallResearchState) -> float:
        """Calculate source diversity score"""
        
        sources = state.get("sources_gathered", [])
        
        if not sources:
            return 0.0
        
        # Count unique source types
        source_types = set(source.source_type for source in sources)
        max_types = 4  # academic, technical, institutional, web
        
        return len(source_types) / max_types
    
    def _prepare_final_results(self, state: OverallResearchState) -> Dict[str, Any]:
        """Prepare final results dictionary"""
        
        # Extract final answer from messages
        messages = state.get("messages", [])
        final_answer = messages[-1].content if messages else "No answer generated"
        
        # Get reflection history
        reflections = state.get("reflection_history", [])
        
        # Get sources
        sources = state.get("sources_gathered", [])
        
        return {
            "success": True,
            "session_id": state.get("session_id"),
            "research_question": state.get("research_question"),
            "final_answer": final_answer,
            "research_summary": {
                "total_iterations": state.get("research_loop_count", 0),
                "total_queries": len(state.get("search_queries", [])),
                "total_sources": len(sources),
                "total_findings": len(state.get("all_findings", [])),
                "execution_time": state.get("total_execution_time", 0)
            },
            "quality_metrics": {
                "knowledge_coverage": state.get("knowledge_coverage_score", 0),
                "research_depth": state.get("research_depth_score", 0),
                "source_diversity": state.get("source_diversity_score", 0)
            },
            "research_process": {
                "initial_queries": state.get("search_queries", []),
                "reflections": [
                    {
                        "iteration": i + 1,
                        "is_sufficient": r.is_sufficient,
                        "knowledge_gap": r.knowledge_gap,
                        "confidence": r.confidence_score
                    }
                    for i, r in enumerate(reflections)
                ],
                "sources_by_type": self._group_sources_by_type(sources)
            },
            "sources": [
                {
                    "title": source.title,
                    "url": source.url,
                    "type": source.source_type,
                    "relevance": source.relevance_score,
                    "credibility": source.credibility_score
                }
                for source in sources[:20]  # Limit for output size
            ]
        }
    
    def _group_sources_by_type(self, sources) -> Dict[str, int]:
        """Group sources by type for summary"""
        
        type_counts = {}
        for source in sources:
            type_counts[source.source_type] = type_counts.get(source.source_type, 0) + 1
        
        return type_counts


def create_research_workflow(llm, config: Optional[ResearchConfiguration] = None) -> ResearchWorkflowGraph:
    """
    Factory function to create a research workflow graph.
    
    Args:
        llm: LLM instance from Part 1's multi-provider system
        config: Optional research configuration
        
    Returns:
        Configured ResearchWorkflowGraph
    """
    
    return ResearchWorkflowGraph(llm, config)


def demonstrate_research_workflow():
    """Demonstrate the research workflow capabilities"""
    
    print("🔬 RESEARCH WORKFLOW DEMONSTRATION")
    print("=" * 34)
    
    print("\\n🔄 Workflow Steps:")
    workflow_steps = [
        "1. Generate Initial Queries - Transform question into search strategies",
        "2. Execute Web Search - Parallel search across multiple queries",
        "3. Reflect on Completeness - Identify knowledge gaps using LLM analysis",
        "4. Generate Follow-ups - Create targeted queries for missing information",
        "5. Execute Follow-up Search - Search for gap-filling information", 
        "6. Synthesize Answer - Combine all findings into comprehensive response"
    ]
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print("\\n🎯 Key Features:")
    features = [
        "Multi-provider LLM support (any provider from Part 1)",
        "Reflection-based iterative improvement",
        "Parallel search execution for efficiency",
        "Comprehensive source validation and scoring",
        "Academic research optimization",
        "Quality metrics and assessment"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    print("\\n📊 Output Quality Metrics:")
    metrics = [
        "Knowledge Coverage Score - How comprehensively the topic is covered",
        "Research Depth Score - How thoroughly each aspect is explored", 
        "Source Diversity Score - Variety of source types and perspectives",
        "Confidence Score - Reliability of the research completeness assessment"
    ]
    
    for metric in metrics:
        print(f"   • {metric}")


if __name__ == "__main__":
    demonstrate_research_workflow()
    
    print("\\n" + "="*60)
    print("🧪 RESEARCH WORKFLOW TESTING")
    print("="*60)
    
    print("\\n💡 To test the complete workflow:")
    print("""
# Import from Part 1 for LLM
import sys
sys.path.append('../Part1_Foundations/modules')
from llm_providers import create_research_llm

from research_graph import create_research_workflow
from state_schemas import ResearchConfiguration

# Create LLM and workflow
llm = create_research_llm()  # Auto-selects available provider
config = ResearchConfiguration.for_academic_research()
workflow = create_research_workflow(llm, config)

# Conduct research
results = workflow.conduct_research(
    "What are recent advances in quantum error correction?"
)

# Display results
if results["success"]:
    print(f"Research Question: {results['research_question']}")
    print(f"\\nFinal Answer:\\n{results['final_answer']}")
    print(f"\\nSummary: {results['research_summary']}")
    print(f"Quality Metrics: {results['quality_metrics']}")
else:
    print(f"Research failed: {results['error']}")
""")