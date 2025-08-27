"""
Reflection Agent for Research Workflows - Function-Based Node

LangGraph node function for analyzing research completeness and identifying knowledge gaps.
Adapted from Google's reference architecture for multi-provider LLM support.
"""

import sys
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.runnables import RunnableConfig

# Add path for Part 1 LLM providers  
sys.path.append('../Part1_Foundations/modules')
from llm_providers import create_research_llm, LLMProvider

from state_schemas import OverallState, ReflectionState
from tools_and_schemas import Reflection


def reflection(state: OverallState, config: RunnableConfig = None) -> ReflectionState:
    """
    LangGraph node that identifies knowledge gaps and generates follow-up queries.
    
    Analyzes the current research summary to identify areas for further research and
    generates potential follow-up queries. Uses structured output for reliable parsing.
    
    Args:
        state: Current graph state containing the research results and topic
        config: Configuration for the runnable (optional for tutorial)
        
    Returns:
        Dictionary with state update including reflection analysis
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
    
    # Increment research loop count and get reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", "default")
    
    # Create LLM using Part 1's multi-provider system
    try:
        llm = create_research_llm(
            provider_name="auto",  # Auto-select best available
            temperature=1.0,
            fast_mode=False
        )
    except Exception as e:
        print(f"⚠️ Failed to create LLM for reflection: {e}")
        return _create_fallback_reflection(state)
    
    # Format the prompt (adapted from reference)
    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))
    
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
        summaries=summaries
    )
    
    try:
        # Use structured output for reliable JSON parsing
        result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
        
        return {
            "is_sufficient": result.is_sufficient,
            "knowledge_gap": result.knowledge_gap,
            "follow_up_queries": result.follow_up_queries,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state.get("search_query", []))
        }
        
    except Exception as e:
        print(f"⚠️ Reflection analysis failed: {e}")
        return _create_fallback_reflection(state)


def analyze_research_completeness_extended(
    original_question: str,
    research_findings: List[str],
    sources: List[Dict[str, Any]],
    current_iteration: int
) -> Dict[str, Any]:
    """
    Extended research completeness analysis function.
    
    Provides more detailed analysis than the basic reflection node,
    including coverage metrics and quality scoring.
    
    Args:
        original_question: Original research question
        research_findings: Current research findings
        sources: List of research sources
        current_iteration: Current research iteration number
        
    Returns:
        Dictionary with detailed reflection analysis
    """
    print(f"🤔 Analyzing research completeness (iteration {current_iteration})...")
    
    try:
        # Create LLM for analysis
        llm = create_research_llm(temperature=0.3, fast_mode=False)
        
        # Analyze coverage quality
        coverage_analysis = _analyze_coverage_quality(original_question, research_findings, sources)
        
        # Generate LLM-based reflection
        reflection_result = _generate_detailed_reflection(
            llm, original_question, research_findings, coverage_analysis
        )
        
        # Add systematic gap analysis
        systematic_gaps = _identify_systematic_gaps(original_question, research_findings, sources)
        
        # Calculate confidence score
        confidence_score = _calculate_confidence_score(research_findings, sources, coverage_analysis)
        
        # Combine results
        final_result = {
            "is_sufficient": reflection_result.get("is_sufficient", False),
            "knowledge_gap": reflection_result.get("knowledge_gap", "Additional analysis needed"),
            "follow_up_queries": reflection_result.get("follow_up_queries", []),
            "missing_aspects": systematic_gaps,
            "confidence_score": confidence_score,
            "coverage_analysis": coverage_analysis
        }
        
        print(f"   📊 Coverage score: {coverage_analysis.get('overall_score', 0):.1%}")
        print(f"   🎯 Confidence: {confidence_score:.2f}")
        print(f"   🔍 Gaps identified: {len(systematic_gaps)}")
        
        return final_result
        
    except Exception as e:
        print(f"⚠️ Extended reflection analysis failed: {e}")
        return {
            "is_sufficient": False,
            "knowledge_gap": f"Analysis error: {str(e)}",
            "follow_up_queries": [f"{original_question} comprehensive research"],
            "missing_aspects": ["detailed analysis required"],
            "confidence_score": 0.0,
            "coverage_analysis": {"overall_score": 0.0}
        }


# Helper functions (adapted from reference and original)
def get_current_date():
    """Get current date in readable format"""
    return datetime.now().strftime("%B %d, %Y")


def get_research_topic(messages) -> str:
    """Extract research topic from messages"""
    if messages and len(messages) > 0:
        # Handle both dict and Message object formats
        if hasattr(messages[0], 'content'):
            return messages[0].content
        elif isinstance(messages[0], dict):
            return messages[0].get('content', 'Unknown topic')
    return "Unknown research topic"


# Prompt template adapted from reference
reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}"""


def _create_fallback_reflection(state: OverallState) -> ReflectionState:
    """Create fallback reflection when LLM analysis fails"""
    
    research_topic = get_research_topic(state["messages"])
    
    # Simple heuristic: if we have few results, more research is needed
    web_results = state.get("web_research_result", [])
    is_sufficient = len(web_results) >= 3  # Basic threshold
    
    if is_sufficient:
        return {
            "is_sufficient": True,
            "knowledge_gap": "",
            "follow_up_queries": [],
            "research_loop_count": state.get("research_loop_count", 1),
            "number_of_ran_queries": len(state.get("search_query", []))
        }
    else:
        return {
            "is_sufficient": False,
            "knowledge_gap": "Insufficient research coverage. Need more comprehensive analysis.",
            "follow_up_queries": [f"{research_topic} detailed analysis", f"{research_topic} recent developments"],
            "research_loop_count": state.get("research_loop_count", 1),
            "number_of_ran_queries": len(state.get("search_query", []))
        }


def _analyze_coverage_quality(question: str, findings: List[str], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze coverage quality across different dimensions"""
    
    return {
        "total_findings": len(findings),
        "total_sources": len(sources),
        "source_diversity": _calculate_source_diversity(sources),
        "temporal_coverage": _assess_temporal_coverage(sources),
        "topic_breadth": _assess_topic_breadth(question, findings),
        "overall_score": min(1.0, (len(sources) / 5 + len(findings) / 3) / 2)  # Simple heuristic
    }


def _calculate_source_diversity(sources: List[Dict[str, Any]]) -> float:
    """Calculate diversity of source types"""
    if not sources:
        return 0.0
    
    # Extract source types from URLs (simple heuristic)
    source_types = set()
    for source in sources:
        url = source.get("url", "")
        if "arxiv.org" in url or "scholar.google" in url:
            source_types.add("academic")
        elif "github.com" in url:
            source_types.add("technical")
        elif any(domain in url for domain in [".edu", ".gov", ".org"]):
            source_types.add("institutional")
        else:
            source_types.add("web")
    
    return len(source_types) / 4  # Normalize to 4 types


def _assess_temporal_coverage(sources: List[Dict[str, Any]]) -> float:
    """Assess temporal coverage of sources"""
    if not sources:
        return 0.0
    
    current_year = datetime.now().year
    recent_count = 0
    
    for source in sources:
        url = source.get("url", "")
        title = source.get("title", "")
        # Simple heuristic: look for year indicators
        if str(current_year) in url or str(current_year) in title:
            recent_count += 1
        elif str(current_year - 1) in url or str(current_year - 1) in title:
            recent_count += 0.5
    
    return min(1.0, recent_count / len(sources))


def _assess_topic_breadth(question: str, findings: List[str]) -> float:
    """Assess breadth of topic coverage"""
    if not findings:
        return 0.0
    
    # Extract key terms from question
    question_terms = set(re.findall(r'\b\w{4,}\b', question.lower()))
    
    # Count unique concepts in findings  
    findings_text = " ".join(findings).lower()
    findings_terms = set(re.findall(r'\b\w{4,}\b', findings_text))
    
    if not question_terms:
        return 0.5
    
    overlap = len(question_terms.intersection(findings_terms))
    coverage = overlap / len(question_terms)
    
    return min(1.0, coverage)


def _generate_detailed_reflection(llm, question: str, findings: List[str], coverage: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed reflection using LLM"""
    
    findings_summary = _prepare_findings_summary(findings)
    coverage_summary = _prepare_coverage_summary(coverage)
    
    detailed_prompt = f"""Analyze the research completeness for: "{question}"

RESEARCH FINDINGS:
{findings_summary}

COVERAGE ANALYSIS:
{coverage_summary}

Assess whether this research comprehensively answers the question. Consider:
1. Completeness: Are all major aspects covered?
2. Depth: Is there sufficient detail and evidence?
3. Recency: Are recent developments included?
4. Perspectives: Are different viewpoints represented?

Respond with:
SUFFICIENT: [YES/NO]
KNOWLEDGE_GAP: [What specific information is missing]
FOLLOW_UP_QUERIES: [2-3 specific queries to fill gaps, one per line]"""

    try:
        response = llm.invoke(detailed_prompt)
        return _parse_detailed_reflection(response.content)
    except Exception as e:
        print(f"⚠️ Detailed reflection failed: {e}")
        return {"is_sufficient": False, "knowledge_gap": "Analysis incomplete", "follow_up_queries": []}


def _identify_systematic_gaps(question: str, findings: List[str], sources: List[Dict[str, Any]]) -> List[str]:
    """Identify systematic gaps using predefined categories"""
    
    gaps = []
    question_lower = question.lower()
    findings_text = " ".join(findings).lower()
    
    # Check for methodological gaps
    if not any(term in findings_text for term in ["method", "approach", "technique", "algorithm"]):
        if any(word in question_lower for word in ["how", "method", "approach"]):
            gaps.append("implementation methodology")
    
    # Check for empirical gaps
    if not any(term in findings_text for term in ["result", "performance", "evaluation", "experiment"]):
        if any(word in question_lower for word in ["performance", "results", "effectiveness"]):
            gaps.append("performance evaluation")
    
    # Check for contextual gaps
    if not any(term in findings_text for term in ["application", "use case", "real-world", "practical"]):
        if any(word in question_lower for word in ["applications", "use", "practical"]):
            gaps.append("practical applications")
    
    # Check for temporal gaps (basic heuristic)
    current_year = str(datetime.now().year)
    if current_year not in findings_text and "recent" in question_lower:
        gaps.append("recent developments")
    
    return gaps[:4]  # Limit to 4 gaps


def _calculate_confidence_score(findings: List[str], sources: List[Dict[str, Any]], coverage: Dict[str, Any]) -> float:
    """Calculate confidence score in the reflection assessment"""
    
    factors = [
        min(len(sources) / 5, 1.0),  # Source count factor
        coverage.get("source_diversity", 0.0),
        coverage.get("overall_score", 0.0),
        min(len(findings) / 3, 1.0)  # Findings count factor
    ]
    
    return sum(factors) / len(factors)


def _parse_detailed_reflection(response: str) -> Dict[str, Any]:
    """Parse detailed reflection response"""
    
    result = {"is_sufficient": False, "knowledge_gap": "", "follow_up_queries": []}
    
    # Extract sufficiency
    if re.search(r'SUFFICIENT:\s*YES', response, re.IGNORECASE):
        result["is_sufficient"] = True
    
    # Extract knowledge gap
    gap_match = re.search(r'KNOWLEDGE_GAP:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if gap_match:
        result["knowledge_gap"] = gap_match.group(1).strip()
    
    # Extract follow-up queries
    queries_section = re.search(r'FOLLOW_UP_QUERIES:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
    if queries_section:
        queries_text = queries_section.group(1).strip()
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
        result["follow_up_queries"] = queries[:3]
    
    return result


def _prepare_findings_summary(findings: List[str]) -> str:
    """Prepare findings summary for analysis"""
    if not findings:
        return "No findings available."
    
    summary = []
    for i, finding in enumerate(findings[:5], 1):
        truncated = finding[:200] + "..." if len(finding) > 200 else finding
        summary.append(f"{i}. {truncated}")
    
    return "\n".join(summary)


def _prepare_coverage_summary(coverage: Dict[str, Any]) -> str:
    """Prepare coverage analysis summary"""
    return f"""Sources: {coverage.get('total_sources', 0)}
Findings: {coverage.get('total_findings', 0)}
Source Diversity: {coverage.get('source_diversity', 0):.1%}
Overall Score: {coverage.get('overall_score', 0):.1%}"""


# Legacy compatibility class for existing tutorial code
class ResearchReflectionAgent:
    """Legacy compatibility wrapper for the class-based interface"""
    
    def __init__(self, llm, config=None):
        self.llm = llm
        self.config = config
    
    def analyze_research_completeness(self, original_question: str, current_findings: List[str], current_sources: List, research_iteration: int):
        """Legacy compatibility method"""
        
        # Convert sources to expected format
        sources_dict = []
        for source in current_sources:
            if hasattr(source, 'url'):  # Pydantic model
                sources_dict.append({
                    "url": source.url,
                    "title": source.title,
                    "snippet": source.snippet
                })
            elif isinstance(source, dict):
                sources_dict.append(source)
        
        # Call extended analysis function
        result = analyze_research_completeness_extended(
            original_question, current_findings, sources_dict, research_iteration
        )
        
        # Convert back to legacy format
        class LegacyReflectionResult:
            def __init__(self, data):
                self.is_sufficient = data["is_sufficient"]
                self.knowledge_gap = data["knowledge_gap"]
                self.follow_up_queries = data["follow_up_queries"]
                self.confidence_score = data["confidence_score"]
                self.missing_aspects = data["missing_aspects"]
        
        return LegacyReflectionResult(result)


if __name__ == "__main__":
    print("🤔 FUNCTION-BASED REFLECTION AGENT")
    print("=" * 35)
    
    print("\n✅ Key Features:")
    features = [
        "Function-based LangGraph node architecture",
        "Multi-provider LLM support via Part 1 integration",
        "Structured output with Pydantic models",
        "Extended analysis capabilities with quality metrics",
        "Legacy compatibility wrapper for existing code"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    print("\n🔍 Gap Analysis Categories:")
    categories = [
        "Methodological: implementation details, algorithms, techniques",
        "Empirical: performance metrics, experimental evidence", 
        "Contextual: applications, use cases, real-world deployment",
        "Temporal: recent developments, current trends"
    ]
    
    for category in categories:
        print(f"   • {category}")
    
    print("\n🔄 Usage:")
    print("""
    # Function-based (LangGraph)
    result = reflection(state, config)
    
    # Extended analysis
    analysis = analyze_research_completeness_extended(question, findings, sources, 1)
    
    # Legacy compatibility
    agent = ResearchReflectionAgent(llm)
    result = agent.analyze_research_completeness(question, findings, sources, 1)
    """)
    
    print("\n✅ Ready for intelligent reflection analysis!")