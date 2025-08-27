"""
Query Generator for Research Workflows - Function-Based Node

LangGraph node function for generating search queries from research questions.
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

from state_schemas import OverallState, QueryGenerationState
from tools_and_schemas import SearchQueryList


def generate_query(state: OverallState, config: RunnableConfig = None) -> QueryGenerationState:
    """
    LangGraph node that generates search queries based on the user's question.
    
    Adapted from Google's Gemini reference to work with multi-provider LLM system.
    Uses structured output for reliable query generation.
    
    Args:
        state: Current graph state containing the user's question
        config: Configuration for the runnable (optional for tutorial)
        
    Returns:
        Dictionary with state update, including search_query key containing generated queries
    """
    from configuration import Configuration
    
    # Get configuration (with defaults for tutorial)
    if config:
        try:
            configurable = Configuration.from_runnable_config(config)
        except:
            configurable = Configuration()  # Use defaults
    else:
        configurable = Configuration()  # Use defaults for tutorial
    
    # Check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries
    
    # Create LLM using Part 1's multi-provider system
    try:
        llm = create_research_llm(
            provider_name="auto",  # Auto-select best available
            temperature=1.0,
            fast_mode=False
        )
    except Exception as e:
        print(f"⚠️ Failed to create LLM: {e}")
        # Fallback to template-based queries
        return _generate_template_queries(state)
    
    # Use structured output for reliable JSON parsing
    structured_llm = llm.with_structured_output(SearchQueryList)
    
    # Format the prompt (adapted from reference)
    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
        number_queries=state["initial_search_query_count"]
    )
    
    try:
        # Generate the search queries
        result = structured_llm.invoke(formatted_prompt)
        
        # Convert to Query format for compatibility
        queries = [{"query": q, "rationale": result.rationale} for q in result.query]
        
        return {"search_query": queries}
        
    except Exception as e:
        print(f"⚠️ LLM query generation failed: {e}")
        # Fallback to template-based queries
        return _generate_template_queries(state)


def generate_follow_up_queries(
    original_question: str,
    current_findings: List[str], 
    knowledge_gaps: List[str]
) -> List[str]:
    """
    Generate follow-up queries to address identified knowledge gaps.
    
    Args:
        original_question: Original research question
        current_findings: Current research findings
        knowledge_gaps: Identified gaps in knowledge
        
    Returns:
        List of follow-up search queries
    """
    try:
        # Create LLM for follow-up generation
        llm = create_research_llm(temperature=0.7, fast_mode=True)
        
        # Create follow-up prompt
        follow_up_prompt = f"""Based on this research question: "{original_question}"

Current findings include:
{_format_findings(current_findings[:5])}

The following knowledge gaps have been identified:
{_format_gaps(knowledge_gaps)}

Generate 2-4 specific search queries that would help fill these knowledge gaps.
Focus on the missing aspects and provide queries that would find complementary information.

Format your response as a simple list of queries, one per line."""

        response = llm.invoke(follow_up_prompt)
        follow_up_queries = _parse_llm_queries(response.content)
        
        # Add template-based follow-ups for gaps
        template_follow_ups = _generate_gap_specific_queries(knowledge_gaps)
        
        # Combine and optimize
        all_follow_ups = follow_up_queries + template_follow_ups
        optimized_follow_ups = _optimize_query_list(all_follow_ups)
        
        return optimized_follow_ups[:4]  # Limit to 4 follow-ups
        
    except Exception as e:
        print(f"⚠️ Follow-up query generation failed: {e}")
        return _generate_gap_specific_queries(knowledge_gaps)


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
query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


def _generate_template_queries(state: OverallState) -> QueryGenerationState:
    """Fallback template-based query generation"""
    
    research_topic = get_research_topic(state["messages"])
    current_year = str(datetime.now().year)
    
    # Basic template queries
    template_queries = [
        f"{research_topic} recent advances {current_year}",
        f"{research_topic} applications and methods",
        f"{research_topic} research overview"
    ]
    
    # Convert to Query format
    queries = [
        {"query": q, "rationale": "Template-based fallback query"} 
        for q in template_queries[:state.get("initial_search_query_count", 3)]
    ]
    
    return {"search_query": queries}


def _format_findings(findings: List[str]) -> str:
    """Format findings for LLM prompts"""
    if not findings:
        return "No specific findings available yet."
    
    formatted = []
    for i, finding in enumerate(findings[:5], 1):
        formatted.append(f"{i}. {finding[:100]}...")
    return "\n".join(formatted)


def _format_gaps(gaps: List[str]) -> str:
    """Format knowledge gaps for LLM prompts"""
    if not gaps:
        return "No specific gaps identified."
    
    formatted = []
    for i, gap in enumerate(gaps[:3], 1):
        formatted.append(f"{i}. {gap}")
    return "\n".join(formatted)


def _parse_llm_queries(llm_response: str) -> List[str]:
    """Parse LLM response into query list"""
    
    queries = []
    lines = llm_response.strip().split('\n')
    
    for line in lines:
        # Remove common prefixes and clean up
        clean_line = re.sub(r'^[\d\.\-\*\+]\s*', '', line.strip())
        clean_line = re.sub(r'^["""]', '', clean_line)
        clean_line = re.sub(r'["""]$', '', clean_line)
        
        if clean_line and len(clean_line) > 5:
            queries.append(clean_line)
    
    return queries


def _optimize_query_list(queries: List[str]) -> List[str]:
    """Optimize query list by removing duplicates"""
    
    if not queries:
        return []
    
    unique_queries = []
    for query in queries:
        query_clean = query.strip()
        if query_clean and len(query_clean) > 5 and query_clean not in unique_queries:
            unique_queries.append(query_clean)
    
    return unique_queries[:6]  # Max 6 queries


def _generate_gap_specific_queries(knowledge_gaps: List[str]) -> List[str]:
    """Generate queries specifically targeting knowledge gaps"""
    
    gap_queries = []
    for gap in knowledge_gaps[:3]:
        # Simple keyword extraction from gap
        words = gap.lower().split()
        key_terms = [w for w in words if len(w) > 3 and w.isalpha()]
        if key_terms:
            gap_queries.append(" ".join(key_terms[:3]))
    
    return gap_queries


# Legacy compatibility class for existing tutorial code
class ResearchQueryGenerator:
    """Legacy compatibility wrapper for the class-based interface"""
    
    def __init__(self, llm, config=None):
        self.llm = llm
        self.config = config
    
    def generate_initial_queries(self, research_question: str):
        """Legacy compatibility method"""
        from langchain_core.messages import HumanMessage
        from state_schemas import OverallState
        
        # Create state for function call
        state = OverallState(
            messages=[HumanMessage(content=research_question)],
            search_query=[],
            web_research_result=[],
            sources_gathered=[],
            initial_search_query_count=3,
            max_research_loops=2,
            research_loop_count=0,
            reasoning_model="default"
        )
        
        # Call function-based node
        result = generate_query(state)
        
        # Convert back to legacy format
        class LegacyResult:
            def __init__(self, queries_data):
                self.queries = [q["query"] for q in queries_data]
                self.rationale = queries_data[0]["rationale"] if queries_data else "Generated queries"
                self.research_strategy = "Multi-provider research approach"
        
        return LegacyResult(result["search_query"])
    
    def generate_follow_up_queries(self, original_question: str, current_findings: List[str], knowledge_gaps: List[str]):
        """Legacy compatibility method"""
        return generate_follow_up_queries(original_question, current_findings, knowledge_gaps)


if __name__ == "__main__":
    print("🔍 FUNCTION-BASED QUERY GENERATOR")
    print("=" * 35)
    
    print("\n✅ Key Features:")
    features = [
        "Function-based LangGraph node architecture",
        "Multi-provider LLM support via Part 1 integration", 
        "Structured output with Pydantic models",
        "Fallback template-based query generation",
        "Legacy compatibility wrapper for existing code"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    print("\n🔄 Usage in LangGraph:")
    print("""
    # Add as node to LangGraph workflow
    builder.add_node("generate_query", generate_query)
    
    # Or use legacy wrapper for existing code
    llm = create_research_llm()
    generator = ResearchQueryGenerator(llm)
    result = generator.generate_initial_queries("Your research question")
    """)
    
    print("\n✅ Ready for LangGraph integration!")