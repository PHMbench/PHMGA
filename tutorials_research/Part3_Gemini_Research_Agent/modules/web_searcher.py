"""
Multi-Provider Web Search for Research Workflows

Intelligent search strategy that uses Google's native search API when available,
with fallback to simulated search for multi-provider compatibility.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

# Add path for Part 1 LLM providers
sys.path.append('../Part1_Foundations/modules')
from llm_providers import create_research_llm, LLMProvider

from state_schemas import OverallState, WebSearchState
from tools_and_schemas import ResearchSource


def web_research(state: WebSearchState, config: RunnableConfig = None) -> OverallState:
    """
    LangGraph node that performs web research using intelligent search strategy.
    
    Uses Google's native search API when available (like reference),
    with fallbacks for multi-provider compatibility.
    
    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update including sources_gathered and web_research_result
    """
    # Determine search strategy based on available resources
    search_strategy = _determine_search_strategy()
    
    if search_strategy == "google_native":
        return _web_research_google_native(state, config)
    elif search_strategy == "simulated":
        return _web_research_simulated(state, config)
    else:
        return _web_research_fallback(state, config)


def _determine_search_strategy() -> str:
    """Determine which search strategy to use based on available resources"""
    
    # Check for Google Search API availability
    google_api_key = os.getenv('GEMINI_API_KEY')
    google_search_key = os.getenv('GOOGLE_API_KEY') 
    google_search_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    if google_api_key and google_search_key and google_search_id:
        return "google_native"
    else:
        return "simulated"  # Tutorial-friendly fallback


def _web_research_google_native(state: WebSearchState, config: RunnableConfig = None) -> OverallState:
    """Use Google's native search API (reference implementation pattern)"""
    try:
        from google.genai import Client
        from configuration import Configuration
        
        # Get configuration
        configurable = Configuration.from_runnable_config(config) if config else None
        
        # Format prompt for Google Search
        formatted_prompt = web_searcher_instructions.format(
            current_date=datetime.now().strftime("%B %d, %Y"),
            research_topic=state["search_query"]
        )
        
        # Use Google genai client for native search
        genai_client = Client(api_key=os.getenv('GEMINI_API_KEY'))
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp",
            contents=formatted_prompt,
            config={
                "tools": [{"google_search": {}}],
                "temperature": 0,
            }
        )
        
        # Process grounding metadata (reference pattern)
        sources_gathered = _process_google_grounding_metadata(response, state["id"])
        
        return {
            "sources_gathered": sources_gathered,
            "search_query": [state["search_query"]], 
            "web_research_result": [response.text]
        }
        
    except Exception as e:
        print(f"⚠️ Google native search failed: {e}")
        return _web_research_simulated(state, config)


def _web_research_simulated(state: WebSearchState, config: RunnableConfig = None) -> OverallState:
    """Simulated search for tutorial and multi-provider environments"""
    
    print(f"🔍 Simulating web research for: {state['search_query']}")
    
    # Create realistic simulated sources
    simulated_sources = _create_simulated_sources(state["search_query"])
    
    # Generate LLM-based research summary if possible
    try:
        llm = create_research_llm(temperature=0.3, fast_mode=True)
        
        summary_prompt = f"""Conduct research on "{state['search_query']}" and provide a comprehensive summary.

Include information about:
- Recent developments and findings
- Key applications and use cases  
- Current challenges and limitations
- Future directions and opportunities

Format your response as a well-structured research summary with specific details."""

        response = llm.invoke(summary_prompt)
        research_summary = response.content
        
    except Exception as e:
        print(f"⚠️ LLM summary generation failed: {e}")
        research_summary = f"Research findings on {state['search_query']} indicate active development in this area with multiple approaches being explored."
    
    return {
        "sources_gathered": simulated_sources,
        "search_query": [state["search_query"]],
        "web_research_result": [research_summary]
    }


def _web_research_fallback(state: WebSearchState, config: RunnableConfig = None) -> OverallState:
    """Basic fallback search strategy"""
    
    sources = [
        {
            "url": f"https://example.com/research/{state['search_query'].replace(' ', '-')}",
            "title": f"Research on {state['search_query']}",
            "snippet": f"Comprehensive overview of {state['search_query']} with recent findings and applications.",
            "short_url": f"[{state['id']}]",
            "value": f"Research Source {state['id']}"
        }
    ]
    
    summary = f"Fallback research summary for: {state['search_query']}"
    
    return {
        "sources_gathered": sources,
        "search_query": [state["search_query"]],
        "web_research_result": [summary]
    }


def _create_simulated_sources(query: str) -> List[Dict[str, Any]]:
    """Create realistic simulated sources for tutorial purposes"""
    
    query_terms = query.lower().split()
    sources = []
    
    # Academic source simulation
    if any(term in query_terms for term in ['research', 'study', 'analysis', 'advances']):
        sources.append({
            "url": f"https://arxiv.org/abs/{datetime.now().year}.{len(query):04d}",
            "title": f"Recent Advances in {query.title()}",
            "snippet": f"This paper presents a comprehensive study of {query.lower()}, discussing recent developments and future directions in the field.",
            "short_url": f"[arxiv-{len(sources)+1}]",
            "value": f"arXiv Research Paper on {query}"
        })
    
    # Technical source simulation
    if any(term in query_terms for term in ['implementation', 'methods', 'algorithms', 'techniques']):
        sources.append({
            "url": f"https://github.com/research/{query.replace(' ', '-')}",
            "title": f"{query.title()}: Implementation Guide",
            "snippet": f"Open source implementation of {query.lower()} with practical examples, benchmarks and performance analysis.",
            "short_url": f"[github-{len(sources)+1}]", 
            "value": f"GitHub Implementation of {query}"
        })
    
    # News/Industry source
    sources.append({
        "url": f"https://techreport.com/{query.replace(' ', '-')}-2024",
        "title": f"Industry Report: {query.title()} in 2024",
        "snippet": f"Latest industry trends and commercial applications of {query.lower()}, including market analysis and future outlook.",
        "short_url": f"[report-{len(sources)+1}]",
        "value": f"Industry Report on {query}"
    })
    
    # Academic journal
    sources.append({
        "url": f"https://journal.nature.com/{query.replace(' ', '-')}",
        "title": f"Journal Article: Breakthrough in {query.title()}",
        "snippet": f"Peer-reviewed research on {query.lower()} published in leading scientific journal, with experimental validation.",
        "short_url": f"[nature-{len(sources)+1}]",
        "value": f"Nature Article on {query}"
    })
    
    return sources[:3]  # Return top 3 simulated sources


def _process_google_grounding_metadata(response, search_id: str) -> List[Dict[str, Any]]:
    """Process Google's grounding metadata (reference pattern)"""
    
    sources = []
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0] 
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                grounding_chunks = candidate.grounding_metadata.grounding_chunks
                
                for i, chunk in enumerate(grounding_chunks[:5]):  # Limit to 5 sources
                    source = {
                        "url": chunk.web.uri if hasattr(chunk, 'web') else f"https://example.com/source-{i}",
                        "title": chunk.web.title if hasattr(chunk, 'web') and hasattr(chunk.web, 'title') else f"Source {i+1}",
                        "snippet": f"Content from grounding chunk {i+1}",
                        "short_url": f"[gs-{search_id}-{i}]",
                        "value": f"Google Search Result {i+1}"
                    }
                    sources.append(source)
    except Exception as e:
        print(f"⚠️ Error processing grounding metadata: {e}")
    
    return sources if sources else _create_simulated_sources("search results")


# Prompt template from reference
web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""


# Legacy compatibility class (WebSearchExecutor)
class WebSearchExecutor:
    """
    Legacy compatibility class for existing tutorial code.
    Wraps the function-based web research node.
    """
    
    def __init__(self, config=None):
        self.config = config
    
    def execute_parallel_search(self, queries: List[str]) -> Dict[str, List]:
        """Execute multiple search queries"""
        results = {}
        
        for i, query in enumerate(queries):
            # Create state for function call
            search_state = WebSearchState(search_query=query, id=str(i))
            
            # Call function-based node
            result = web_research(search_state)
            
            # Convert to legacy format
            legacy_sources = []
            for source in result.get("sources_gathered", []):
                legacy_source = {
                    "title": source.get("title", "No title"),
                    "url": source.get("url", ""),
                    "snippet": source.get("snippet", "No description")
                }
                legacy_sources.append(legacy_source)
            
            results[query] = legacy_sources
        
        return results
    
    def search_single_query(self, query: str) -> List:
        """Execute a single search query"""
        search_state = WebSearchState(search_query=query, id="0")
        result = web_research(search_state)
        
        # Convert to legacy format
        legacy_sources = []
        for source in result.get("sources_gathered", []):
            legacy_source = {
                "title": source.get("title", "No title"), 
                "url": source.get("url", ""),
                "snippet": source.get("snippet", "No description")
            }
            legacy_sources.append(legacy_source)
        
        return legacy_sources
    
    def execute_search(self, query: str, max_results: int = 5) -> List:
        """Execute search with result limit"""
        results = self.search_single_query(query)
        return results[:max_results]
    
    def _create_demo_results(self, query: str) -> List:
        """Create demo results for tutorial"""
        return self.execute_search(query, max_results=3)


if __name__ == "__main__":
    print("🌐 MULTI-PROVIDER WEB SEARCH")
    print("=" * 32)
    
    print("\n🔍 Search Strategy Selection:")
    strategies = [
        ("Google Native", "Uses Google's search API with grounding metadata (reference pattern)"),
        ("Simulated", "LLM-generated research summaries with realistic source simulation"),
        ("Fallback", "Basic template-based search results")
    ]
    
    for strategy, description in strategies:
        print(f"   • {strategy}: {description}")
    
    print(f"\n📊 Current Strategy: {_determine_search_strategy().upper()}")
    
    print("\n✅ Key Features:")
    features = [
        "Intelligent strategy selection based on available APIs",
        "Google native search integration (reference pattern)",
        "Multi-provider LLM compatibility for research summaries", 
        "Realistic source simulation for tutorial environments",
        "Legacy compatibility with existing tutorial code"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    print("\n🔄 Usage:")
    print("""
    # Function-based (LangGraph)
    result = web_research(WebSearchState(search_query="AI research", id="1"))
    
    # Legacy compatibility
    searcher = WebSearchExecutor()
    results = searcher.execute_search("machine learning trends")
    """)
    
    print("\n✅ Ready for intelligent multi-provider search!")