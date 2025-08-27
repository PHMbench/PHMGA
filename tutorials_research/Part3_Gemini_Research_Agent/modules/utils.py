"""
Utils - Following Google Reference Architecture

Basic utility functions matching the reference implementation
from Google's Gemini LangGraph quickstart.
"""

from typing import List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    Matches Google reference implementation exactly.
    """
    # Check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


# Simplified citation utilities (optional for educational purposes)
def create_mock_citations(sources: List[dict], text: str) -> str:
    """
    Create simple citations for educational purposes.
    In production, this would handle real citation formatting.
    """
    if not sources:
        return text
    
    # Add simple source references
    citations = []
    for i, source in enumerate(sources[:5], 1):
        title = source.get("title", f"Source {i}")[:50]
        url = source.get("short_url", "#")
        citations.append(f"[{i}. {title}...]({url})")
    
    if citations:
        text += "\n\n**Sources:**\n" + "\n".join(citations)
    
    return text


def format_research_summary(sources: List[dict], research_results: List[str]) -> str:
    """
    Format a basic research summary for educational purposes.
    """
    summary = "## Research Summary\n\n"
    
    if research_results:
        summary += "### Key Findings:\n"
        for i, result in enumerate(research_results, 1):
            summary += f"\n**Finding {i}:**\n{result}\n"
    
    if sources:
        summary += f"\n### Sources ({len(sources)} total):\n"
        for i, source in enumerate(sources[:5], 1):
            title = source.get("title", "Unknown Title")
            summary += f"{i}. {title}\n"
        
        if len(sources) > 5:
            summary += f"... and {len(sources) - 5} more sources\n"
    
    return summary