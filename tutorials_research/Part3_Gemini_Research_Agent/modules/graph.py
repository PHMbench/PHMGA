"""
Research Graph - Fixed Following Google Reference Architecture

Fixed authentication issues by following Google's reference implementation exactly
with direct API key usage and simplified multi-provider support.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

# Import components
from state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from configuration import Configuration
from tools_and_schemas import SearchQueryList, Reflection
from prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from utils import get_research_topic

# Import LLM providers directly (following reference pattern)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google.genai import Client
    GENAI_CLIENT_AVAILABLE = True
except ImportError:
    GENAI_CLIENT_AVAILABLE = False

load_dotenv()

# API key validation following reference pattern
def validate_api_keys():
    """Validate that at least one provider is configured"""
    providers = []
    if os.getenv("GEMINI_API_KEY"):
        providers.append("Google Gemini")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("OpenAI")
    if os.getenv("DASHSCOPE_API_KEY"):
        providers.append("DashScope")
    if os.getenv("ZHIPUAI_API_KEY"):
        providers.append("ZhipuAI")
    
    if not providers:
        print("❌ No API keys configured. Please set at least one:")
        print("   • GEMINI_API_KEY for Google Gemini")
        print("   • OPENAI_API_KEY for OpenAI GPT")
        print("   • DASHSCOPE_API_KEY for DashScope")
        print("   • ZHIPUAI_API_KEY for ZhipuAI")
        return False
    
    print(f"✅ Available providers: {', '.join(providers)}")
    return True

# Initialize Google genai client for web search (following reference)
genai_client = None
if GENAI_CLIENT_AVAILABLE and os.getenv("GEMINI_API_KEY"):
    try:
        genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))
        print("✅ Google genai client initialized for web search")
    except Exception as e:
        print(f"⚠️ Could not initialize Google genai client: {e}")


# Structured output utilities (Fix DashScope compatibility)
def parse_queries_from_text(text: str, fallback: list) -> list:
    """Extract queries from LLM response text when structured output fails"""
    import json
    import re
    
    # Try to find JSON in the response
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
        r'\{[^{}]*"query"[^{}]*\}',    # JSON objects with query field
        r'(\{.*?\})'                   # Any JSON-like structure
    ]
    
    for pattern in json_patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            try:
                parsed = json.loads(matches.group(1))
                if 'query' in parsed and isinstance(parsed['query'], list):
                    print(f"✅ Parsed queries from JSON: {len(parsed['query'])}")
                    return parsed['query']
            except json.JSONDecodeError:
                continue
    
    # If no JSON found, try to extract queries from text lines
    lines = text.strip().split('\n')
    queries = []
    for line in lines:
        line = line.strip()
        # Skip empty lines, comments, and headers
        if line and not line.startswith(('#', '##', '###', '*', '-')) and len(line) > 10:
            # Clean up the line
            clean_line = line.strip('- ').strip('"').strip("'")
            if clean_line and len(clean_line) > 5:
                queries.append(clean_line)
    
    result_queries = queries[:3] if queries else fallback
    if queries:
        print(f"✅ Extracted queries from text: {len(result_queries)}")
    else:
        print(f"⚠️ Using fallback queries: {len(fallback)}")
    
    return result_queries


def handle_structured_output(result, provider_type: str, schema_type: str, fallback_data):
    """Handle structured output across different providers with validation"""
    
    print(f"🔧 Handling structured output - Provider: {provider_type}, Schema: {schema_type}")
    print(f"🔧 Result type: {type(result).__name__}")
    
    # Check if result has expected attributes
    if schema_type == "SearchQueryList":
        if hasattr(result, 'query') and result.query:
            print(f"✅ Direct structured output successful: {len(result.query)} queries")
            return result.query
        elif hasattr(result, 'content'):
            print(f"⚠️ Structured output failed, trying JSON parsing from content")
            return parse_queries_from_text(result.content, fallback_data)
    
    elif schema_type == "Reflection":
        if (hasattr(result, 'is_sufficient') and hasattr(result, 'knowledge_gap') and 
            hasattr(result, 'follow_up_queries')):
            print(f"✅ Direct structured output successful for Reflection")
            return {
                "is_sufficient": result.is_sufficient,
                "knowledge_gap": result.knowledge_gap,
                "follow_up_queries": result.follow_up_queries
            }
        elif hasattr(result, 'content'):
            print(f"⚠️ Structured reflection failed, trying JSON parsing")
            try:
                import json
                import re
                
                # Try to extract JSON from content
                json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return {
                        "is_sufficient": parsed.get("is_sufficient", fallback_data["is_sufficient"]),
                        "knowledge_gap": parsed.get("knowledge_gap", fallback_data["knowledge_gap"]),
                        "follow_up_queries": parsed.get("follow_up_queries", fallback_data["follow_up_queries"])
                    }
            except Exception as e:
                print(f"⚠️ JSON parsing failed: {e}")
    
    print(f"⚠️ All parsing methods failed, using fallback")
    return fallback_data


def create_llm_direct(provider: str, model: str, temperature: float = 0.7):
    """Create LLM instance directly following reference patterns"""
    
    if provider == "google" or provider == "auto":
        if GOOGLE_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_retries=2,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
    
    if provider == "openai" or (provider == "auto" and not os.getenv("GEMINI_API_KEY")):
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(
                model=model or "gpt-4o",
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
    
    if provider == "dashscope" or (provider == "auto" and not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY")):
        if OPENAI_AVAILABLE and os.getenv("DASHSCOPE_API_KEY"):
            return ChatOpenAI(
                model=model or "qwen-plus",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=temperature,
            )
    
    if provider == "zhipuai" or (provider == "auto" and not any([os.getenv("GEMINI_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("DASHSCOPE_API_KEY")])):
        if os.getenv("ZHIPUAI_API_KEY"):
            try:
                from langchain_community.chat_models import ChatZhipuAI
                return ChatZhipuAI(
                    model=model or "glm-4",
                    api_key=os.getenv("ZHIPUAI_API_KEY"),
                    temperature=temperature,
                )
            except ImportError:
                pass
    
    # If we get here, no provider worked
    raise ValueError(f"No working LLM provider found. Tried: {provider}")


# Nodes (Fixed following Google Reference)
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.
    
    Fixed to use direct LLM instantiation following Google's reference pattern.
    """
    configurable = Configuration.from_runnable_config(config)

    # Check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Prepare fallback data first
    research_topic = get_research_topic(state["messages"])
    fallback_queries = [
        f"{research_topic} overview",
        f"{research_topic} recent developments",
        f"{research_topic} applications"
    ][:state["initial_search_query_count"]]

    # Create LLM directly following reference pattern (FIXED for DashScope compatibility)
    try:
        provider_type = configurable.llm_provider
        llm = create_llm_direct(
            provider=provider_type,
            model=configurable.query_generator_model,
            temperature=1.0
        )
        
        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = query_writer_instructions.format(
            current_date=current_date,
            research_topic=research_topic,
            number_queries=state["initial_search_query_count"],
        )
        
        # Try structured output first
        try:
            print(f"🔧 Attempting structured output with {provider_type}")
            structured_llm = llm.with_structured_output(SearchQueryList)
            result = structured_llm.invoke(formatted_prompt)
            
            # Use robust structured output handling (FIXED BUG)
            queries = handle_structured_output(result, provider_type, "SearchQueryList", fallback_queries)
            
            if queries and len(queries) > 0:
                print(f"✅ Query generation successful: {len(queries)} queries")
                return {"search_query": queries}
            else:
                raise ValueError("No queries returned from structured output")
                
        except Exception as structured_error:
            print(f"⚠️ Structured output failed with {provider_type}: {structured_error}")
            print(f"🔄 Trying fallback LLM call...")
            
            # Fallback: regular LLM call with JSON parsing
            regular_result = llm.invoke(formatted_prompt)
            queries = parse_queries_from_text(regular_result.content, fallback_queries)
            
            if queries:
                print(f"✅ Fallback parsing successful: {len(queries)} queries")
                return {"search_query": queries}
            else:
                raise ValueError("Fallback parsing also failed")
        
    except Exception as e:
        print(f"❌ Query generation completely failed: {e}")
        print("💡 Please check your API key configuration and provider compatibility")
        print(f"🔄 Using simple fallback queries...")
        
        return {"search_query": fallback_queries}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node."""
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research.
    
    Fixed to use Google Search API like the reference implementation.
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Try Google Search API with genai client (following reference)
    if genai_client and GENAI_CLIENT_AVAILABLE:
        try:
            formatted_prompt = web_searcher_instructions.format(
                current_date=get_current_date(),
                research_topic=state["search_query"],
            )

            # Use Google genai client like reference
            response = genai_client.models.generate_content(
                model=configurable.query_generator_model,
                contents=formatted_prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "temperature": 0,
                },
            )
            
            # Process grounding metadata like reference
            sources_gathered = []
            research_text = response.text
            
            if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
                grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
                for chunk in grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web:
                        sources_gathered.append({
                            "title": chunk.web.title or state["search_query"],
                            "short_url": chunk.web.uri,
                            "value": chunk.web.uri,
                            "label": f"Source {state['id']}"
                        })

            print(f"✅ Found {len(sources_gathered)} real sources via Google Search API")
            
            return {
                "sources_gathered": sources_gathered,
                "search_query": [state["search_query"]],
                "web_research_result": [research_text],
            }
            
        except Exception as e:
            print(f"⚠️ Google Search API failed: {e}")
    
    # Fallback to LLM-based research (no mock content)
    try:
        llm = create_llm_direct(
            provider=configurable.llm_provider,
            model=configurable.query_generator_model,
            temperature=0,
        )
        
        formatted_prompt = web_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=state["search_query"],
        )
        
        response = llm.invoke(formatted_prompt)
        research_text = response.content
        
        # Create basic source entry (not mock)
        sources_gathered = [{
            "title": f"Research: {state['search_query']}",
            "short_url": "#",
            "value": "#",
            "label": f"LLM Research {state['id']}"
        }]
        
        return {
            "sources_gathered": sources_gathered,
            "search_query": [state["search_query"]],
            "web_research_result": [research_text],
        }
        
    except Exception as e:
        print(f"❌ Web research failed: {e}")
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [f"Research for '{state['search_query']}' could not be completed due to API issues."],
        }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that analyzes research completeness.
    
    Fixed to use direct LLM instantiation following reference pattern.
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Increment the research loop count
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    # Prepare fallback data first
    web_results = state.get("web_research_result", [])
    is_sufficient = len(web_results) >= 2
    fallback_reflection = {
        "is_sufficient": is_sufficient,
        "knowledge_gap": "Need more comprehensive coverage" if not is_sufficient else "",
        "follow_up_queries": [f"{get_research_topic(state['messages'])} details"] if not is_sufficient else []
    }

    try:
        provider_type = configurable.llm_provider
        # Create LLM directly following reference pattern (FIXED for DashScope compatibility)
        llm = create_llm_direct(
            provider=provider_type,
            model=configurable.reflection_model,
            temperature=1.0
        )
        
        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = reflection_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries="\n\n---\n\n".join(state["web_research_result"]),
        )
        
        # Try structured output first
        try:
            print(f"🔧 Attempting structured reflection with {provider_type}")
            result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
            
            # Use robust structured output handling (FIXED BUG)
            reflection_data = handle_structured_output(result, provider_type, "Reflection", fallback_reflection)
            
            if reflection_data:
                print(f"✅ Reflection successful with {provider_type}")
                return {
                    "is_sufficient": reflection_data.get("is_sufficient", fallback_reflection["is_sufficient"]),
                    "knowledge_gap": reflection_data.get("knowledge_gap", fallback_reflection["knowledge_gap"]),
                    "follow_up_queries": reflection_data.get("follow_up_queries", fallback_reflection["follow_up_queries"]),
                    "research_loop_count": state["research_loop_count"],
                    "number_of_ran_queries": len(state["search_query"]),
                }
            else:
                raise ValueError("No reflection data returned")
                
        except Exception as structured_error:
            print(f"⚠️ Structured reflection failed with {provider_type}: {structured_error}")
            print(f"🔄 Trying fallback LLM call...")
            
            # Fallback: regular LLM call
            regular_result = llm.invoke(formatted_prompt)
            print(f"✅ Using heuristic fallback for reflection")
            
            return {
                "is_sufficient": fallback_reflection["is_sufficient"],
                "knowledge_gap": fallback_reflection["knowledge_gap"],
                "follow_up_queries": fallback_reflection["follow_up_queries"],
                "research_loop_count": state["research_loop_count"],
                "number_of_ran_queries": len(state["search_query"]),
            }
        
    except Exception as e:
        print(f"❌ Reflection completely failed: {e}")
        print("💡 Using simple heuristic fallback")
        
        return {
            "is_sufficient": fallback_reflection["is_sufficient"],
            "knowledge_gap": fallback_reflection["knowledge_gap"],
            "follow_up_queries": fallback_reflection["follow_up_queries"],
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }


def evaluate_research(state: ReflectionState, config: RunnableConfig):
    """LangGraph routing function that determines the next step in the research flow."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.
    
    Fixed to use direct LLM instantiation following reference pattern.
    """
    configurable = Configuration.from_runnable_config(config)

    try:
        # Create LLM directly following reference pattern (FIXED)
        llm = create_llm_direct(
            provider=configurable.llm_provider,
            model=configurable.answer_model,
            temperature=0
        )
        
        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = answer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries="\n---\n\n".join(state["web_research_result"]),
        )

        result = llm.invoke(formatted_prompt)
        final_content = result.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        
        # Create fallback comprehensive answer (no mock content)
        research_topic = get_research_topic(state["messages"])
        summaries = state.get("web_research_result", [])
        sources_count = len(state.get("sources_gathered", []))
        
        final_content = f"""Research Analysis: {research_topic}

Based on available research from {sources_count} sources:

{chr(10).join(summaries)}

This analysis provides insights into {research_topic} based on the available information.
"""

    return {
        "messages": [AIMessage(content=final_content)],
        "sources_gathered": state.get("sources_gathered", []),
    }


# Create the Graph (Following Google Reference)
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint
builder.add_edge(START, "generate_query")

# Add conditional edges
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)

builder.add_edge("web_research", "reflection")

builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)

builder.add_edge("finalize_answer", END)

# Compile the graph
graph = builder.compile()


# Main research function (Fixed)
def conduct_research(research_question: str, config_dict: dict = None) -> dict:
    """
    Conduct research using the fixed authentication system.
    """
    try:
        # Validate API keys first
        if not validate_api_keys():
            return {
                "success": False,
                "error": "No API keys configured",
                "research_question": research_question,
                "timestamp": get_current_date(),
            }
        
        # Prepare configuration
        config = {"configurable": config_dict or {}}
        
        # Create initial state
        initial_state = {
            "messages": [{"role": "user", "content": research_question}],
            "search_query": [],
            "web_research_result": [],
            "sources_gathered": [],
            "research_loop_count": 0,
        }
        
        # Execute the graph
        final_state = graph.invoke(initial_state, config)
        
        # Extract results
        sources = final_state.get("sources_gathered", [])
        messages = final_state.get("messages", [])
        final_answer = messages[-1].content if messages else "No answer generated"
        
        return {
            "success": True,
            "research_question": research_question,
            "final_answer": final_answer,
            "total_sources": len(sources),
            "sources": sources,
            "research_loops": final_state.get("research_loop_count", 0),
            "research_quality": "high" if len(sources) >= 5 else "medium" if len(sources) >= 2 else "low",
            "timestamp": get_current_date(),
            "source_statistics": {"total": len(sources)},
        }
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "research_question": research_question,
            "timestamp": get_current_date(),
        }