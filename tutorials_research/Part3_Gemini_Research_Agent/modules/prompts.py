"""
Prompt Templates for Multi-Provider Research Workflows

Production-tested prompt templates adapted from Google's Gemini LangGraph reference
implementation, optimized for multi-provider LLM compatibility.
"""

from datetime import datetime
from typing import Dict, Any, List


def get_current_date() -> str:
    """Get current date in a readable format"""
    return datetime.now().strftime("%B %d, %Y")


def get_current_year() -> str:
    """Get current year as string"""
    return str(datetime.now().year)


# Core prompt templates (adapted from reference)

QUERY_WRITER_INSTRUCTIONS = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

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


WEB_SEARCHER_INSTRUCTIONS = """Conduct targeted web searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}"""


REFLECTION_INSTRUCTIONS = """You are an expert research assistant analyzing summaries about "{research_topic}".

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


ANSWER_INSTRUCTIONS = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- Include the sources you used from the Summaries in the answer correctly, use markdown format (e.g. [source](url)). THIS IS A MUST.

User Context:
{research_topic}

Summaries:
{summaries}"""


# Extended prompt templates for multi-provider compatibility

ACADEMIC_QUERY_INSTRUCTIONS = """You are a research librarian helping to create search queries for academic literature review.

Research Question: "{research_topic}"

Generate {number_queries} diverse academic search queries that would help comprehensively research this topic. Focus on:

1. Different aspects and perspectives of the topic
2. Methodological approaches and theoretical frameworks
3. Recent developments and current state of research
4. Applications and practical implementations
5. Gaps and future directions

Current date: {current_date}

Format your response as JSON:
{{
    "rationale": "Brief explanation of the search strategy",
    "query": ["query1", "query2", "query3"]
}}

Ensure queries will find peer-reviewed academic sources and recent publications."""


TECHNICAL_QUERY_INSTRUCTIONS = """You are a technical researcher creating search queries for implementation-focused research.

Technical Question: "{research_topic}"

Generate {number_queries} technical search queries focusing on:

1. Implementation details and code examples
2. Performance benchmarks and comparisons
3. Best practices and design patterns
4. Tools, libraries, and frameworks
5. Real-world case studies and applications

Current date: {current_date}

Format as JSON with rationale and query list.
Prioritize queries that will find technical documentation, GitHub repositories, and developer resources."""


COMPREHENSIVE_REFLECTION_INSTRUCTIONS = """You are a senior research analyst conducting a comprehensive review of research completeness.

Original Question: "{research_topic}"
Current Date: {current_date}

RESEARCH SUMMARIES:
{summaries}

ANALYSIS FRAMEWORK:
Evaluate the research across these dimensions:

1. **Completeness**: Are all major aspects of the topic covered?
2. **Depth**: Is there sufficient detail and evidence for each aspect?
3. **Currency**: Are recent developments (2023-2024) adequately represented?
4. **Diversity**: Are multiple perspectives, methodologies, and sources included?
5. **Applications**: Are practical uses, implementations, and real-world examples covered?
6. **Limitations**: Are challenges, limitations, and open questions addressed?

RESPONSE FORMAT:
Provide your analysis as a JSON object:

{{
    "is_sufficient": boolean,
    "overall_assessment": "Brief summary of research quality",
    "knowledge_gap": "Specific areas needing more coverage",
    "follow_up_queries": ["specific query 1", "specific query 2"],
    "coverage_scores": {{
        "completeness": 0.0-1.0,
        "depth": 0.0-1.0,
        "currency": 0.0-1.0,
        "diversity": 0.0-1.0
    }}
}}

Be thorough in your analysis and specific in identifying gaps."""


SYNTHESIS_INSTRUCTIONS = """You are a research synthesis expert creating a comprehensive final answer.

User's Question: "{research_topic}"
Current Date: {current_date}

RESEARCH FINDINGS:
{summaries}

SYNTHESIS REQUIREMENTS:

1. **Structure**: Organize information logically with clear sections
2. **Evidence**: Support all claims with specific source references
3. **Balance**: Present multiple perspectives where they exist
4. **Currency**: Emphasize recent developments and current state
5. **Completeness**: Address all aspects of the original question
6. **Actionability**: Include practical implications where relevant

FORMATTING REQUIREMENTS:
- Use markdown formatting for readability
- Include proper source citations: [Source Title](URL)
- Use headings to organize content
- Include bullet points for key findings
- Add a brief conclusion section

CITATION REQUIREMENTS:
- Every major claim must be cited
- Use the exact URLs provided in the summaries
- Format: [Brief Source Description](full_url)

Generate a comprehensive, well-structured research report that fully answers the user's question."""


# Prompt template functions for dynamic generation

def format_query_prompt(research_topic: str, number_queries: int = 3, style: str = "general") -> str:
    """
    Format query generation prompt based on research style.
    
    Args:
        research_topic: The research question or topic
        number_queries: Number of queries to generate
        style: Prompt style ("general", "academic", "technical")
        
    Returns:
        Formatted prompt string
    """
    current_date = get_current_date()
    
    if style == "academic":
        template = ACADEMIC_QUERY_INSTRUCTIONS
    elif style == "technical":
        template = TECHNICAL_QUERY_INSTRUCTIONS
    else:
        template = QUERY_WRITER_INSTRUCTIONS
    
    return template.format(
        research_topic=research_topic,
        number_queries=number_queries,
        current_date=current_date
    )


def format_web_search_prompt(research_topic: str, additional_context: str = "") -> str:
    """
    Format web search prompt with optional additional context.
    
    Args:
        research_topic: The search query or topic
        additional_context: Additional context or instructions
        
    Returns:
        Formatted prompt string
    """
    current_date = get_current_date()
    base_prompt = WEB_SEARCHER_INSTRUCTIONS.format(
        research_topic=research_topic,
        current_date=current_date
    )
    
    if additional_context:
        base_prompt += f"\n\nAdditional Context:\n{additional_context}"
    
    return base_prompt


def format_reflection_prompt(research_topic: str, summaries: List[str], style: str = "standard") -> str:
    """
    Format reflection prompt based on analysis depth.
    
    Args:
        research_topic: Original research question
        summaries: List of research summaries to analyze
        style: Analysis style ("standard", "comprehensive")
        
    Returns:
        Formatted prompt string
    """
    summaries_text = "\n\n---\n\n".join(summaries)
    
    if style == "comprehensive":
        template = COMPREHENSIVE_REFLECTION_INSTRUCTIONS
    else:
        template = REFLECTION_INSTRUCTIONS
    
    return template.format(
        research_topic=research_topic,
        summaries=summaries_text,
        current_date=get_current_date()
    )


def format_answer_prompt(research_topic: str, summaries: List[str], style: str = "standard") -> str:
    """
    Format final answer synthesis prompt.
    
    Args:
        research_topic: Original research question
        summaries: List of research summaries to synthesize
        style: Synthesis style ("standard", "comprehensive")
        
    Returns:
        Formatted prompt string
    """
    summaries_text = "\n\n---\n\n".join(summaries)
    
    if style == "comprehensive":
        template = SYNTHESIS_INSTRUCTIONS
    else:
        template = ANSWER_INSTRUCTIONS
    
    return template.format(
        research_topic=research_topic,
        summaries=summaries_text,
        current_date=get_current_date()
    )


# Utility functions for prompt customization

def get_domain_specific_instructions(domain: str) -> Dict[str, str]:
    """Get domain-specific instruction additions for various research fields"""
    
    domain_instructions = {
        "machine_learning": {
            "query_focus": "Include queries for algorithms, datasets, benchmarks, and recent model architectures",
            "search_emphasis": "Prioritize papers from top ML conferences (NeurIPS, ICML, ICLR) and recent arXiv preprints",
            "reflection_criteria": "Assess coverage of theoretical foundations, empirical results, and practical applications"
        },
        "quantum_computing": {
            "query_focus": "Cover quantum algorithms, hardware platforms, error correction, and near-term applications",
            "search_emphasis": "Include both theoretical advances and experimental demonstrations",
            "reflection_criteria": "Evaluate theoretical depth, experimental validation, and practical feasibility"
        },
        "biotechnology": {
            "query_focus": "Include molecular mechanisms, clinical applications, regulatory aspects, and market developments",
            "search_emphasis": "Prioritize peer-reviewed medical journals and clinical trial databases",
            "reflection_criteria": "Assess scientific rigor, clinical relevance, and regulatory compliance"
        },
        "climate_science": {
            "query_focus": "Cover climate models, observational data, policy implications, and mitigation strategies",
            "search_emphasis": "Include IPCC reports, climate data repositories, and policy documents",
            "reflection_criteria": "Evaluate scientific consensus, data quality, and policy relevance"
        }
    }
    
    return domain_instructions.get(domain, {
        "query_focus": "Generate diverse queries covering multiple aspects of the topic",
        "search_emphasis": "Prioritize recent, credible sources from authoritative domains",
        "reflection_criteria": "Assess completeness, accuracy, and current relevance"
    })


def customize_prompts_for_provider(prompts: Dict[str, str], provider: str) -> Dict[str, str]:
    """
    Customize prompts for specific LLM providers based on their strengths.
    
    Args:
        prompts: Dictionary of prompt templates
        provider: LLM provider name ("google", "openai", "dashscope", "zhipuai")
        
    Returns:
        Customized prompts dictionary
    """
    # Provider-specific optimizations
    if provider == "google":
        # Google models excel at structured reasoning
        for key in prompts:
            if "reasoning" in key.lower() or "analysis" in key.lower():
                prompts[key] = prompts[key] + "\n\nUse step-by-step reasoning to ensure thoroughness."
    
    elif provider == "openai":
        # OpenAI models are strong at following format instructions
        for key in prompts:
            if "format" in key.lower() or "json" in prompts[key].lower():
                prompts[key] = prompts[key] + "\n\nStrictly follow the specified format."
    
    elif provider in ["dashscope", "zhipuai"]:
        # Chinese providers - ensure clarity and avoid complex cultural references
        for key in prompts:
            prompts[key] = prompts[key].replace("e.g.", "for example")
            prompts[key] = prompts[key].replace("i.e.", "that is")
    
    return prompts


if __name__ == "__main__":
    print("📝 MULTI-PROVIDER RESEARCH PROMPTS")
    print("=" * 35)
    
    print("\n🎯 Available Prompt Templates:")
    templates = [
        "Query Generation (General, Academic, Technical)",
        "Web Search Instructions", 
        "Reflection Analysis (Standard, Comprehensive)",
        "Answer Synthesis (Standard, Comprehensive)"
    ]
    
    for template in templates:
        print(f"   • {template}")
    
    print(f"\n📅 Current Date Context: {get_current_date()}")
    
    print("\n🔧 Dynamic Formatting Examples:")
    
    # Example query prompt
    sample_topic = "What are recent advances in quantum error correction?"
    query_prompt = format_query_prompt(sample_topic, 3, "academic")
    print(f"\n📋 Academic Query Prompt (first 100 chars):")
    print(f"   {query_prompt[:100]}...")
    
    # Example domain customization
    domain_instructions = get_domain_specific_instructions("quantum_computing")
    print(f"\n🔬 Quantum Computing Domain Instructions:")
    print(f"   Focus: {domain_instructions['query_focus'][:60]}...")
    
    print("\n✅ Key Features:")
    features = [
        "Production-tested templates from Google reference",
        "Multi-provider compatibility optimizations",
        "Dynamic formatting with current date injection",
        "Domain-specific customization support",
        "Flexible prompt styling (standard/comprehensive)"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    print("\n🔄 Usage:")
    print("""
    # Format prompts dynamically
    prompt = format_query_prompt("research topic", 3, "academic")
    
    # Get domain-specific instructions
    instructions = get_domain_specific_instructions("machine_learning")
    
    # Customize for provider
    optimized = customize_prompts_for_provider(prompts, "google")
    """)
    
    print("\n✅ Production-ready prompt system loaded!")