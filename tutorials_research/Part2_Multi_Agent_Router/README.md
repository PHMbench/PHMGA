# Part 2: Multi-Agent Router - Research Assistant

## 🎯 Research Scenario
**Problem**: You're writing a research paper and need to conduct a comprehensive literature review, format citations in different styles (IEEE/APA/MLA), generate figure captions, and summarize key findings. Each task requires different expertise and tools.

**Real-world Challenge**: Managing multiple research tasks manually is overwhelming. You need different approaches for literature search vs citation formatting vs content summarization. A single agent can't excel at all these specialized tasks.

## 🎓 Learning Objectives

By the end of this tutorial, you will understand:

1. **Multi-Agent Architecture**: How specialized agents collaborate
2. **Router Patterns**: Intelligent task delegation and routing
3. **Real Tool Integration**: Working with ArXiv, Semantic Scholar, CrossRef
4. **Agent Coordination**: Managing state and communication between agents
5. **Research Workflow Orchestration**: Building production-ready research assistants

## 📚 What You'll Build

### Main System: Intelligent Research Assistant
- **Router Agent**: Analyzes queries and delegates to specialized agents
- **Literature Search Agent**: Finds and retrieves relevant papers
- **Citation Formatter Agent**: Converts to IEEE/APA/MLA styles
- **Summary Generator Agent**: Extracts key findings and insights
- **Figure Caption Agent**: Generates captions for research visuals

### Example Workflow
```
Input: "Find recent advances in transformer architectures and format citations in IEEE style"

Router → Literature Search → Summary Generator → Citation Formatter → Final Report
```

## 🛠 Technical Concepts

### Multi-Agent Architecture
```python
class ResearchRouter:
    def __init__(self):
        self.agents = {
            "literature": LiteratureSearchAgent(),
            "citations": CitationFormatterAgent(), 
            "summary": SummaryGeneratorAgent(),
            "figures": FigureCaptionAgent()
        }
    
    def route_task(self, query: str) -> List[str]:
        # Intelligent routing logic
        pass
```

### Real Tool Integration
- **ArXiv API**: Academic paper search and retrieval
- **Semantic Scholar API**: Citation networks and metadata
- **CrossRef API**: DOI resolution and bibliographic data
- **Web Scraping**: Supplementary content extraction

### Agent Communication Patterns
- **Sequential**: A → B → C (pipeline processing)
- **Parallel**: A + B + C (concurrent processing)
- **Hierarchical**: Router → Specialists → Aggregator
- **Collaborative**: Agents share intermediate results

## 📋 Prerequisites

- Completion of Part 1 (Foundations)
- At least one LLM provider configured
- Internet connection for API access
- Basic understanding of research workflows

## ⏱ Estimated Time
**3 hours** (including hands-on implementation and testing)

## 🚀 Getting Started

1. **Environment Check**: Ensure `../setup_environment.ipynb` shows ready status
2. **API Access**: Verify internet connectivity for research tool APIs
3. **LLM Ready**: Confirm working LLM provider from Part 1
4. **Open Notebook**: Start with `02_Tutorial.ipynb`

## 📖 Tutorial Structure

### Section 1: Multi-Agent Concepts (45 min)
- Router pattern fundamentals
- Agent specialization principles
- Communication and coordination
- Real-world research applications

### Section 2: Literature Search Agent (45 min)
- ArXiv API integration
- Semantic Scholar connectivity
- Query optimization and filtering
- Result ranking and relevance scoring

### Section 3: Citation & Summary Agents (45 min)
- Citation style formatting (IEEE/APA/MLA)
- Key finding extraction
- Abstract and conclusion generation
- Quality assessment metrics

### Section 4: Router Integration (30 min)
- Intelligent task analysis
- Dynamic agent selection
- Workflow orchestration
- Error handling and fallbacks

### Section 5: Complete System Demo (15 min)
- End-to-end literature review
- Performance optimization
- Production deployment considerations

## 🎯 Key Insights

`★ Insight ─────────────────────────────────────`
- Router agents analyze task complexity and delegate appropriately
- Specialized agents outperform general-purpose agents on domain tasks
- Real API integration provides authentic research capabilities
- Agent coordination requires careful state management
`─────────────────────────────────────────────────`

## 📁 Files in This Part

- `02_Tutorial.ipynb`: Main interactive tutorial
- `modules/research_router.py`: Router agent implementation
- `modules/literature_agent.py`: ArXiv and Semantic Scholar integration
- `modules/citation_agent.py`: Multi-style citation formatting
- `modules/figure_caption_agent.py`: Figure and table caption generation
- `modules/research_tools.py`: API wrappers and utilities

## 🔧 Real Tools Integration

### ArXiv API
- Paper search by keywords, authors, categories
- Full-text access and metadata extraction
- Recent paper filtering and relevance ranking

### Semantic Scholar API
- Citation network analysis
- Paper influence metrics
- Author disambiguation and collaboration networks

### CrossRef API
- DOI resolution and validation
- Publisher metadata access
- Reference linking and verification

## 💼 Research Applications

### Literature Review Automation
```python
query = "transformer attention mechanisms 2023-2024"
results = research_assistant.conduct_literature_review(
    query=query,
    max_papers=20,
    citation_style="ieee",
    include_figures=True
)
```

### Citation Management
```python
paper_doi = "10.1038/nature12373"
citations = citation_agent.format_citation(
    doi=paper_doi,
    styles=["ieee", "apa", "mla"]
)
```

### Figure Caption Generation
```python
figure_description = "Neural network architecture diagram showing transformer layers"
caption = figure_agent.generate_caption(
    description=figure_description,
    context="machine learning research paper"
)
```

## 🔗 Connection to Other Parts

- **Part 1 Foundation**: Uses LLM providers and basic agent concepts
- **Part 3 Gemini**: Advanced web research with reflection loops
- **Part 4 DAG**: Complex workflow orchestration patterns  
- **Part 5 PHM**: Production system integration examples

## 📊 Expected Outcomes

By completing Part 2, you'll have:

1. **Working Research Assistant**: Multi-agent system for literature review
2. **Real API Integration**: Connections to academic databases
3. **Flexible Citation System**: Support for multiple academic styles
4. **Scalable Architecture**: Pattern for adding new specialized agents
5. **Production-Ready Code**: System you can deploy for research use

## 🚧 Common Challenges

### API Rate Limits
- Implement proper request throttling
- Cache results to avoid repeated calls
- Handle API failures gracefully

### Citation Style Complexity
- IEEE vs APA vs MLA formatting differences
- Special cases for different publication types
- Author name disambiguation

### Agent Coordination
- State management between agents
- Error propagation and recovery
- Performance optimization

## 💡 Pro Tips

1. **Start Simple**: Begin with 2-3 agents, add more as needed
2. **Cache Everything**: Research API calls are expensive
3. **Validate Citations**: Always verify generated citations
4. **Monitor Performance**: Track agent success rates and timing
5. **Plan for Scale**: Design for larger research projects

---

**Ready to build your research assistant?** Open [`02_Tutorial.ipynb`](02_Tutorial.ipynb) to start creating a multi-agent system for academic research!