# Part 1: Foundations - Code to LaTeX Agent

## 🎯 Research Scenario
**Problem**: You have algorithm implementations in Python/JavaScript and need to convert them to publication-ready LaTeX for your research paper.

**Real-world Challenge**: Manual conversion is time-consuming and error-prone. You need mathematical expressions, algorithm blocks, and formatted equations that match academic standards.

## 🎓 Learning Objectives

By the end of this tutorial, you will understand:

1. **Agent Fundamentals**: Core concepts without complex frameworks
2. **LLM Provider Integration**: Multi-provider setup for global accessibility
3. **Hardcoded vs Intelligent Approaches**: Comparing rule-based and LLM-based solutions
4. **StateGraph Basics**: Introduction to LangGraph without mocks
5. **Research Application**: Building tools you can use in your actual research workflow

## 📚 What You'll Build

### Main Example: Code-to-LaTeX Converter
- **Input**: Python NumPy code for mathematical operations
- **Output**: Publication-ready LaTeX mathematical expressions
- **Approaches**: Compare hardcoded regex-based vs LLM-based conversion

### Supported Conversions
1. **Mathematical Expressions**: `np.sqrt(x**2 + y**2)` → `$\\sqrt{x^2 + y^2}$`
2. **Algorithm Blocks**: Python functions → LaTeX algorithm environment
3. **Matrix Operations**: NumPy arrays → LaTeX matrices
4. **Statistical Formulas**: SciPy functions → LaTeX equations

## 🛠 Technical Concepts

### Core Agent Architecture
```python
class CodeToLatexAgent:
    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.conversion_patterns = {...}
    
    def convert(self, code: str) -> str:
        # LLM-based intelligent conversion
        pass
```

### LLM Provider Support
- **Google Gemini**: Best for mathematical reasoning
- **OpenAI GPT-4**: Reliable for code understanding
- **DashScope Qwen**: Cost-effective with good performance
- **Zhipu GLM**: Optimized for Chinese researchers

### StateGraph Introduction
Basic workflow: Input → Analysis → Conversion → Validation → Output

## 📋 Prerequisites

- Python 3.8+
- At least one LLM provider API key
- Basic understanding of mathematical notation
- Familiarity with LaTeX (helpful but not required)

## ⏱ Estimated Time
**2 hours** (including hands-on coding and experimentation)

## 🚀 Getting Started

1. **Environment Check**: Ensure you've completed `../setup_environment.ipynb`
2. **API Keys**: Verify at least one LLM provider is configured
3. **Open Notebook**: Start with `01_Tutorial.ipynb`

## 📖 Tutorial Structure

### Section 1: Problem Analysis (30 min)
- Research context: Why automate code-to-LaTeX conversion?
- Manual process pain points
- Requirements for publication-quality output

### Section 2: Hardcoded Approach (30 min)
- Regex-based conversion system
- Pattern matching for common mathematical operations
- Limitations and edge cases

### Section 3: Agent-Based Solution (45 min)
- LLM provider setup and configuration
- Building the intelligent conversion agent
- Handling complex mathematical expressions

### Section 4: StateGraph Workflow (30 min)
- Introduction to LangGraph concepts
- Building a simple conversion workflow
- State management for multi-step processes

### Section 5: Comparison and Applications (15 min)
- Performance comparison
- When to use each approach
- Integration into research workflow

## 🎯 Key Insights

`★ Insight ─────────────────────────────────────`
- LLM agents excel at understanding context and intent
- Hardcoded approaches are faster but limited in scope
- Multi-provider setup ensures global accessibility
- StateGraph provides structure without complexity
`─────────────────────────────────────────────────`

## 📁 Files in This Part

- `01_Tutorial.ipynb`: Main interactive tutorial
- `modules/agent_basics.py`: Core agent concepts
- `modules/code_to_latex.py`: Conversion implementations
- `modules/llm_providers.py`: Multi-provider setup
- `modules/graph_introduction.py`: Basic StateGraph patterns

## 🔗 Next Steps

After completing Part 1:
- **Part 2**: Multi-Agent Router for research task delegation
- **Part 3**: Gemini Research Agent for web research workflows
- **Part 4**: DAG Architecture for complex research pipelines
- **Part 5**: Complete PHM system integration

## 📝 Expected Outputs

By the end of this tutorial, you'll have:
1. Working code-to-LaTeX converter
2. Multi-provider LLM setup
3. Basic StateGraph implementation
4. Understanding of agent vs hardcoded approaches
5. Tools you can adapt for your research needs

---

**Ready to begin?** Open [`01_Tutorial.ipynb`](01_Tutorial.ipynb) to start your journey into research-oriented LLM agents!