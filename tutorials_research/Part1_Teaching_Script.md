# Part 1 Teaching Script: Introduction to Research Agents
**Duration:** 2 hours  
**Class Size:** Individual or small group  
**Prerequisites:** Python basics, basic understanding of research workflows  

---

## 🎓 Class Opening (10 minutes)

**Welcome, everyone!** 

Welcome to "Introduction to Research Agents" - the first part of our Research-Oriented PHMGA Tutorial Series. I'm excited to start this journey with you into the world of intelligent agents for academic research.

### Today's Learning Objectives

By the end of today's class, you will:
1. **Understand** why agents are transformative for researchers
2. **Build** your first Code-to-LaTeX agent from scratch
3. **Configure** multiple LLM providers for global accessibility  
4. **Compare** hardcoded vs intelligent approaches
5. **Create** your first StateGraph workflow

### Why This Matters

Let me ask you a question: *How many hours have you spent converting mathematical expressions from your Python code into LaTeX format for papers?* 

*[Pause for responses]*

Exactly! This tedious work is why we're here. Today, we're building tools that eliminate this manual labor and give you more time for actual research.

### Our Real Research Challenge

**Scenario:** You've implemented a brilliant algorithm in Python. Now you need to present it in your paper with proper mathematical notation. 

**Traditional Approach:** 
- Copy code snippets
- Manually convert `np.sqrt(x**2 + y**2)` to `$\\sqrt{x^2 + y^2}$`
- Hope you didn't make transcription errors
- Repeat for dozens of expressions

**Our Approach Today:**
- Build an intelligent agent that understands your code
- Automatically converts to publication-ready LaTeX
- Works with any LLM provider you have access to
- Scales to any complexity level

*Any questions before we dive in?*

---

## 📚 Lesson 1: Why Agents Matter for Researchers (30 minutes)

### The Research Productivity Crisis

Let me share something that might surprise you. According to recent studies, researchers spend **60-70%** of their time on non-research tasks. That's more time on bureaucracy than breakthrough discoveries!

**Traditional Research Workflow:**
```
Idea → Literature Search (manual) → Code Implementation → 
Paper Writing (manual formatting) → Citation Management (manual) → 
Submission → Revisions (more manual work)
```

**Agent-Enhanced Workflow:**
```
Idea → Agent-Assisted Research → Code Implementation → 
Agent-Generated LaTeX → Auto-Citations → Submission → 
Agent-Supported Revisions
```

### Three Core Benefits of Agents

#### 1. Task Decomposition
Instead of "write a literature review" (overwhelming), agents break this into:
- Search for recent papers
- Extract key findings
- Identify research gaps  
- Generate structured summary
- Format citations

#### 2. Parallel Execution
While you're thinking about research design, agents can simultaneously:
- Monitor new publications in your field
- Update citation databases
- Prepare background research
- Format existing content

#### 3. Elimination of Manual Labor
No more:
- Copy-pasting between applications
- Manual citation formatting
- Tedious code-to-LaTeX conversion
- Repetitive formatting tasks

### Interactive Exercise: Identify Your Time Wasters

*[This would be done in the notebook]*

Take 2 minutes to think about your current research workflow. What tasks would you love to automate?

Common answers I hear:
- Literature search and summarization
- Citation formatting
- Code documentation
- Figure caption generation
- Mathematical notation conversion

*Let's keep these in mind as we build our first agent.*

### The Code-to-LaTeX Problem

This is perfect for demonstrating agent benefits because:
- **Common Pain Point**: Every researcher with code faces this
- **Clear Input/Output**: Python code → LaTeX equations
- **Measurable Success**: Correct mathematical notation
- **Immediate Utility**: Use it in your next paper

*Ready to see this in action? Let's build our first agent!*

---

## 🛠 Lesson 2: Building Your First Code-to-LaTeX Agent (45 minutes)

### Setting Up Our Environment

First, let's make sure everyone's environment is ready. Open your notebook and run the setup cell.

```python
# Let's start by importing our foundation modules
from tutorials_research.Part1_Foundations.modules import (
    create_multi_provider_llm,
    CodeToLatexAgent,
    HardcodedConverter
)

print("🎯 Part 1: Introduction to Research Agents")
print("📚 Building intelligent tools for academic workflows")
```

*If anyone gets import errors, let me know now - we'll troubleshoot together.*

### The Simple Approach First

Let's start with the hardcoded approach to understand the problem:

```python
class HardcodedCodeToLatex:
    def __init__(self):
        self.patterns = {
            r'np\.sqrt\(([^)]+)\)': r'\\sqrt{\1}',
            r'np\.sum\(([^)]+)\)': r'\\sum{\1}',
            r'\*\*2': r'^2',
            r'\*\*([0-9]+)': r'^{\1}'
        }
    
    def convert(self, code: str) -> str:
        result = code
        for pattern, replacement in self.patterns.items():
            result = re.sub(pattern, replacement, result)
        return f"${result}$"
```

**Let's test this together:**

```python
hardcoded = HardcodedCodeToLatex()
test_code = "np.sqrt(x**2 + y**2)"
print(f"Input:  {test_code}")
print(f"Output: {hardcoded.convert(test_code)}")
```

*What do you expect the output to be?*

**Expected:** `$\\sqrt{x^2 + y^2}$`

*Let's run it and see...*

### The Limitations Become Clear

Now let's try something more complex:

```python
complex_code = "np.log(np.exp(alpha * beta) + gamma)"
print(f"Complex input: {complex_code}")
print(f"Hardcoded result: {hardcoded.convert(complex_code)}")
```

*What went wrong here?*

The hardcoded approach fails because:
1. Limited pattern coverage
2. No understanding of mathematical context
3. Brittle regex patterns
4. No handling of nested expressions

**This is exactly why we need intelligent agents!**

### Building the Intelligent Agent

Now let's build our AI-powered version:

```python
class IntelligentCodeToLatexAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = """
        Convert this Python mathematical expression to LaTeX format.
        
        Rules:
        1. Use proper LaTeX mathematical notation
        2. Wrap the result in $ symbols for inline math
        3. Handle nested functions correctly
        4. Maintain mathematical meaning
        
        Python code: {code}
        
        LaTeX output:
        """
    
    def convert(self, code: str) -> str:
        prompt = self.prompt_template.format(code=code)
        response = self.llm.invoke(prompt)
        return response.strip()
```

### Multi-Provider Magic

Here's where our multi-provider approach shines:

```python
# Let's try with different providers
providers_to_test = ["google", "openai", "dashscope"]  # Use what you have

for provider in providers_to_test:
    if provider_available(provider):
        print(f"\n🤖 Testing with {provider}:")
        llm = create_multi_provider_llm(provider)
        agent = IntelligentCodeToLatexAgent(llm)
        
        result = agent.convert("np.sqrt(x**2 + y**2)")
        print(f"Result: {result}")
```

*Let's run this and compare results across providers.*

### Interactive Coding Session

*[Students work in their notebooks]*

**Your turn!** Take 10 minutes to:

1. Test both approaches with these expressions:
   - `x**3 + y**3`
   - `np.exp(-0.5 * (x - mu)**2 / sigma**2)`
   - `np.log(1 + np.exp(x))`

2. Compare the results
3. Note where each approach succeeds or fails

*Let's discuss your findings...*

### Understanding the Differences

**Hardcoded Approach:**
- ✅ Fast and predictable
- ✅ No external dependencies
- ❌ Limited coverage
- ❌ Brittle and error-prone

**Intelligent Agent:**
- ✅ Understands mathematical context
- ✅ Handles complex expressions
- ✅ Adapts to new patterns
- ❌ Requires LLM access
- ❌ Slightly slower

**The Key Insight:** Use the right tool for the job. Simple patterns? Hardcoded. Complex mathematical reasoning? Intelligent agents.

*Questions about this comparison?*

---

## 🌐 Lesson 3: Multi-Provider LLM Setup and Comparison (30 minutes)

### Why Multi-Provider Matters

Let me share a real-world story. Last month, a PhD student in China couldn't access OpenAI APIs due to regional restrictions. With our multi-provider setup, they switched to DashScope (Qwen) and continued their research without missing a beat.

**Global Research Requires Global Accessibility**

### Provider Comparison Matrix

| Provider | Best For | Availability | Cost | Mathematical Reasoning |
|----------|----------|--------------|------|----------------------|
| **Google Gemini** | Research tasks, web integration | Global | Mid | Excellent |
| **OpenAI GPT-4** | Reliable, consistent output | Limited regions | High | Excellent |
| **DashScope Qwen** | Cost-effective, Chinese support | Global | Low | Good |
| **Zhipu GLM** | Chinese researchers | China-focused | Low | Good |

### Hands-On Provider Configuration

Let's configure multiple providers together:

```python
# Provider configuration
PROVIDER_CONFIGS = {
    "google": {
        "model": "gemini-pro",
        "api_key_env": "GEMINI_API_KEY",
        "strengths": ["mathematical reasoning", "web integration"]
    },
    "openai": {
        "model": "gpt-4",
        "api_key_env": "OPENAI_API_KEY", 
        "strengths": ["consistency", "instruction following"]
    },
    "dashscope": {
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "strengths": ["cost efficiency", "Chinese support"]
    }
}

# Let's see what you have configured
available_providers = check_available_providers()
print(f"✅ Available providers: {available_providers}")
```

*What providers do you have available?*

### Comparative Testing

Let's run the same mathematical expression through different providers:

```python
test_expression = "np.log(np.sum(np.exp(x), axis=1))"  # LogSumExp

print("🧪 Testing LogSumExp conversion across providers:")
print("=" * 50)

for provider in available_providers:
    llm = create_multi_provider_llm(provider)
    agent = IntelligentCodeToLatexAgent(llm)
    
    start_time = time.time()
    result = agent.convert(test_expression)
    duration = time.time() - start_time
    
    print(f"\n{provider.upper()}:")
    print(f"Result: {result}")
    print(f"Time: {duration:.2f}s")
```

*Let's analyze the results together...*

### Quality Assessment Exercise

*[Interactive evaluation]*

For each result, let's evaluate:
1. **Mathematical Correctness**: Is the LaTeX mathematically equivalent?
2. **Notation Quality**: Does it follow LaTeX best practices?
3. **Readability**: Would this look good in a paper?

**Rate each result 1-5 for:**
- Accuracy
- Formatting
- Completeness

### The Fallback Strategy

Here's a professional pattern - automatic fallback:

```python
class RobustCodeToLatexAgent:
    def __init__(self, provider_priority=["google", "openai", "dashscope"]):
        self.providers = []
        for provider in provider_priority:
            if provider_available(provider):
                self.providers.append(provider)
    
    def convert_with_fallback(self, code: str) -> dict:
        results = {}
        
        for provider in self.providers:
            try:
                llm = create_multi_provider_llm(provider)
                agent = IntelligentCodeToLatexAgent(llm)
                result = agent.convert(code)
                
                results[provider] = {
                    "result": result,
                    "status": "success"
                }
                
                # If first provider succeeds, that's our primary result
                if len(results) == 1:
                    primary_result = result
                    
            except Exception as e:
                results[provider] = {
                    "result": None,
                    "status": f"failed: {e}"
                }
        
        return {
            "primary_result": primary_result if 'primary_result' in locals() else None,
            "all_results": results
        }
```

*This is a production-ready pattern you can use in your own research!*

---

## 🕸️ Lesson 4: Introduction to StateGraphs (15 minutes)

### What is a StateGraph?

Think of StateGraph as a **flowchart for AI workflows**. Instead of one big agent doing everything, we create a series of specialized steps that pass information between each other.

**Traditional Agent:**
```
Input → [Big AI Agent] → Output
```

**StateGraph Workflow:**
```
Input → [Analyze] → [Convert] → [Validate] → [Format] → Output
```

### Why StateGraphs Matter

1. **Modularity**: Each step has one job
2. **Debuggability**: Easy to see where things go wrong
3. **Flexibility**: Swap out individual components
4. **Reliability**: Validate at each step

### Simple StateGraph Example

Let's build a basic workflow for our code converter:

```python
from langgraph.graph import StateGraph, END

# Define our state
class ConversionState:
    input_code: str = ""
    analysis_result: str = ""
    latex_output: str = ""
    validation_passed: bool = False

# Create the graph
workflow = StateGraph(ConversionState)

# Add nodes
workflow.add_node("analyze", analyze_code)
workflow.add_node("convert", convert_to_latex)  
workflow.add_node("validate", validate_output)
workflow.add_node("format", format_result)

# Add edges (workflow steps)
workflow.add_edge("analyze", "convert")
workflow.add_edge("convert", "validate")
workflow.add_edge("validate", "format")
workflow.add_edge("format", END)

# Set entry point
workflow.set_entry_point("analyze")

# Compile the graph
app = workflow.compile()
```

### Interactive Demo

Let's run our StateGraph:

```python
# Test our workflow
initial_state = ConversionState(
    input_code="np.sqrt(a**2 + b**2)"
)

# Execute the workflow
final_state = app.invoke(initial_state)

print("🕸️ StateGraph Execution:")
print(f"Input: {final_state.input_code}")
print(f"Analysis: {final_state.analysis_result}")
print(f"Output: {final_state.latex_output}")
print(f"Valid: {final_state.validation_passed}")
```

### Why Start with StateGraphs?

Even though this seems like overkill for simple conversion, learning StateGraphs now prepares you for:
- **Part 2**: Multi-agent coordination
- **Part 3**: Complex research workflows
- **Part 4**: DAG-based processing
- **Part 5**: Production systems

*Think of this as learning to drive in a parking lot before hitting the highway!*

### Quick Checkpoint

*[Knowledge check]*

**Question 1:** What's the main advantage of StateGraph over a single large agent?

**Question 2:** In our conversion workflow, what happens if the validation step fails?

**Question 3:** How would you add a step to handle different mathematical notation styles?

*Let's discuss your answers...*

---

## ✅ Assessment: Build Your Working Converter (20 minutes)

### Your Challenge

Build a complete Code-to-LaTeX converter that:

1. **Uses your preferred LLM provider**
2. **Handles at least 5 different mathematical expressions**
3. **Includes basic validation**
4. **Provides helpful error messages**

### Test Cases

Your converter should handle:
```python
test_cases = [
    "x**2 + y**2",
    "np.sqrt(np.sum(errors**2))",
    "np.exp(-0.5 * (x - mu)**2 / sigma**2)",
    "np.log(1 + np.exp(x))",
    "np.sum(weights * features, axis=1)"
]
```

### Starter Template

```python
class MyCodeToLatexConverter:
    def __init__(self):
        # TODO: Initialize your preferred LLM
        # TODO: Set up any patterns or configurations
        pass
    
    def convert(self, code: str) -> str:
        # TODO: Implement conversion logic
        # TODO: Add error handling
        # TODO: Include validation
        pass
    
    def validate_output(self, latex: str) -> bool:
        # TODO: Basic validation
        # Check for balanced $, valid LaTeX syntax, etc.
        pass
```

### Work Time

*[Students work individually]*

Take 15 minutes to implement your converter. I'll walk around to help with any issues.

**Hints:**
- Start simple, then add complexity
- Use the provider you're most comfortable with
- Don't forget error handling!

### Demo Time

*[Students share results]*

Who wants to show their converter in action? Let's see what you built!

*[Discussion of different approaches and solutions]*

---

## 🎯 Class Closing (10 minutes)

### What We Accomplished Today

Look at what you built in just 2 hours:
- ✅ **Understanding** of agent benefits for research
- ✅ **Working Code-to-LaTeX converter**
- ✅ **Multi-provider LLM setup**
- ✅ **Comparison** of hardcoded vs intelligent approaches
- ✅ **First StateGraph workflow**

### Key Insights from Today

**"Agents excel at understanding context and intent"**
- Your converter can handle expressions it's never seen before
- It understands mathematical meaning, not just syntax

**"Multi-provider setup ensures global accessibility"**
- Research shouldn't be limited by API availability
- Different providers have different strengths

**"StateGraphs provide structure without complexity"**
- Break complex tasks into manageable steps
- Foundation for advanced workflows we'll build

### Prepare for Part 2: Multi-Agent Systems

Next class, we'll explore:
- **Router patterns** for task delegation
- **Multiple specialized agents** working together
- **Real API integration** with ArXiv and Semantic Scholar
- **Citation formatting** and academic workflows

### Homework (Optional but Recommended)

1. **Expand your converter** to handle matrix operations
2. **Test with different LLM providers** if you have access
3. **Try the StateGraph approach** for your converter
4. **Think about other research automation** you'd like to build

### Final Q&A

*Any questions about today's material?*

*Concerns about the next tutorial?*

*Ideas for how you'll use this in your research?*

### Course Feedback

Quick feedback (raise hands):
- **Too fast? Too slow? Just right?**
- **More examples? More theory? More hands-on?**
- **What was most valuable today?**

---

## 📚 Resources for Continued Learning

### Documentation Links
- LangGraph Official Docs: [langgraph.org](https://langgraph.org)
- Multi-Provider Setup Guide: See `../setup_environment.ipynb`
- Code Examples: All code from today is in your notebook

### Community Support
- Questions? Check the main repository issues
- Share your improvements via pull requests
- Connect with other researchers using these tools

### Next Class
**Part 2: Multi-Agent Router - Research Assistant**
- **When:** [Next session time]
- **What to bring:** Working environment from today
- **What we'll build:** Complete research assistant with multiple specialized agents

---

**Thank you for a great first class! See you next time when we build multi-agent research systems!**

*[Class dismissed]*

---

## 📝 Teaching Notes (For Instructors)

### Timing Adjustments
- **If running fast:** Add more complex examples, deeper StateGraph exploration
- **If running slow:** Skip some comparative testing, focus on core concepts

### Common Student Questions
1. **"Which LLM provider is best?"** → Emphasize it depends on access, cost, and use case
2. **"Why not just use ChatGPT?"** → Explain API vs web interface, automation needs
3. **"Is this better than manual conversion?"** → Discuss scale and complexity

### Technical Troubleshooting
- **Import errors:** Usually environment setup issues
- **API failures:** Check API keys, rate limits
- **Model access:** Regional restrictions, account issues

### Extension Activities
- Advanced mathematical notation conversion
- Integration with existing research workflows  
- Custom validation rules for specific domains
- Performance benchmarking across providers