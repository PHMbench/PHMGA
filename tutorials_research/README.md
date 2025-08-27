# 📚 Research-Oriented PHMGA Tutorial Series

Welcome to the **Research-Oriented PHMGA Tutorial Series** - a comprehensive learning path designed specifically for researchers and academics who want to leverage LLM agents for their paper writing and research workflows.

## 🎯 Tutorial Philosophy

Unlike traditional tutorials that use abstract examples, this series focuses on **real research tasks** that academics face every day:

- Converting code implementations to publication-ready LaTeX
- Automating literature reviews with proper citations
- Conducting comprehensive web research for paper backgrounds  
- Building complex research pipelines using DAG architectures
- Integrating everything into production-ready systems like PHMGA

**No Mock Implementations** - Every example uses real LLM providers, actual APIs, and working code you can adapt to your research.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Lab/Notebook
- API keys for at least one LLM provider

### Installation
```bash
# Clone and navigate to tutorials_research
cd tutorials_research/

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Setup
Start with the setup notebook to verify your environment:
```bash
jupyter lab setup_environment.ipynb
```

## 📖 Learning Path

### Part 1: Foundations - Code to LaTeX Agent
**🎯 Research Scenario**: Converting algorithm implementations to publication-ready LaTeX

**What You'll Learn**:
- Basic agent concepts without complex frameworks
- Multi-provider LLM setup (Google/OpenAI/DashScope/Zhipu) 
- Comparison of hardcoded vs intelligent approaches
- Introduction to LangGraph StateGraph

**Real Example**: Python NumPy code → LaTeX mathematical expressions

**Duration**: ~2 hours

---

### Part 2: Multi-Agent Router - Research Assistant  
**🎯 Research Scenario**: Automated literature review and citation management

**What You'll Learn**:
- Router patterns for task delegation
- Multi-agent collaboration
- Real tool integration (ArXiv, Semantic Scholar)
- Citation formatting in multiple styles (IEEE/APA/MLA)

**Real Example**: Generate a complete literature review section with proper citations

**Duration**: ~3 hours

---

### Part 3: Gemini Research Agent - Web Research Workflow
**🎯 Research Scenario**: Comprehensive background research using web sources

**What You'll Learn**:
- Query generation from research questions
- Parallel web research execution
- Reflection loops for knowledge gap identification
- Source credibility and citation tracking

**Real Example**: Research latest advances in a specific field using the Gemini research agent

**Duration**: ~2.5 hours

---

### Part 4: DAG Architecture - Research Pipeline Construction
**🎯 Research Scenario**: Building complex, non-linear research workflows

**What You'll Learn**:
- DAG fundamentals and node relationships
- PHMGA's DAG architecture deep dive
- Dynamic graph construction
- Immutable state management patterns

**Real Example**: Build a systematic review pipeline as a DAG

**Duration**: ~3 hours

---

### Part 5: PHM Case Study - Complete System Integration
**🎯 Research Scenario**: Industrial bearing fault diagnosis (complete production system)

**What You'll Learn**:
- Integration of all learned concepts
- Builder-Executor pattern for flexibility
- Multi-agent system coordination
- Production-ready considerations

**Real Example**: Complete walkthrough of PHMGA case1 bearing fault diagnosis

**Duration**: ~4 hours

## 🛠 Supported LLM Providers

This tutorial series supports multiple LLM providers to accommodate researchers worldwide:

| Provider | Models | API Key Required | Best For |
|----------|--------|------------------|----------|
| **Google Gemini** | gemini-2.5-pro, gemini-2.5-flash | `GEMINI_API_KEY` | Research tasks, web integration |
| **OpenAI** | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` | General-purpose, reliable |
| **DashScope (Qwen)** | qwen-plus, qwen-turbo | `DASHSCOPE_API_KEY` | Cost-effective, Chinese support |
| **Zhipu AI** | glm-4 | `ZHIPUAI_API_KEY` | Chinese researchers |

## 📚 Key Features

- ✅ **Real Research Examples**: Every tutorial addresses actual academic tasks
- ✅ **No Mocks**: All implementations use real APIs and tools
- ✅ **Multi-Provider Support**: Works with global LLM providers
- ✅ **Progressive Learning**: From basic concepts to production systems
- ✅ **Hands-on Coding**: Interactive Jupyter notebooks
- ✅ **Production Ready**: Patterns you can use in real research

## 🎓 Who Should Use This

- **PhD Students**: Automate literature reviews and paper writing
- **Researchers**: Build custom research workflows
- **Academics**: Learn modern AI agent patterns
- **Engineers**: Understand production LLM agent systems
- **Data Scientists**: Apply agent patterns to research pipelines

## 📝 Tutorial Structure

Each part follows a consistent structure:
1. **Research Scenario**: Real academic problem
2. **Conceptual Overview**: Key concepts and insights  
3. **Hands-on Implementation**: Step-by-step coding
4. **Comparison Analysis**: Different approaches compared
5. **Practical Applications**: How to adapt to your research

## 🆘 Getting Help

- **Environment Issues**: Check `setup_environment.ipynb`
- **API Problems**: Verify your API keys in `.env`
- **Concept Questions**: See individual part READMEs
- **Bug Reports**: Create issues in the main repository

## 🔄 Tutorial Updates

This tutorial series is actively maintained. Check for updates:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## 📄 License

This tutorial series is part of the PHMGA project and follows the same license terms.

---

**Ready to start?** Begin with [setup_environment.ipynb](setup_environment.ipynb) to configure your research environment!