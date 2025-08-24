# PHMGA Tutorial Series

Welcome to the comprehensive PHMGA (Prognostics and Health Management Graph Agent) tutorial series! This progressive educational series takes you from beginner to expert level in building and deploying graph-based intelligent agents for predictive health management.

## 📚 Tutorial Structure

### Part 1: Foundations - Graph Agent Basics
**Estimated Time**: 2-3 hours  
**Prerequisites**: Basic Python knowledge, familiarity with AI concepts  
**Objective**: Understand Graph Agent fundamentals and LLM integration

📖 **[Theory: Foundations README](Part1_Foundations/README.md)**  
💻 **[Practice: 01_Tutorial Notebook](Part1_Foundations/01_Tutorial.ipynb)**

- **1.1 Graph Agent Concepts**: Understanding graph-based AI architectures
- **1.2 LLM Provider Integration**: Seamless multi-provider support (Google, OpenAI, etc.)
- **1.3 Agent Basics**: State management and decision-making patterns
- **1.4 Hands-on Practice**: Building your first Graph Agent

### Part 2: Building Blocks - Core Components  
**Estimated Time**: 3-4 hours  
**Prerequisites**: Completion of Part 1, understanding of state machines  
**Objective**: Master advanced state management and workflow orchestration

📖 **[Theory: Building Blocks README](Part2_Building_Blocks/README.md)**  
💻 **[Practice: 02_Tutorial Notebook](Part2_Building_Blocks/02_Tutorial.ipynb)**

- **2.1 State Management**: TypedDict states and Annotated merging
- **2.2 LangGraph Workflows**: Complex graph-based workflow construction
- **2.3 Router Patterns**: Intelligent routing and load balancing
- **2.4 System Integration**: Combining components into cohesive systems

### Part 3: Agent Architectures - Advanced Patterns
**Estimated Time**: 4-5 hours  
**Prerequisites**: Completion of Parts 1-2, understanding of agent patterns  
**Objective**: Implement ReAct patterns and multi-agent collaboration

📖 **[Theory: Agent Architectures README](Part3_Agent_Architectures/README.md)**  
💻 **[Practice: 03_Tutorial Notebook](Part3_Agent_Architectures/03_Tutorial.ipynb)**

- **3.1 ReAct Pattern**: Reasoning-Acting loops for intelligent behavior
- **3.2 Multi-Agent Teams**: Specialized agent collaboration systems
- **3.3 Tool Integration**: External tool usage and API integration
- **3.4 Enterprise Architecture**: Production-ready agent systems

### Part 4: PHM Integration - Real Components
**Estimated Time**: 5-6 hours  
**Prerequisites**: Completion of Parts 1-3, signal processing basics  
**Objective**: Integrate with real PHM components and data

📖 **[Theory: PHM Integration README](Part4_PHM_Integration/README.md)**  
💻 **[Practice: 04_Tutorial Notebook](Part4_PHM_Integration/04_Tutorial.ipynb)**

- **4.1 PHM Framework**: Understanding predictive health management
- **4.2 Signal Processing**: Real-world sensor data analysis  
- **4.3 Fault Diagnosis**: Automated bearing fault detection
- **4.4 System Monitoring**: Production health monitoring systems

### Part 5: Complete PHMGA - Production Deployment
**Estimated Time**: 4-5 hours  
**Prerequisites**: Completion of Parts 1-4, production deployment knowledge  
**Objective**: Deploy complete PHMGA systems with optimization

📖 **[Theory: Complete System README](Part5_PHMGA_Complete/README.md)**  
💻 **[Practice: 05_Tutorial Notebook](Part5_PHMGA_Complete/05_Tutorial.ipynb)**

- **5.1 System Integration**: Complete end-to-end system assembly
- **5.2 Real Case Studies**: Actual industrial applications  
- **5.3 Performance Optimization**: System tuning and scaling
- **5.4 Production Deployment**: Monitoring, maintenance, and operations

## 🎯 Learning Objectives

By completing this tutorial series, you will be able to:

1. **Master Graph Agent Architecture**: Understand dual-layer design with LangGraph workflows and dynamic signal processing
2. **Implement Multi-LLM Systems**: Build agents that seamlessly switch between Google Gemini, OpenAI GPT, and other providers
3. **Design ReAct Patterns**: Create reasoning-acting loops for intelligent autonomous behavior
4. **Build Multi-Agent Teams**: Orchestrate specialized agents for complex collaborative tasks
5. **Integrate PHM Components**: Connect with real signal processing and predictive maintenance systems
6. **Deploy Production Systems**: Set up monitoring, scaling, optimization, and enterprise-grade maintenance

## 🛠️ Setup Instructions

### Prerequisites Installation

```bash
# Clone the repository
git clone https://github.com/PHMbench/PHMGA.git
cd PHMGA

# Create virtual environment
python -m venv tutorial_env
source tutorial_env/bin/activate  # On Windows: tutorial_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install jupyter notebook  # For interactive tutorials
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Multi-LLM Provider Support (at least one required)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DASHSCOPE_API_KEY=your_tongyi_api_key_here  # Optional: 通义千问
ZHIPUAI_API_KEY=your_zhipu_api_key_here     # Optional: 智谱GLM

# LLM Configuration
LLM_PROVIDER=google  # or openai, tongyi, zhipu
LLM_MODEL=gemini-2.5-pro  # or gpt-4o, qwen-max, glm-4

# Optional: Disable LangSmith for cleaner logs
LANGCHAIN_TRACING_V2=false
```

### Verify Installation

```python
# Run this in Python to verify setup
from src.tools.signal_processing_schemas import OP_REGISTRY
from src.model import get_llm
from src.utils import initialize_state

print(f"Available signal processing operators: {len(OP_REGISTRY)}")
print(f"LLM provider configured: {get_llm()}")
print("✅ PHMGA Graph Agent system ready!")

# Should output: Available operators: 60+ (depending on implementation)
```

## 📖 How to Use This Tutorial

### 🎯 Modular Learning Approach

Each part follows a **Theory + Practice** structure:
- **📖 Theory (README.md)**: Comprehensive concepts, architecture, and design patterns
- **💻 Practice (Notebook)**: Concise hands-on implementation and experimentation

### For Beginners
1. **Start with [00_START_HERE.ipynb](00_START_HERE.ipynb)** for environment setup
2. For each part: **Read Theory first, then Practice**
3. Work through Parts 1-5 sequentially - each builds on the previous
4. Complete all practical exercises before moving forward
5. Join community discussions for help and clarification

### For Intermediate Users  
1. **Quick review**: Skim theory sections for familiar concepts
2. **Deep dive**: Focus on Graph Agent and LangGraph specifics in Parts 2-3
3. **Hands-on**: Spend more time with practical notebooks and experimentation
4. **Extend**: Try the extension challenges and contribute improvements

### For Advanced Users
1. **Targeted learning**: Jump to specific parts based on your needs
2. **Focus areas**: Parts 4-5 for production deployment and optimization
3. **Contribute**: Add new operators, improve documentation, mentor others
4. **Production**: Use Part 5 as a template for real-world deployments

## 🎓 Assessment and Certification

### Self-Assessment Rubrics

Each tutorial part includes:
- **Knowledge Checks**: Quick quizzes to test understanding
- **Practical Exercises**: Hands-on coding challenges
- **Project Assignments**: Comprehensive implementation tasks
- **Peer Review**: Community-based code review process

### Completion Criteria

To complete each part:
- [ ] Pass all knowledge checks (80% minimum)
- [ ] Complete all practical exercises
- [ ] Submit one project assignment
- [ ] Participate in peer review (give and receive feedback)

## 🤝 Community and Support

### Getting Help
- **GitHub Discussions**: Ask questions and share insights
- **Discord Channel**: Real-time chat with other learners
- **Office Hours**: Weekly Q&A sessions with maintainers
- **Study Groups**: Form local or virtual study groups

### Contributing Back
- **Bug Reports**: Help improve the tutorials
- **Content Contributions**: Add examples, exercises, or explanations
- **Translation**: Help make tutorials accessible in other languages
- **Mentoring**: Help guide new learners through the material

## 📊 Progress Tracking

Use this checklist to track your progress:

### Part 1: Foundations - Graph Agent Basics
- [ ] 1.1 Graph Agent Concepts & Architecture
- [ ] 1.2 LLM Provider Integration (Multi-provider support)  
- [ ] 1.3 Agent State Management Basics
- [ ] 1.4 Practical Implementation & Exercises

### Part 2: Building Blocks - Core Components
- [ ] 2.1 Advanced State Management (TypedDict & Annotated)
- [ ] 2.2 LangGraph Workflows & Complex Graphs
- [ ] 2.3 Router Patterns & Intelligent Routing
- [ ] 2.4 System Integration & Component Composition

### Part 3: Agent Architectures - Advanced Patterns
- [ ] 3.1 ReAct Pattern Implementation
- [ ] 3.2 Multi-Agent Team Collaboration  
- [ ] 3.3 Tool Integration & External APIs
- [ ] 3.4 Enterprise Architecture & Scalability

### Part 4: PHM Integration - Real Components
- [ ] 4.1 PHM Framework Understanding
- [ ] 4.2 Signal Processing & Sensor Data
- [ ] 4.3 Fault Diagnosis & Classification
- [ ] 4.4 Production Health Monitoring

### Part 5: Complete PHMGA - Production Deployment  
- [ ] 5.1 End-to-End System Integration
- [ ] 5.2 Real Case Studies & Applications
- [ ] 5.3 Performance Optimization & Tuning
- [ ] 5.4 Production Deployment & Operations

## 🔗 Additional Resources

### Documentation
- [Framework API Reference](../README.md#api-reference)
- [Signal Processing Operators](../README.md#signal-processing-operators)
- [Troubleshooting Guide](../README.md#troubleshooting)

### Research Papers
- "Graph Agents for Signal Processing Optimization"
- "LLM-Enhanced Graph Agent Systems"
- "Autonomous Health Monitoring Systems"

### External Learning
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Agents Guide](https://python.langchain.com/docs/modules/agents/)
- [Multi-Agent Systems Research](https://arxiv.org/abs/2308.04026)
- [ReAct Pattern Paper](https://arxiv.org/abs/2210.03629)

---

## 🚀 Quick Start

**Ready to begin your Graph Agent journey?** 

1. **🔧 [Start Here - Environment Setup](00_START_HERE.ipynb)** - Configure your environment and check dependencies
2. **📖 [Part 1 Theory - Foundations](Part1_Foundations/README.md)** - Learn Graph Agent concepts  
3. **💻 [Part 1 Practice - Hands-on](Part1_Foundations/01_Tutorial.ipynb)** - Build your first agent

**Need Help?** Check our [community discussions](https://github.com/PHMbench/PHMGA/discussions) or review the comprehensive documentation in each part's README.
