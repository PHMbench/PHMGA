# PHMGA - PHM Graph Agent Demo

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-green.svg)](https://langchain-ai.github.io/langgraph/)

A **minimal demo** of the PHM Graph Agent system - an intelligent multi-agent framework for bearing fault diagnosis using LangGraph and dynamic signal processing DAGs.

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
# Clone the repository (NVTA_2025_Version branch)
git clone -b NVTA_2025_Version https://github.com/PHMbench/PHMGA.git
cd PHMGA

# Install requirements  
pip install -r requirements.txt
```

### 2. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Google API key:
GOOGLE_API_KEY="your_gemini_api_key" 
USE_REAL_LLM="1"
```

### 3. **Run Demo Cases**
```bash
# Run by case config name under config/
python main.py case1
python main.py case_exp2
python main.py case_exp2.5
python main.py case_exp_ottawa
```

## 🏗️ System Architecture

**Dual-Layer Design:**
- **Outer Layer**: LangGraph workflow orchestrating multi-agent system
- **Inner Layer**: Dynamic signal processing DAG construction

**Agent Workflow:**
```
with_report:
START → Plan → Execute → Reflect → (Plan | Inquire) → Prepare → Train → Report → END

builder:
START → Plan → Execute → Reflect → (Plan | END)

executor:
START → Inquire → Prepare → Train → Report → END
```

## 📊 Demo Cases

| Case | Description | Fault States | Dataset Type |
|------|-------------|--------------|--------------|
| **case_exp2** | Standard bearing diagnosis | 5 states (health, ball, cage, inner, outer) | Fixed speed |
| **case_exp2.5** | Alternative configuration | 5 states (health, ball, cage, inner, outer) | Fixed speed |
| **case_exp_ottawa** | Variable speed analysis | 3 states (health, inner, outer) | Variable speed |

## 🎯 What This Demo Shows

1. **Multi-Agent Planning**: LLM-powered decomposition of analysis tasks
2. **Dynamic DAG Construction**: Automatic signal processing pipeline creation
3. **Iterative Refinement**: Plan-Execute-Reflect loops for optimal results
4. **Signal Processing**: 60+ operators for time-frequency analysis
5. **Fault Classification**: Intelligent bearing condition assessment

## 📁 Project Structure

```
PHMGA/
├── config/              # Case configuration files
│   ├── case_exp2.yaml
│   ├── case_exp2.5.yaml
│   └── case_exp_ottawa.yaml
├── src/
│   ├── agents/          # Multi-agent system
│   ├── cases/           # Demo implementations
│   ├── states/          # State management
│   ├── tools/           # Signal processing operators
│   └── utils/           # Utilities
├── main.py             # Main entry point
├── requirements.txt    # Minimal dependencies
└── CLAUDE.md          # Developer guidance
```

## 🔧 Configuration

Each case uses a YAML configuration file with:
- **Data paths**: Metadata and signal data locations
- **Signal IDs**: Reference and test signal identifiers  
- **User instruction**: Natural language task description
- **Builder settings**: DAG depth and graph selection parameters

**Example configuration:**
```yaml
name: "case_exp2"
builder:
  graph: with_report
  min_depth: 4
  max_depth: 8
user_instruction: >
  Analyze bearing signals for potential faults. The reference set contains
  signals for 5 different states. Classify each test signal accordingly.
ref_ids: [47050, 47052, 47044, 47046, 47047]
test_ids: [47051, 47045, 47048, 47054, 47057]
```

## ⚠️ Demo Notes

- **Data Files**: Demo handles missing data files gracefully with warnings
- **API Keys**: Requires valid Google Gemini API key only (OpenAI not supported in this branch)
- **Processing**: Creates DAG visualizations and saves results to `save/` directory
- **Mock Mode**: Falls back to sample data if real datasets unavailable
- **Dataset Access**: Visit [PHMbench homepage](https://github.com/PHMbench/) for dataset download instructions

## 🧪 Expected Output

```json
{
  "status": "ok",
  "case_name": "case_exp_ottawa",
  "graph": "with_report",
  "state_save_path": ".../exp2.5_built_state_ottawa.pkl",
  "report_path": ".../exp2.5_final_report_ottawa.md",
  "dag_depth": 4,
  "dag_nodes": 12,
  "has_final_report": true
}
```

## 🤝 Support

For questions or issues:
1. Check the CLAUDE.md for developer guidance
2. Verify your API keys are correctly configured
3. Ensure Python 3.8+ and dependencies are installed

---

**Note**: This is a research prototype demonstrating graph-based multi-agent systems for PHM applications.
