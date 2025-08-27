# Part 5 PHM Case 1 - Setup Guide

## 🎯 Overview

This tutorial demonstrates the complete PHMGA production system integration. The tutorial is designed to work in **two modes**:

1. **🏭 Production Mode**: Uses real components from `src/` directory

## 🚀 Quick Start

### Option 1: Demo Mode (Recommended for Learning)
```bash
cd tutorials_research/Part5_PHM_Case1/
jupyter notebook 05_Tutorial.ipynb
```

The tutorial will automatically detect missing production components and run in demo mode with full educational value.

### Option 2: Production Mode Setup
For full production integration, ensure you have:

1. **Complete PHMGA System**: The `src/` directory with all production components
2. **Dependencies**: All required Python packages
3. **API Keys**: Configured LLM provider credentials

## 🔧 System Requirements

### Python Dependencies
```bash
pip install langgraph langchain numpy matplotlib pandas jupyter ipywidgets
pip install networkx seaborn scipy pydantic
```

### Production Mode Additional Requirements
```bash
# LLM Provider packages (choose based on your provider)
pip install langchain-google-genai  # For Google Gemini
pip install langchain-openai        # For OpenAI
pip install langchain-anthropic     # For Claude
```

## 🎓 Understanding the Two Modes

### Demo Mode
**When it runs**: Automatically when production components are unavailable
**What you get**:
- ✅ Complete tutorial experience
- ✅ All educational concepts demonstrated
- ✅ Mock components show real patterns
- ✅ No external dependencies required
- ✅ Immediate learning value

**Limitations**:
- Mock workflows instead of real LangGraph execution
- Synthetic data instead of real signal processing
- Educational demonstrations instead of production analysis

### Production Mode
**When it runs**: When complete `src/` directory is available and configured
**What you get**:
- ✅ Real LangGraph workflows with actual agent coordination
- ✅ Production signal processing with 40+ operators
- ✅ Actual LLM-powered intelligent agents
- ✅ Real state management and DAG construction
- ✅ Industrial-grade error handling and monitoring

**Requirements**:
- Complete PHMGA system in `../../../src/`
- Configured LLM API keys
- All production dependencies

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
**Error**: `ImportError: attempted relative import beyond top-level package`
**Solution**: The tutorial will automatically fall back to demo mode. This is expected and educational.

#### 2. Missing src/ Directory
**Error**: Various import failures from src/
**Solution**: 
- **For Learning**: Continue in demo mode - full educational value maintained
- **For Production**: Clone/setup the complete PHMGA system

#### 3. LLM API Configuration
**Error**: LLM provider authentication failures
**Solution**: 
- **Demo Mode**: No API keys required, mock agents demonstrate concepts
- **Production Mode**: Configure API keys for your chosen provider

#### 4. Jupyter Widget Issues
**Error**: Interactive widgets not displaying
**Solution**: 
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## 📖 Educational Value in Both Modes

| Concept | Demo Mode | Production Mode |
|---------|-----------|-----------------|
| **LangGraph Workflows** | Mock workflows show structure | Real workflows with execution |
| **Agent Coordination** | Educational demonstrations | Actual LLM-powered agents |
| **Signal Processing** | Synthetic signals + mock operators | Real operators on actual data |
| **DAG Construction** | Simulated DAG building | Dynamic production DAG creation |
| **State Management** | Mock state transitions | Real PHMState management |
| **Error Handling** | Educational error scenarios | Production error recovery |

**🎓 Key Point**: All tutorial learning objectives are achieved in both modes!

## 🔄 Mode Detection

The system automatically detects which mode to run:

```python
# The tutorial checks these conditions:
1. src/ directory exists and accessible
2. Required production components importable  
3. LLM providers configured (for production)

# If any fail → Demo Mode
# If all succeed → Production Mode
```

## 📊 Mode Indicators

Look for these indicators in the tutorial output:

### Demo Mode Indicators
```
⚠️ Production mode unavailable: <reason>
🎓 Running in educational demo mode
✅ Demo LangGraph workflows initialized
🎓 Demo mode: Creating mock state
```

### Production Mode Indicators
```
✅ Production PHMGA components loaded
🏭 Production Mode: Full production component integration
✅ Production LangGraph workflows initialized
🏭 Production system initialization successful
```

## 🎯 Getting Maximum Educational Value

### In Demo Mode:
1. **Focus on Concepts**: Understand the architecture and patterns
2. **Read the Code**: Examine mock implementations to understand real patterns
3. **Follow the Workflow**: See how components interact
4. **Experiment**: Modify parameters and see educational responses

### In Production Mode:
1. **Experience Real System**: See actual LLM reasoning and agent coordination
2. **Monitor Performance**: Observe real execution times and resource usage
3. **Analyze Real Results**: Work with actual signal processing outcomes
4. **Scale Understanding**: Learn production deployment considerations

## 🚀 Next Steps

### After Completing the Tutorial:
1. **Understand Architecture**: Both modes teach the same architectural patterns
2. **Explore Components**: Dive deeper into specific agents or operators
3. **Apply Knowledge**: Use concepts in your own projects
4. **Contribute**: Help improve the educational materials

### For Production Deployment:
1. **Setup Complete System**: Get the full PHMGA production system
2. **Configure Infrastructure**: Set up monitoring, databases, etc.
3. **Integrate with Systems**: Connect to your industrial systems
4. **Scale and Monitor**: Deploy with production considerations

## 📞 Support

### If You Need Help:
1. **Check Console Output**: Mode indicators explain current status
2. **Read Error Messages**: Educational guidance provided for common issues
3. **Try Demo Mode**: Full educational value without production complexity
4. **Review This Guide**: Solutions for common scenarios

### Educational Questions:
- Focus on understanding concepts demonstrated in both modes
- Mock implementations show real production patterns
- All learning objectives achievable in demo mode

### Production Questions:
- Ensure complete PHMGA system setup
- Check LLM provider configurations
- Verify all dependencies installed

## 🎉 Success Metrics

You'll know the setup is working when:
- ✅ Notebook cells execute without critical errors
- ✅ Mode indicators clearly show which mode is active
- ✅ Educational content displays properly
- ✅ Visualizations and demonstrations work
- ✅ You understand the PHMGA system architecture

**Remember**: Both modes provide full educational value - the choice between them is about production integration, not learning capability!