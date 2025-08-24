#!/usr/bin/env python3
"""
PHMGA Tutorial Environment Checker

This script checks the environment and dependencies for the PHMGA tutorial series.
Run this before starting the tutorials to ensure everything is properly configured.

Usage:
    python check_environment.py
    
    or from tutorials directory:
    python -m check_environment
"""

import sys
import os
from pathlib import Path
import importlib.util
import subprocess
from typing import Dict, List, Tuple, Optional

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    print_header("Python Version Check")
    
    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible (≥ 3.8)")
        return True
    else:
        print_error("Python 3.8 or higher is required")
        return False

def check_package_installed(package_name: str, import_name: Optional[str] = None) -> bool:
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def check_core_dependencies() -> Dict[str, bool]:
    """Check core Python dependencies"""
    print_header("Core Dependencies Check")
    
    dependencies = {
        # Core tutorial dependencies
        'pathlib': 'pathlib',
        'typing': 'typing',
        'json': 'json',
        'sys': 'sys',
        'os': 'os',
        
        # Scientific computing
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        
        # Jupyter
        'notebook': 'notebook',
        'jupyter': 'jupyter_core',
        'ipykernel': 'ipykernel',
        
        # Optional: LangChain ecosystem
        'langchain': 'langchain',
        'langchain-core': 'langchain_core',
        'langchain-community': 'langchain_community',
        'langgraph': 'langgraph',
        'langchain-google-genai': 'langchain_google_genai',
        'langchain-openai': 'langchain_openai',
    }
    
    results = {}
    required = ['pathlib', 'typing', 'json', 'sys', 'os', 'numpy', 'matplotlib']
    optional = ['scipy', 'notebook', 'jupyter', 'ipykernel']
    llm_related = ['langchain', 'langchain-core', 'langchain-community', 'langgraph', 
                   'langchain-google-genai', 'langchain-openai']
    
    print("\n📦 Required packages:")
    for package in required:
        import_name = dependencies[package]
        is_available = check_package_installed(package, import_name)
        results[package] = is_available
        
        if is_available:
            print_success(f"{package}: Available")
        else:
            print_error(f"{package}: Missing")
    
    print("\n📦 Optional packages:")
    for package in optional:
        import_name = dependencies[package]
        is_available = check_package_installed(package, import_name)
        results[package] = is_available
        
        if is_available:
            print_success(f"{package}: Available")
        else:
            print_warning(f"{package}: Not installed (optional)")
    
    print("\n🤖 LLM/LangChain packages:")
    llm_count = 0
    for package in llm_related:
        import_name = dependencies[package]
        is_available = check_package_installed(package, import_name)
        results[package] = is_available
        
        if is_available:
            print_success(f"{package}: Available")
            llm_count += 1
        else:
            print_warning(f"{package}: Not installed")
    
    if llm_count == 0:
        print_warning("No LangChain packages found - will use Mock providers")
    elif llm_count < len(llm_related):
        print_warning(f"Only {llm_count}/{len(llm_related)} LangChain packages installed")
    else:
        print_success("All LangChain packages are available")
    
    return results

def check_environment_variables() -> Dict[str, bool]:
    """Check for required environment variables"""
    print_header("Environment Variables Check")
    
    env_vars = {
        'OPENAI_API_KEY': 'OpenAI GPT models',
        'GEMINI_API_KEY': 'Google Gemini models', 
        'DASHSCOPE_API_KEY': '通义千问 (Tongyi Qwen) models',
        'ZHIPUAI_API_KEY': '智谱GLM models',
    }
    
    results = {}
    configured_count = 0
    
    print("🔑 LLM API Keys:")
    for var, description in env_vars.items():
        value = os.getenv(var)
        is_configured = bool(value and len(value.strip()) > 0)
        results[var] = is_configured
        
        if is_configured:
            print_success(f"{var}: Configured ({description})")
            configured_count += 1
        else:
            print_warning(f"{var}: Not set ({description})")
    
    print(f"\n📊 API Keys configured: {configured_count}/{len(env_vars)}")
    
    if configured_count == 0:
        print_warning("No API keys configured - tutorials will use Mock providers")
        print_info("You can still run all tutorials in demonstration mode")
    elif configured_count < len(env_vars):
        print_info(f"Partial API configuration - some providers will be available")
    else:
        print_success("All API keys configured - full functionality available")
    
    # Check LLM configuration
    llm_provider = os.getenv('LLM_PROVIDER', 'mock')
    llm_model = os.getenv('LLM_MODEL', 'mock-model')
    print(f"\n⚙️ LLM Configuration:")
    print_info(f"LLM_PROVIDER: {llm_provider}")
    print_info(f"LLM_MODEL: {llm_model}")
    
    return results

def check_project_structure() -> bool:
    """Check if PHMGA project structure is available"""
    print_header("PHMGA Project Structure Check")
    
    # Get project paths
    current_dir = Path.cwd()
    tutorials_dir = current_dir if current_dir.name == 'tutorials' else current_dir / 'tutorials'
    project_root = tutorials_dir.parent
    
    print_info(f"Current directory: {current_dir}")
    print_info(f"Tutorials directory: {tutorials_dir}")
    print_info(f"Project root: {project_root}")
    
    # Check tutorial structure
    tutorial_parts = [
        'Part1_Foundations',
        'Part2_Building_Blocks', 
        'Part3_Agent_Architectures',
        'Part4_PHM_Integration',
        'Part5_PHMGA_Complete'
    ]
    
    missing_parts = []
    print("\n📁 Tutorial Parts:")
    for part in tutorial_parts:
        part_path = tutorials_dir / part
        if part_path.exists():
            print_success(f"{part}: Found")
            
            # Check for key files
            readme_path = part_path / 'README.md'
            notebook_path = part_path / f"{part.split('_')[0].lower()}_Tutorial.ipynb"
            
            if part == 'Part1_Foundations':
                notebook_path = part_path / '01_Tutorial.ipynb'
            elif part == 'Part2_Building_Blocks':
                notebook_path = part_path / '02_Tutorial.ipynb' 
            elif part == 'Part3_Agent_Architectures':
                notebook_path = part_path / '03_Tutorial.ipynb'
            elif part == 'Part4_PHM_Integration':
                notebook_path = part_path / '04_Tutorial.ipynb'
            elif part == 'Part5_PHMGA_Complete':
                notebook_path = part_path / '05_Tutorial.ipynb'
                
            if readme_path.exists():
                print_success(f"  README.md: Found")
            else:
                print_warning(f"  README.md: Missing")
                
            if notebook_path.exists():
                print_success(f"  Tutorial notebook: Found")
            else:
                print_warning(f"  Tutorial notebook: Missing")
                
        else:
            print_error(f"{part}: Not found")
            missing_parts.append(part)
    
    # Check PHMGA project structure
    print("\n🏗️ PHMGA Project Structure:")
    phmga_dirs = ['src', 'src/agents', 'src/tools', 'src/utils', 'src/model']
    
    project_available = True
    for dir_path in phmga_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print_success(f"{dir_path}: Found")
        else:
            print_warning(f"{dir_path}: Not found (tutorials will use demo mode)")
            if dir_path == 'src':
                project_available = False
    
    if project_available:
        print_success("PHMGA project structure is available")
    else:
        print_warning("PHMGA project not found - tutorials will run in demonstration mode")
    
    return len(missing_parts) == 0

def check_jupyter_environment() -> bool:
    """Check Jupyter environment"""
    print_header("Jupyter Environment Check")
    
    # Check if running in Jupyter
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            print_success("Running in IPython/Jupyter environment")
            if 'IPKernelApp' in ipython.config:
                print_success("Running in Jupyter Notebook/Lab")
            else:
                print_info("Running in IPython terminal")
        else:
            print_info("Not running in IPython/Jupyter (command line execution)")
    except ImportError:
        print_warning("IPython not available")
    
    # Check Jupyter packages
    jupyter_packages = ['notebook', 'jupyter_core', 'ipykernel']
    available_packages = []
    
    for package in jupyter_packages:
        if check_package_installed(package):
            available_packages.append(package)
            print_success(f"{package}: Available")
        else:
            print_warning(f"{package}: Not installed")
    
    if len(available_packages) >= 2:
        print_success("Jupyter environment is functional")
        return True
    else:
        print_warning("Jupyter environment incomplete - some features may not work")
        return False

def generate_installation_commands(missing_deps: List[str]) -> str:
    """Generate installation commands for missing dependencies"""
    if not missing_deps:
        return ""
    
    commands = []
    
    # Group packages
    basic_packages = []
    langchain_packages = []
    
    for dep in missing_deps:
        if dep.startswith('langchain'):
            langchain_packages.append(dep)
        else:
            basic_packages.append(dep)
    
    if basic_packages:
        commands.append(f"pip install {' '.join(basic_packages)}")
    
    if langchain_packages:
        commands.append(f"pip install {' '.join(langchain_packages)}")
    
    return '\n'.join(commands)

def main():
    """Main environment check function"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("🚀 PHMGA Tutorial Environment Checker")
    print("=====================================")
    print("Checking your environment for PHMGA tutorial compatibility...")
    print(f"{Colors.END}")
    
    # Perform all checks
    python_ok = check_python_version()
    deps_results = check_core_dependencies()
    env_results = check_environment_variables()
    structure_ok = check_jupyter_environment()
    project_ok = check_project_structure()
    
    # Summary
    print_header("Environment Check Summary")
    
    # Count results
    total_deps = len(deps_results)
    available_deps = sum(deps_results.values())
    total_env = len(env_results)
    configured_env = sum(env_results.values())
    
    print(f"📊 Overall Status:")
    print(f"  Python Version: {'✅' if python_ok else '❌'}")
    print(f"  Dependencies: {available_deps}/{total_deps} available")
    print(f"  API Keys: {configured_env}/{total_env} configured")
    print(f"  Tutorial Structure: {'✅' if structure_ok else '❌'}")
    print(f"  Project Structure: {'✅' if project_ok else '⚠️'}")
    
    # Determine overall status
    critical_deps = ['pathlib', 'typing', 'json', 'sys', 'os', 'numpy']
    critical_available = all(deps_results.get(dep, False) for dep in critical_deps)
    
    if python_ok and critical_available and structure_ok:
        print_success("\n🎉 Environment is ready for PHMGA tutorials!")
        
        if configured_env == 0:
            print_warning("💡 Consider configuring API keys for full LLM functionality")
            print_info("   Tutorials will work in demonstration mode without API keys")
        elif configured_env < total_env:
            print_info("💡 Additional API keys available for more LLM providers")
        
        if not project_ok:
            print_info("💡 PHMGA project not found - tutorials will run in demo mode")
            
    elif python_ok and critical_available:
        print_warning("\n⚠️  Environment partially ready")
        print_info("   Basic functionality available, some features may be limited")
        
    else:
        print_error("\n❌ Environment has critical issues")
        print_error("   Please resolve the issues above before running tutorials")
        
        # Generate installation commands
        missing_deps = [dep for dep, available in deps_results.items() if not available]
        if missing_deps:
            install_commands = generate_installation_commands(missing_deps)
            if install_commands:
                print_info("\n📋 To install missing packages, run:")
                print(f"{Colors.YELLOW}{install_commands}{Colors.END}")
    
    # Getting started guide
    print_header("Getting Started")
    print("🚀 To start the tutorials:")
    print("   1. Open 00_START_HERE.ipynb")
    print("   2. Follow the learning path: Part1 → Part2 → Part3 → Part4 → Part5")
    print("   3. Read README.md in each part for theory")
    print("   4. Run the notebooks for hands-on practice")
    
    print("\n🆘 Need help?")
    print("   - Check tutorial READMEs for detailed documentation")
    print("   - Review error messages and stack traces")
    print("   - Visit project GitHub issues for support")
    
    return python_ok and critical_available and structure_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)