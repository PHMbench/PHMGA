#!/usr/bin/env python3
"""
PHMGA Setup Verification Script
Run this script to verify your environment is correctly configured.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
    """Check if key dependencies can be imported"""
    deps = [
        ('yaml', 'PyYAML'),
        ('numpy', 'numpy'), 
        ('pandas', 'pandas'),
        ('langgraph', 'langgraph'),
        ('langchain_core', 'langchain-core'),
        ('pydantic', 'pydantic')
    ]
    
    missing = []
    for module, package in deps:
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_environment():
    """Check environment variables"""
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for Google/Gemini API key
        google_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if google_key and google_key != "your_gemini_api_key_here":
            print("✅ Google/Gemini API key configured")
            return True
        else:
            print("⚠️  No valid Google/Gemini API key found in .env (OpenAI not supported in NVTA branch)")
            return False
    else:
        print("⚠️  .env file not found (copy from .env.example)")
        return False

def check_config_files():
    """Check configuration files exist"""
    configs = ['case_exp2.yaml', 'case_exp2.5.yaml', 'case_exp_ottawa.yaml']
    all_exist = True
    
    for config in configs:
        config_path = Path('config') / config
        if config_path.exists():
            print(f"✅ {config}")
        else:
            print(f"❌ {config} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks"""
    print("🚀 PHMGA Setup Verification")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", lambda: check_dependencies()[0]),
        ("Environment", check_environment),
        ("Configuration Files", check_config_files)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}:")
        try:
            passed = check_func()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"❌ Error during {check_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    
    if all_passed:
        print("🎉 All checks passed! You can run the demo:")
        print("   python main.py case_exp_ottawa")
        print("   python src/cases/case1.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n💡 Quick fixes:")
        print("   pip install -r requirements.txt")
        print("   cp .env.example .env  # Then edit .env with your API keys")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)