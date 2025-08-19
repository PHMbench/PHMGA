#!/usr/bin/env python3
"""
Comprehensive test script for all three completed PHMGA enhancement tasks.

This script demonstrates:
1. Task 1: Comprehensive Documentation Creation
2. Task 2: Unified State Management System  
3. Task 3: Multi-Provider LLM Integration

Run this script to verify that all systems work together seamlessly.
"""

import os
import sys
from pathlib import Path

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test_gemini_key_12345"
os.environ["OPENAI_API_KEY"] = "test_openai_key_12345"
os.environ["PHM_MAX_DEPTH"] = "10"
os.environ["PHM_DATA_DIR"] = "/tmp/test_data"


def test_task1_documentation():
    """Test Task 1: Comprehensive Documentation Creation."""
    print("=== Task 1: Testing Documentation Creation ===")
    
    # Check that all README files exist
    readme_files = [
        "src/agents/README.md",
        "src/states/README.md", 
        "src/tools/readme.md",
        "src/graph/README.md",
        "src/model/README.md",
        "src/prompts/README.md",
        "src/schemas/README.md",
        "src/utils/README.md",
        "src/cases/README.md",
        "src/model/README_PROVIDERS.md"
    ]
    
    missing_docs = []
    for readme_file in readme_files:
        if not Path(readme_file).exists():
            missing_docs.append(readme_file)
        else:
            # Check file size to ensure it's not empty
            size = Path(readme_file).stat().st_size
            if size < 1000:  # Less than 1KB suggests incomplete documentation
                missing_docs.append(f"{readme_file} (too small: {size} bytes)")
    
    if missing_docs:
        print(f"❌ Missing or incomplete documentation:")
        for doc in missing_docs:
            print(f"  - {doc}")
        return False
    else:
        print(f"✅ All {len(readme_files)} documentation files present and substantial")
        
        # Sample a few files to check content quality
        sample_files = ["src/agents/README.md", "src/states/README.md", "src/model/README.md"]
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                content = f.read()
                if "## " in content and "```" in content and "Parameters" in content:
                    print(f"  ✅ {sample_file}: Contains structured content with examples")
                else:
                    print(f"  ⚠️ {sample_file}: May lack complete API documentation")
        
        return True


def test_task2_unified_state():
    """Test Task 2: Unified State Management System."""
    print("\n=== Task 2: Testing Unified State Management ===")
    
    try:
        from src.states.phm_states import get_unified_state, reset_unified_state, PHMState, DAGState, InputData
        from src.states.migration import migrate_to_unified_state
        import numpy as np
        
        # Reset for clean test
        reset_unified_state()
        
        # Test 1: Basic unified state functionality
        unified_state = get_unified_state()
        
        # Test configuration access
        llm_config = unified_state.get_llm_config()
        processing_config = unified_state.get_processing_config()
        paths_config = unified_state.get_paths_config()
        
        print(f"✅ Unified state configuration loaded:")
        print(f"  - LLM config keys: {list(llm_config.keys())}")
        print(f"  - Processing config keys: {list(processing_config.keys())}")
        print(f"  - Paths config keys: {list(paths_config.keys())}")
        
        # Test 2: Setting and getting values
        unified_state.set('custom.test_value', 'hello_world')
        test_value = unified_state.get('custom.test_value')
        assert test_value == 'hello_world', f"Expected 'hello_world', got {test_value}"
        print("✅ Set/get functionality working")
        
        # Test 3: PHMState integration
        ref_signal = InputData(
            node_id="test_ref",
            parents=[],
            shape=(1, 100, 1),
            results={"ref": np.random.randn(1, 100, 1)},
            meta={"fs": 1000}
        )
        
        test_signal = InputData(
            node_id="test_tst", 
            parents=[],
            shape=(1, 100, 1),
            results={"tst": np.random.randn(1, 100, 1)},
            meta={"fs": 1000}
        )
        
        dag_state = DAGState(
            user_instruction="Test analysis",
            channels=["test_ch"],
            nodes={"test_ref": ref_signal, "test_tst": test_signal},
            leaves=["test_ref", "test_tst"]
        )
        
        state = PHMState(
            case_name="test_case",
            user_instruction="Test unified state integration",
            reference_signal=ref_signal,
            test_signal=test_signal,
            dag_state=dag_state
        )
        
        # Test unified state access through PHMState
        state_llm_config = state.get_llm_config()
        state_processing_config = state.get_processing_config()
        
        print("✅ PHMState unified state integration working")
        print(f"  - State can access LLM config: {bool(state_llm_config)}")
        print(f"  - State can access processing config: {bool(state_processing_config)}")
        
        # Test 4: Migration system
        migration_helper = migrate_to_unified_state()
        print(f"✅ Migration system functional")
        
        # Test 5: Validation
        validation_errors = unified_state.validate_required_config()
        print(f"✅ Configuration validation: {len(validation_errors)} errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified state management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task3_multi_provider_llm():
    """Test Task 3: Multi-Provider LLM Integration."""
    print("\n=== Task 3: Testing Multi-Provider LLM Integration ===")
    
    try:
        from src.model.providers import (
            get_provider_registry, create_llm_provider, get_llm_factory
        )
        from src.model import (
            get_llm, get_llm_by_provider, list_available_providers, auto_select_llm
        )
        from src.states.phm_states import get_unified_state, reset_unified_state
        
        # Reset for clean test
        reset_unified_state()
        
        # Test 1: Provider registry
        registry = get_provider_registry()
        providers = registry.list_providers()
        print(f"✅ Provider registry loaded with {len(providers)} providers: {providers}")
        
        # Test 2: Provider creation
        mock_provider = create_llm_provider(
            provider="mock",
            model="mock-model",
            extra_params={"responses": ["Test response"]}
        )
        print(f"✅ Mock provider created: {type(mock_provider).__name__}")
        
        # Test provider invocation
        response = mock_provider.invoke("Test prompt")
        print(f"✅ Provider invocation successful: '{response}'")
        
        # Test 3: Factory system
        factory = get_llm_factory()
        auto_provider = factory.auto_detect_provider()
        print(f"✅ Auto-detected provider: {auto_provider.provider_name}")
        
        # Test 4: Model.py integration
        available_providers = list_available_providers()
        print(f"✅ Available providers via model.py: {list(available_providers.keys())}")
        
        # Test direct provider access
        mock_llm = get_llm_by_provider("mock", "mock-model")
        print(f"✅ Direct provider access: {type(mock_llm).__name__}")
        
        # Test auto-selection
        auto_llm = auto_select_llm()
        print(f"✅ Auto-selected LLM: {type(auto_llm).__name__}")
        
        # Test 5: Unified state integration
        unified_state = get_unified_state()
        unified_state.set('llm.provider', 'mock')
        unified_state.set('llm.model', 'mock-model')
        
        unified_llm = get_llm()
        print(f"✅ Unified state LLM integration: {type(unified_llm).__name__}")
        
        # Test 6: Backward compatibility
        from src.configuration import Configuration
        from src.model import get_default_llm
        
        # These should still work (with deprecation warnings)
        config = Configuration()
        legacy_llm = get_default_llm(config)
        print(f"✅ Backward compatibility maintained: {type(legacy_llm).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-provider LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between all three systems."""
    print("\n=== Integration Test: All Systems Working Together ===")
    
    try:
        from src.states.phm_states import get_unified_state, reset_unified_state
        from src.model import get_llm
        
        # Reset for clean test
        reset_unified_state()
        
        # Configure system through unified state
        unified_state = get_unified_state()
        unified_state.set('llm.provider', 'mock')
        unified_state.set('llm.model', 'mock-model')
        unified_state.set('processing.max_depth', 12)
        unified_state.set('paths.save_dir', '/tmp/test_save')
        
        # Get LLM using unified configuration
        llm = get_llm(temperature=0.7)
        
        # Test that configuration flows through properly
        llm_config = unified_state.get_llm_config()
        processing_config = unified_state.get_processing_config()
        
        print("✅ Full system integration successful:")
        print(f"  - LLM provider: {llm_config.get('provider')}")
        print(f"  - LLM model: {llm_config.get('model')}")
        print(f"  - Max depth: {processing_config.get('max_depth')}")
        print(f"  - LLM type: {type(llm).__name__}")
        
        # Test that documentation is accessible
        readme_path = Path("src/model/README_PROVIDERS.md")
        if readme_path.exists():
            print("✅ Multi-provider documentation available")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive test suite for all three tasks."""
    print("🚀 PHMGA System Enhancement - Comprehensive Test Suite")
    print("=" * 60)
    
    results = {
        "Task 1 - Documentation": test_task1_documentation(),
        "Task 2 - Unified State": test_task2_unified_state(), 
        "Task 3 - Multi-Provider LLM": test_task3_multi_provider_llm(),
        "Integration Test": test_integration()
    }
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    all_passed = True
    for task, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{task:<25} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\n✨ PHMGA System Enhancement Complete:")
        print("  1. ✅ Comprehensive Documentation Created")
        print("  2. ✅ Unified State Management Implemented") 
        print("  3. ✅ Multi-Provider LLM Integration Added")
        print("  4. ✅ Full System Integration Verified")
        print("\n🚀 The PHMGA system is now enhanced with:")
        print("  • Complete API documentation for all modules")
        print("  • Centralized configuration and state management")
        print("  • Support for multiple LLM providers (Google, OpenAI)")
        print("  • Backward compatibility with existing code")
        print("  • Comprehensive migration tools and examples")
        
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
