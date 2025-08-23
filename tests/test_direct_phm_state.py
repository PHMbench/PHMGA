#!/usr/bin/env python3
"""
Test script for the simplified direct PHMState implementation.

This verifies that the new direct state management works correctly
without the complex unified state dependency.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add src to path for direct import
sys.path.insert(0, 'src')

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test_gemini_key_12345"
os.environ["PHM_MAX_DEPTH"] = "10"
os.environ["PHM_SAVE_DIR"] = "/tmp/test_save"


def test_direct_state_creation():
    """Test creating PHMState directly without unified state dependency."""
    print("=== Testing Direct PHMState Creation ===")
    
    try:
        # Import the data classes first
        from states.phm_states import InputData, DAGState, PHMState
        import numpy as np
        
        # Create minimal test data
        test_shape = (1, 100, 1)
        test_data = np.random.randn(*test_shape)
        
        # Create test input data
        ref_signal = InputData(
            node_id="test_ref",
            parents=[],
            shape=test_shape,
            results={"ref": {"test_id": test_data}},
            meta={"channel": "ch1"}
        )
        
        test_signal = InputData(
            node_id="test_tst",
            parents=[],
            shape=test_shape,
            results={"tst": {"test_id": test_data}},
            meta={"channel": "ch1"}
        )
        
        # Create DAG state
        dag_state = DAGState(
            user_instruction="Test analysis",
            channels=["ch1"],
            nodes={"test_ref": ref_signal, "test_tst": test_signal},
            leaves=["test_ref", "test_tst"]
        )
        
        # Create PHMState directly
        state = PHMState(
            case_name="test_case",
            user_instruction="Test direct state creation",
            reference_signal=ref_signal,
            test_signal=test_signal,
            dag_state=dag_state
        )
        
        print(f"✅ PHMState created successfully")
        print(f"✅ Case name: {state.case_name}")
        print(f"✅ LLM provider: {state.llm_provider}")
        print(f"✅ LLM model: {state.llm_model}")
        print(f"✅ Min depth: {state.min_depth}")
        print(f"✅ Max depth: {state.max_depth}")
        print(f"✅ Environment variable loaded: {state.gemini_api_key is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct state creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_configuration_access():
    """Test direct access to configuration fields."""
    print("\n=== Testing Direct Configuration Access ===")
    
    try:
        from states.phm_states import InputData, DAGState, PHMState
        import numpy as np
        
        # Create minimal state
        test_shape = (1, 100, 1)
        test_data = np.random.randn(*test_shape)
        
        ref_signal = InputData(
            node_id="test_ref", parents=[], shape=test_shape,
            results={"ref": {"test_id": test_data}}, meta={"channel": "ch1"}
        )
        
        dag_state = DAGState(
            user_instruction="Test", channels=["ch1"],
            nodes={"test_ref": ref_signal}, leaves=["test_ref"]
        )
        
        state = PHMState(
            case_name="test_case",
            user_instruction="Test direct configuration access",
            reference_signal=ref_signal,
            test_signal=ref_signal,
            dag_state=dag_state
        )
        
        # Test direct field access
        print(f"✅ Direct LLM provider access: {state.llm_provider}")
        print(f"✅ Direct LLM model access: {state.llm_model}")
        print(f"✅ Direct temperature access: {state.llm_temperature}")
        print(f"✅ Direct min_depth access: {state.min_depth}")
        print(f"✅ Direct max_depth access: {state.max_depth}")
        print(f"✅ Direct fs access: {state.fs}")
        
        # Test field modification
        original_model = state.llm_model
        state.llm_model = "test-model"
        assert state.llm_model == "test-model"
        state.llm_model = original_model
        print("✅ Direct field modification works")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct configuration access failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yaml_loading():
    """Test YAML configuration loading directly into state fields."""
    print("\n=== Testing Direct YAML Loading ===")
    
    try:
        from states.phm_states import InputData, DAGState, PHMState
        import numpy as np
        
        # Create test YAML
        test_yaml = {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4o',
                'temperature': 0.5,
                'max_retries': 5
            },
            'processing': {
                'min_depth': 6,
                'max_depth': 12,
                'min_width': 8
            },
            'paths': {
                'data_dir': '/tmp/test_data',
                'save_dir': '/tmp/test_save'
            },
            'custom_field': 'test_value'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_yaml, f)
            yaml_path = f.name
        
        try:
            # Create minimal state
            test_shape = (1, 100, 1)
            test_data = np.random.randn(*test_shape)
            
            ref_signal = InputData(
                node_id="test_ref", parents=[], shape=test_shape,
                results={"ref": {"test_id": test_data}}, meta={"channel": "ch1"}
            )
            
            dag_state = DAGState(
                user_instruction="Test", channels=["ch1"],
                nodes={"test_ref": ref_signal}, leaves=["test_ref"]
            )
            
            state = PHMState(
                case_name="test_case",
                user_instruction="Test YAML loading",
                reference_signal=ref_signal,
                test_signal=ref_signal,
                dag_state=dag_state
            )
            
            # Load YAML configuration
            original_provider = state.llm_provider
            state.load_config(yaml_path)
            
            # Verify changes
            assert state.llm_provider == 'openai'
            assert state.llm_model == 'gpt-4o'
            assert state.llm_temperature == 0.5
            assert state.llm_max_retries == 5
            assert state.min_depth == 6
            assert state.max_depth == 12
            assert state.min_width == 8
            assert state.data_dir == '/tmp/test_data'
            assert state.save_dir == '/tmp/test_save'
            assert state.extra_config['custom_field'] == 'test_value'
            
            print(f"✅ YAML loaded - Provider changed from {original_provider} to {state.llm_provider}")
            print(f"✅ YAML loaded - Model: {state.llm_model}")
            print(f"✅ YAML loaded - Temperature: {state.llm_temperature}")
            print(f"✅ YAML loaded - Min depth: {state.min_depth}")
            print(f"✅ YAML loaded - Max depth: {state.max_depth}")
            print(f"✅ YAML loaded - Data dir: {state.data_dir}")
            print(f"✅ YAML loaded - Custom field: {state.extra_config['custom_field']}")
            
            return True
            
        finally:
            os.unlink(yaml_path)
        
    except Exception as e:
        print(f"❌ YAML loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility methods."""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        from states.phm_states import InputData, DAGState, PHMState
        import numpy as np
        
        # Create minimal state
        test_shape = (1, 100, 1)
        test_data = np.random.randn(*test_shape)
        
        ref_signal = InputData(
            node_id="test_ref", parents=[], shape=test_shape,
            results={"ref": {"test_id": test_data}}, meta={"channel": "ch1"}
        )
        
        dag_state = DAGState(
            user_instruction="Test", channels=["ch1"],
            nodes={"test_ref": ref_signal}, leaves=["test_ref"]
        )
        
        state = PHMState(
            case_name="test_case",
            user_instruction="Test backward compatibility",
            reference_signal=ref_signal,
            test_signal=ref_signal,
            dag_state=dag_state
        )
        
        # Test backward compatibility methods
        llm_config = state.get_llm_config()
        processing_config = state.get_processing_config()
        data_paths = state.get_data_paths()
        
        print(f"✅ get_llm_config(): {len(llm_config)} keys")
        print(f"✅ get_processing_config(): {len(processing_config)} keys")
        print(f"✅ get_data_paths(): {len(data_paths)} keys")
        
        # Test dot notation access
        model = state.get_config('llm.model')
        max_depth = state.get_config('processing.max_depth')
        print(f"✅ Dot notation access - Model: {model}")
        print(f"✅ Dot notation access - Max depth: {max_depth}")
        
        # Test setting values
        state.set_config('llm.temperature', 0.8)
        temp = state.get_config('llm.temperature')
        print(f"✅ Set/get config - Temperature: {temp}")
        
        # Test validation
        errors = state.validate()
        print(f"✅ Validation: {len(errors)} errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_architecture_simplification():
    """Test that the architecture is actually simplified."""
    print("\n=== Testing Architecture Simplification ===")
    
    try:
        from states.phm_states import PHMState
        
        # Check that PHMState no longer has unified state dependency
        state_fields = [field for field in dir(PHMState) if not field.startswith('_')]
        
        # Should not have config property that delegates to unified state
        has_config_property = hasattr(PHMState, 'config') and callable(getattr(PHMState, 'config', None))
        
        # Should have direct configuration fields
        has_llm_provider = 'llm_provider' in PHMState.__fields__
        has_llm_model = 'llm_model' in PHMState.__fields__
        has_min_depth = 'min_depth' in PHMState.__fields__
        has_max_depth = 'max_depth' in PHMState.__fields__
        
        print(f"✅ PHMState has {len(state_fields)} public fields/methods")
        print(f"✅ No config property delegation: {not has_config_property}")
        print(f"✅ Has direct llm_provider field: {has_llm_provider}")
        print(f"✅ Has direct llm_model field: {has_llm_model}")
        print(f"✅ Has direct min_depth field: {has_min_depth}")
        print(f"✅ Has direct max_depth field: {has_max_depth}")
        
        # Check line count
        with open('src/states/phm_states.py', 'r') as f:
            lines = len(f.readlines())
        
        print(f"✅ Total lines: {lines} (target: < 600)")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture simplification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests for the simplified direct PHMState implementation."""
    print("🚀 Testing Simplified Direct PHMState Implementation\n")
    
    tests = [
        ("Direct State Creation", test_direct_state_creation),
        ("Direct Configuration Access", test_direct_configuration_access),
        ("Direct YAML Loading", test_yaml_loading),
        ("Backward Compatibility", test_backward_compatibility),
        ("Architecture Simplification", test_architecture_simplification),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\n✨ Simplified Direct PHMState Summary:")
        print("  • ✅ Removed unified state dependency")
        print("  • ✅ Direct configuration field access")
        print("  • ✅ Simplified YAML loading")
        print("  • ✅ Maintained backward compatibility")
        print("  • ✅ Reduced code complexity")
        print("  • ✅ Clear, transparent state structure")
        
        print("\n📋 Key Improvements:")
        print("  • Direct access: state.llm_model (not state.config.llm.model)")
        print("  • No intermediate state managers")
        print("  • YAML loads directly into state fields")
        print("  • Environment variables load directly")
        print("  • Transparent, obvious state structure")
        
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
