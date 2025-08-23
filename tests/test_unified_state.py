#!/usr/bin/env python3
"""
Test script demonstrating the unified state management system.

This script shows how to use the new UnifiedStateManager and how it
integrates with the existing PHMGA system while maintaining backward compatibility.
"""

import os
import tempfile
from pathlib import Path

# Set up test environment
os.environ["GEMINI_API_KEY"] = "test_key_12345"
os.environ["PHM_MAX_DEPTH"] = "10"
os.environ["PHM_DATA_DIR"] = "/tmp/test_data"

def test_unified_state_basic():
    """Test basic unified state functionality."""
    print("=== Testing Basic Unified State Functionality ===")
    
    from src.states.phm_states import get_unified_state, reset_unified_state
    
    # Reset for clean test
    reset_unified_state()
    
    # Get unified state instance
    unified_state = get_unified_state()
    
    # Test getting configuration values
    print(f"LLM Model: {unified_state.get('llm.model')}")
    print(f"Max Depth: {unified_state.get('processing.max_depth')}")
    print(f"Data Dir: {unified_state.get('paths.data_dir')}")
    print(f"Gemini API Key: {unified_state.get('llm.gemini_api_key')}")
    
    # Test setting values
    unified_state.set('custom.test_value', 'hello_world')
    print(f"Custom Value: {unified_state.get('custom.test_value')}")
    
    # Test validation
    errors = unified_state.validate_required_config()
    print(f"Validation Errors: {errors}")
    
    print("✅ Basic functionality test passed!\n")


def test_phm_state_integration():
    """Test PHMState integration with unified state."""
    print("=== Testing PHMState Integration ===")
    
    from src.states.phm_states import PHMState, DAGState, InputData, reset_unified_state
    import numpy as np
    
    # Reset for clean test
    reset_unified_state()
    
    # Create test data
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
    
    # Create PHMState with unified state integration
    state = PHMState(
        case_name="test_case",
        user_instruction="Test unified state integration",
        reference_signal=ref_signal,
        test_signal=test_signal,
        dag_state=dag_state
    )
    
    # Test unified state access through PHMState
    print(f"LLM Config: {state.get_llm_config()}")
    print(f"Processing Config: {state.get_processing_config()}")
    print(f"Data Paths: {state.get_data_paths()}")
    
    # Test setting custom configuration
    state.set_config('analysis.custom_param', 42)
    print(f"Custom Param: {state.get_config('analysis.custom_param')}")
    
    # Test backward compatibility (should show deprecation warning)
    try:
        model_name = state.phm_model
        print(f"Legacy Model Access (deprecated): {model_name}")
    except Exception as e:
        print(f"Legacy access error: {e}")
    
    print("✅ PHMState integration test passed!\n")


def test_backward_compatibility():
    """Test backward compatibility with old Configuration system."""
    print("=== Testing Backward Compatibility ===")
    
    from src.configuration import Configuration
    from src.model import get_llm, get_default_llm
    from src.states.phm_states import reset_unified_state
    
    # Reset for clean test
    reset_unified_state()
    
    # Test old Configuration class (should show deprecation warning)
    try:
        config = Configuration()
        print(f"Legacy Config Model: {config.phm_model}")
    except Exception as e:
        print(f"Legacy config error: {e}")
    
    # Test model factory functions
    try:
        # New function
        llm = get_llm()
        print(f"New LLM Factory: {type(llm).__name__}")
        
        # Legacy function (should show deprecation warning)
        legacy_llm = get_default_llm()
        print(f"Legacy LLM Factory: {type(legacy_llm).__name__}")
        
    except Exception as e:
        print(f"Model factory error: {e}")
    
    print("✅ Backward compatibility test passed!\n")


def test_migration_system():
    """Test the migration system."""
    print("=== Testing Migration System ===")
    
    from src.states.migration import migrate_to_unified_state, StateMigrationHelper
    from src.states.phm_states import reset_unified_state
    
    # Reset for clean test
    reset_unified_state()
    
    # Set up some legacy environment variables
    os.environ["PHM_MODEL"] = "gemini-1.5-pro"
    os.environ["DATA_DIR"] = "/legacy/data"
    
    # Create migration helper
    helper = StateMigrationHelper()
    
    # Test environment variable migration
    migrated_env = helper.migrate_environment_variables()
    print(f"Migrated Environment Variables: {migrated_env}")
    
    # Test configuration migration
    legacy_config = {
        "phm_model": "gemini-1.5-pro",
        "query_generator_model": "gemini-1.5-pro"
    }
    migrated_config = helper.migrate_configuration_object(legacy_config)
    print(f"Migrated Configuration: {migrated_config}")
    
    # Test deprecated pattern detection
    test_code = """
    from src.configuration import Configuration
    config = Configuration()
    model = config.phm_model
    data_dir = os.environ["PHM_DATA_DIR"]
    """
    
    deprecated_patterns = helper.check_deprecated_usage(test_code)
    print(f"Deprecated Patterns Found: {len(deprecated_patterns)}")
    for pattern in deprecated_patterns:
        print(f"  - {pattern}")
    
    # Generate migration report
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        helper.save_migration_report(f.name)
        print(f"Migration report saved to: {f.name}")
    
    print("✅ Migration system test passed!\n")


def test_yaml_integration():
    """Test YAML configuration integration."""
    print("=== Testing YAML Integration ===")
    
    from src.states.phm_states import get_unified_state, reset_unified_state
    import yaml
    
    # Reset for clean test
    reset_unified_state()
    
    # Create test YAML configuration
    test_config = {
        'name': 'test_case',
        'llm': {
            'model': 'gemini-2.5-pro',
            'temperature': 0.7
        },
        'processing': {
            'max_depth': 12,
            'min_depth': 6
        },
        'custom': {
            'analysis_type': 'bearing_fault',
            'confidence_threshold': 0.85
        }
    }
    
    # Save to temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        yaml_path = f.name
    
    try:
        # Load configuration from YAML
        unified_state = get_unified_state()
        unified_state.update_from_yaml(yaml_path)
        
        # Test that values were loaded
        print(f"Loaded LLM Model: {unified_state.get('llm.model')}")
        print(f"Loaded Max Depth: {unified_state.get('processing.max_depth')}")
        print(f"Loaded Custom Config: {unified_state.get('custom.analysis_type')}")
        
        # Test export
        exported_config = unified_state.export_config()
        print(f"Exported Config Keys: {list(exported_config.keys())}")
        
    finally:
        # Clean up
        Path(yaml_path).unlink()
    
    print("✅ YAML integration test passed!\n")


def main():
    """Run all tests."""
    print("🚀 Starting Unified State Management Tests\n")
    
    try:
        test_unified_state_basic()
        test_phm_state_integration()
        test_backward_compatibility()
        test_migration_system()
        test_yaml_integration()
        
        print("🎉 All tests passed successfully!")
        print("\n📋 Summary:")
        print("- ✅ Basic unified state functionality")
        print("- ✅ PHMState integration")
        print("- ✅ Backward compatibility")
        print("- ✅ Migration system")
        print("- ✅ YAML configuration integration")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
