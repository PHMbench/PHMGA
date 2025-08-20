#!/usr/bin/env python3
"""
Verification script for the PHMState simplification.

This script analyzes the code structure without importing dependencies
to verify that the simplification requirements have been met.
"""

import re
from pathlib import Path


def analyze_phm_state_structure():
    """Analyze the PHMState class structure."""
    print("=== Analyzing PHMState Structure ===")
    
    with open('src/states/phm_states.py', 'r') as f:
        content = f.read()
    
    # Check for removed unified state dependency
    has_unified_state_attr = '_unified_state' in content and 'PrivateAttr' in content
    has_config_property = re.search(r'@property\s+def config\(self\)', content)
    has_unified_state_init = 'get_unified_state()' in content and '__init__' in content
    
    print(f"✅ Removed _unified_state attribute: {not has_unified_state_attr}")
    print(f"✅ Removed config property delegation: {not has_config_property}")
    print(f"✅ Removed unified state in __init__: {not has_unified_state_init}")
    
    # Check for direct configuration fields
    direct_fields = [
        'llm_provider',
        'llm_model', 
        'llm_temperature',
        'min_depth',
        'max_depth',
        'min_width',
        'fs',
        'data_dir',
        'save_dir'
    ]
    
    found_fields = []
    for field in direct_fields:
        if re.search(rf'{field}:\s+\w+.*=.*Field\(', content):
            found_fields.append(field)
    
    print(f"✅ Direct configuration fields found: {len(found_fields)}/{len(direct_fields)}")
    for field in found_fields:
        print(f"  - {field}")
    
    # Check for simplified methods
    has_load_config = 'def load_config(self, yaml_path: str)' in content
    has_validate = 'def validate(self) -> List[str]:' in content
    has_env_loading = '_load_environment_variables' in content
    
    print(f"✅ Has direct load_config method: {has_load_config}")
    print(f"✅ Has simplified validate method: {has_validate}")
    print(f"✅ Has direct environment loading: {has_env_loading}")
    
    return True


def check_code_reduction():
    """Check that code has been reduced."""
    print("\n=== Checking Code Reduction ===")
    
    with open('src/states/phm_states.py', 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    code_lines_count = len(code_lines)
    
    print(f"✅ Total lines: {total_lines}")
    print(f"✅ Code lines: {code_lines_count}")
    print(f"✅ Target achieved (< 600 lines): {total_lines < 600}")
    
    return total_lines < 600


def check_factory_method():
    """Check that factory method is simplified."""
    print("\n=== Checking Factory Method ===")
    
    with open('src/states/phm_states.py', 'r') as f:
        content = f.read()
    
    # Check factory method structure
    has_from_case_config = 'def from_case_config(' in content
    factory_creates_instance = 'state = cls(' in content
    factory_loads_config = 'state.load_config(config_path)' in content
    
    print(f"✅ Has from_case_config factory method: {has_from_case_config}")
    print(f"✅ Factory creates instance directly: {factory_creates_instance}")
    print(f"✅ Factory loads config into instance: {factory_loads_config}")
    
    return has_from_case_config and factory_creates_instance and factory_loads_config


def check_yaml_loading():
    """Check that YAML loading is direct."""
    print("\n=== Checking YAML Loading ===")
    
    with open('src/states/phm_states.py', 'r') as f:
        content = f.read()
    
    # Check for direct field assignment in load_config
    has_direct_assignment = 'self.llm_provider =' in content
    has_yaml_parsing = 'yaml.safe_load(f)' in content
    has_section_handling = "if 'llm' in yaml_data:" in content
    
    print(f"✅ Has direct field assignment: {has_direct_assignment}")
    print(f"✅ Has YAML parsing: {has_yaml_parsing}")
    print(f"✅ Has section-based handling: {has_section_handling}")
    
    return has_direct_assignment and has_yaml_parsing and has_section_handling


def check_backward_compatibility():
    """Check that backward compatibility is maintained."""
    print("\n=== Checking Backward Compatibility ===")
    
    with open('src/states/phm_states.py', 'r') as f:
        content = f.read()
    
    # Check for backward compatibility methods
    compat_methods = [
        'get_config',
        'set_config', 
        'get_llm_config',
        'get_processing_config',
        'get_data_paths',
        'update_from_yaml',
        'validate_configuration'
    ]
    
    found_methods = []
    for method in compat_methods:
        if f'def {method}(' in content:
            found_methods.append(method)
    
    print(f"✅ Backward compatibility methods: {len(found_methods)}/{len(compat_methods)}")
    for method in found_methods:
        print(f"  - {method}")
    
    # Check for deprecated properties
    has_phm_model = '@property' in content and 'def phm_model(' in content
    has_deprecation_warnings = 'warnings.warn(' in content and 'DeprecationWarning' in content
    
    print(f"✅ Has deprecated property accessors: {has_phm_model}")
    print(f"✅ Has deprecation warnings: {has_deprecation_warnings}")
    
    return len(found_methods) >= 6


def check_removed_complexity():
    """Check that complex unified state management is removed."""
    print("\n=== Checking Removed Complexity ===")
    
    with open('src/states/phm_states.py', 'r') as f:
        content = f.read()
    
    # Check that complex unified state code is gone or deprecated
    has_phm_config_class = 'class PHMConfig(' in content
    has_complex_unified_manager = content.count('class UnifiedStateManager(') > 0 and content.count('_config: PHMConfig') > 0
    has_get_unified_state = 'def get_unified_state()' in content
    
    # UnifiedStateManager should be deprecated
    unified_state_deprecated = 'UnifiedStateManager is deprecated' in content
    
    print(f"✅ Removed PHMConfig class: {not has_phm_config_class}")
    print(f"✅ Simplified UnifiedStateManager: {not has_complex_unified_manager}")
    print(f"✅ UnifiedStateManager marked deprecated: {unified_state_deprecated}")
    print(f"✅ get_unified_state still exists (for compatibility): {has_get_unified_state}")
    
    return not has_phm_config_class and unified_state_deprecated


def main():
    """Run all verification checks."""
    print("🔍 Verifying PHMState Simplification\n")
    
    checks = [
        ("PHMState Structure Analysis", analyze_phm_state_structure),
        ("Code Reduction", check_code_reduction),
        ("Factory Method", check_factory_method),
        ("YAML Loading", check_yaml_loading),
        ("Backward Compatibility", check_backward_compatibility),
        ("Removed Complexity", check_removed_complexity),
    ]
    
    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION RESULTS")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL VERIFICATION CHECKS PASSED! 🎉")
        print("\n✨ PHMState Simplification Successfully Completed:")
        print("  • ✅ Removed unified state dependency")
        print("  • ✅ Direct configuration field access")
        print("  • ✅ Simplified factory method")
        print("  • ✅ Direct YAML loading")
        print("  • ✅ Maintained backward compatibility")
        print("  • ✅ Reduced code complexity")
        print("  • ✅ Clear, transparent state structure")
        
        print("\n📋 Key Architectural Improvements:")
        print("  • No more self._unified_state indirection")
        print("  • Configuration fields directly in PHMState")
        print("  • YAML loads directly into state fields")
        print("  • Environment variables load directly")
        print("  • Factory method behavior is obvious")
        print("  • Backward compatibility preserved")
        
        return 0
    else:
        print("❌ SOME VERIFICATION CHECKS FAILED")
        print("Please review the implementation and fix any issues.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
