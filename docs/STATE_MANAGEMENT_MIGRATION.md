# PHM State Management Migration Guide

This guide shows how to migrate from the old complex state management system to the new simplified system.

## Overview of Changes

### Before (Complex System - 730+ lines)
- Complex `UnifiedStateManager` with multiple data structures
- String-based dot-notation configuration access
- Complex initialization chain with `_apply_unified_config()`
- Multiple redundant configuration accessors
- 25+ field definitions in PHMState

### After (Simplified System - 636 lines)
- Typed `PHMConfig` with structured access
- Factory methods for easy initialization
- Direct attribute access: `state.config.llm.model`
- Single comprehensive validation
- Streamlined PHMState with only essential fields

## Migration Examples

### 1. State Initialization

**Before:**
```python
from src.utils import initialize_state

# Complex parameter passing
state = initialize_state(
    user_instruction=config['user_instruction'],
    metadata_path=config['metadata_path'],
    h5_path=config['h5_path'],
    ref_ids=config['ref_ids'],
    test_ids=config['test_ids'],
    case_name=config['name']
)
```

**After:**
```python
from src.states.phm_states import PHMState

# Simplified factory method
state = PHMState.from_case_config(
    config_path="config/case1.yaml",
    case_name=config['name'],
    user_instruction=config['user_instruction'],
    metadata_path=config['metadata_path'],
    h5_path=config['h5_path'],
    ref_ids=config['ref_ids'],
    test_ids=config['test_ids']
)
```

### 2. Configuration Access

**Before:**
```python
# String-based dot notation
model = state.get_config('llm.model')
max_depth = state.get_config('processing.max_depth')
api_key = state.get_config('llm.gemini_api_key')

# Multiple accessor methods
llm_config = state.get_llm_config()
processing_config = state.get_processing_config()
paths_config = state.get_data_paths()
```

**After:**
```python
# Typed attribute access
model = state.config.llm.model
max_depth = state.config.processing.max_depth
api_key = state.config.llm.gemini_api_key

# Still available for backward compatibility
llm_config = state.get_llm_config()
processing_config = state.get_processing_config()
paths_config = state.get_data_paths()
```

### 3. YAML Configuration Loading

**Before:**
```python
# Complex update mechanism
state.update_from_yaml("config/case1.yaml")
state._apply_unified_config()
```

**After:**
```python
# Direct loading
state.load_config("config/case1.yaml")

# Or during initialization
state = PHMState.from_case_config(
    config_path="config/case1.yaml",
    # ... other parameters
)
```

### 4. Validation

**Before:**
```python
# Multiple validation methods
config_errors = state.validate_config()
required_errors = state.validate_required_config()
data_errors = state.validate_data_integrity()
```

**After:**
```python
# Single comprehensive validation
errors = state.validate()
if errors:
    print("Validation errors:", errors)
```

### 5. Environment Variable Handling

**Before:**
```python
# Complex environment variable mapping
unified_state = get_unified_state()
unified_state._load_environment_variables()
```

**After:**
```python
# Automatic environment variable loading
# Just set environment variables and they're automatically loaded:
# export GEMINI_API_KEY="your_key"
# export PHM_MAX_DEPTH="10"

state = PHMState.from_case_config(...)
print(state.config.llm.gemini_api_key)  # Automatically loaded
print(state.config.processing.max_depth)  # Automatically loaded
```

## YAML Configuration Structure

### New Structure
```yaml
# LLM configuration
llm:
  provider: "google"
  model: "gemini-2.5-pro"
  temperature: 0.7
  max_retries: 3

# Processing configuration  
processing:
  min_depth: 4
  max_depth: 8
  default_fs: 1000.0

# Paths configuration
paths:
  data_dir: "/path/to/data"
  save_dir: "/path/to/save"
  cache_dir: "/path/to/cache"

# Case-specific data (stored in config.extra)
name: "case1"
user_instruction: "Analyze signals..."
ref_ids: [47050, 47052, 47044]
test_ids: [47051, 47045, 47048]
```

## Backward Compatibility

The new system maintains 100% backward compatibility:

```python
# These still work (with deprecation warnings)
model = state.phm_model  # Deprecated, use state.config.llm.model
query_model = state.query_generator_model  # Deprecated

# These still work without warnings
llm_config = state.get_llm_config()
processing_config = state.get_processing_config()
state.update_from_yaml("config.yaml")
```

## Benefits of the New System

1. **Reduced Complexity**: 636 lines vs 730+ lines
2. **Type Safety**: Typed configuration access with IDE support
3. **Better Performance**: Direct attribute access vs string parsing
4. **Cleaner API**: Factory methods vs complex initialization
5. **Easier Testing**: Simplified validation and state management
6. **Better Documentation**: Clear structure with typed fields

## Migration Checklist

- [ ] Update initialization code to use `PHMState.from_case_config()`
- [ ] Replace string-based config access with typed access
- [ ] Update YAML files to use new structure
- [ ] Replace multiple validation calls with single `validate()`
- [ ] Test that existing agents still work
- [ ] Update any custom configuration code

## Common Issues and Solutions

### Issue: "AttributeError: 'PHMConfig' object has no attribute 'custom_field'"
**Solution**: Custom fields are stored in `config.extra`:
```python
# Instead of: state.config.custom_field
# Use: state.config.extra['custom_field']
```

### Issue: "Deprecation warnings for phm_model"
**Solution**: Update to use typed access:
```python
# Instead of: state.phm_model
# Use: state.config.llm.model
```

### Issue: "YAML configuration not loading"
**Solution**: Ensure YAML structure matches new format:
```yaml
llm:
  model: "gemini-2.5-pro"
# Not: model: "gemini-2.5-pro"
```

## Testing the Migration

Run the provided test script to verify everything works:

```bash
python test_simplified_state.py
```

This will test:
- State initialization with new factory method
- Configuration access patterns
- YAML loading
- Backward compatibility
- Validation system
