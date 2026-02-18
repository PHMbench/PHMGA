# Configuration System Bug Report
## Reviewer: reviewer-8 (Configuration Management Specialist)

**Date:** 2026-02-15
**Review Scope:** Configuration system including `src/configuration.py` and all YAML files in `config/`

---

## Executive Summary

This review identified **8 bugs** and **5 potential issues** in the configuration system:

**Critical Bugs (3):**
1. Hardcoded user-specific paths in configuration files prevent portability
2. Missing `run_executor` flag in legacy case configs causes silent failures
3. Configuration class does not validate API key presence before use

**High Priority Bugs (3):**
4. Environment variable fallback inconsistency in Configuration.from_runnable_config
5. Duplicate `.env` loading across multiple files can cause unexpected behavior
6. Missing config field validation for `state_save_path` and `report_path`

**Medium Priority Bugs (2):**
7. Model config schema allows inconsistent `num_classes` vs `out_channels`
8. Case configs lack `train_backend` field, causing inconsistent defaults

**Low Priority Issues (5):**
9. Relative path handling inconsistency between config files
10. Missing validation for `builder.min_depth` vs `builder.max_depth`
11. API key fields in Configuration have no masking in logs/errors
12. `preflight.strict` defaults to True but config files don't document this
13. Duplicate hardcoded model profile map in case1.py

---

## Detailed Bug Analysis

### Bug #1: Hardcoded User-Specific Paths (CRITICAL)

**File:** Multiple config files
- `/home/user/LQ/B_Signal/PHMGA/config/case1.yaml` (lines 5-9)
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp2.yaml` (lines 5-9)
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp2.5.yaml` (lines 5-9)
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp_ottawa.yaml` (lines 5-9)

**Problem Description:**
Configuration files contain hardcoded paths pointing to user `lq`'s home directory:
```yaml
save_dir: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save"
metadata_path: "/mnt/crucial/LQ/PHM-Vibench/metadata_6_11.xlsx"
state_save_path: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/case1/case1_built_state.pkl"
report_path: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/case1/case1_final_report.md"
```

**Potential Impact:**
- Config files will fail for any user other than `lq`
- Causes `FileNotFoundError` when running cases
- Prevents sharing of configuration between team members
- CI/CD pipelines will fail

**Fix Suggestion:**
Use relative paths or environment variables:
```yaml
# Option 1: Relative paths
save_dir: "./save/case1"
metadata_path: "/data/PHM-Vibench/metadata_6_11.xlsx"  # Use a data root env var

# Option 2: Environment variable substitution
save_dir: "${PHM_SAVE_DIR}/save/case1"
metadata_path: "${PHM_DATA_ROOT}/PHM-Vibench/metadata_6_11.xlsx"
```

**Code Snippet of Problematic Code:**
```yaml
# From config/case1.yaml lines 5-9
name: "case1"
save_dir: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save"
metadata_path: "/mnt/crucial/LQ/PHM-Vibench/metadata_6_11.xlsx"
h5_path: "/home/user/data/PHMbenchdata/PHM-Vibench/cache.h5"
state_save_path: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/case1/case1_built_state.pkl"
report_path: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/case1/case1_final_report.md"
```

---

### Bug #2: Missing `run_executor` Flag in Legacy Configs (CRITICAL)

**File:** Multiple config files
- `/home/user/LQ/B_Signal/PHMGA/config/case1.yaml`
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp2.yaml`
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp2.5.yaml`
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp_ottawa.yaml`

**Problem Description:**
Legacy case configs (case1, case_exp2, case_exp2.5, case_exp_ottawa) are missing the `run_executor` flag. When this flag is missing:
- The code at `src/cases/case1.py:338` defaults to `False`
- The executor workflow is silently skipped
- No training, no final report is generated
- Users may think the case ran successfully when only the DAG was built

**Code Location:** `src/cases/case1.py:338`
```python
if bool(config.get("run_executor", False)):
```

**Potential Impact:**
- Users expect reports and training outputs but get nothing
- Silent failure mode is confusing
- Inconsistent behavior between old and new configs

**Fix Suggestion:**
1. Add `run_executor: true` to all legacy configs
2. OR change the default to `True` and require explicit `false` to skip
3. Add a warning when executor is skipped

**Code Snippet:**
```yaml
# Add to all legacy configs:
run_executor: true
```

---

### Bug #3: Configuration Class Missing API Key Validation (CRITICAL)

**File:** `/home/user/LQ/B_Signal/PHMGA/src/configuration.py` (lines 92-110)

**Problem Description:**
The `Configuration.from_runnable_config()` method creates configuration objects without validating that required API keys are present. The method:
1. Reads environment variables using `os.environ.get(name.upper())`
2. Filters out `None` values
3. Creates a Configuration object with remaining values
4. Never checks if API keys actually exist before use

When `get_llm()` is called with a configuration that has no API keys, it only fails at the point of API call, not at configuration time.

**Code Location:** `src/configuration.py:92-110`
```python
@classmethod
def from_runnable_config(
    cls, config: Optional[RunnableConfig] = None
) -> "Configuration":
    """Create a Configuration instance from a RunnableConfig."""
    configurable = (
        config["configurable"] if config and "configurable" in config else {}
    )

    # Get raw values from environment or config
    raw_values: dict[str, Any] = {
        name: os.environ.get(name.upper(), configurable.get(name))
        for name in cls.model_fields.keys()
    }

    # Filter out None values
    values = {k: v for k, v in raw_values.items() if v is not None}

    return cls(**values)
```

**Potential Impact:**
- API calls fail with cryptic errors after significant computation
- Difficult to debug - error appears in agent code, not configuration
- No early validation to catch missing credentials

**Fix Suggestion:**
Add a `validate()` method that checks API key presence based on provider:
```python
def validate_api_keys(self) -> list[str]:
    """Return list of missing API keys for the current provider."""
    provider = self.llm_provider.lower()
    missing = []

    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    elif provider == "glm" and not (self.glm_api_key or os.getenv("GLM_API_KEY")):
        missing.append("GLM_API_KEY")
    elif provider in ("deepseek", "openai_compatible"):
        if not (self.openai_api_key or os.getenv("OPENAI_API_KEY") or
                self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")):
            missing.append(f"{provider.upper()}_API_KEY or OPENAI_API_KEY")

    return missing
```

---

### Bug #4: Environment Variable Fallback Inconsistency (HIGH)

**File:** `/home/user/LQ/B_Signal/PHMGA/src/configuration.py` (line 103)

**Problem Description:**
The `from_runnable_config()` method uses `os.environ.get(name.upper())` to read environment variables. This assumes all config fields have environment variables named exactly as the uppercase field name.

However, looking at actual usage in `src/model/__init__.py`:
- The code checks `os.getenv("LLM_PROVIDER")` (not `LLMPROVIDER`)
- The code checks `os.getenv("QUERY_GENERATOR_MODEL")` (not `QUERYGENERATORMODEL`)

The environment variable names don't match the field names when uppercased.

**Code Location:** `src/configuration.py:103`
```python
raw_values: dict[str, Any] = {
    name: os.environ.get(name.upper(), configurable.get(name))
    for name in cls.model_fields.keys()
}
```

**Actual env vars used:** `src/model/__init__.py:117-124`
```python
provider = (
    os.getenv("LLM_PROVIDER")
    or getattr(configurable, "llm_provider", None)
    or "gemini"
).strip().lower()
model_name = (
    os.getenv("QUERY_GENERATOR_MODEL")
    or os.getenv("PHM_MODEL")
    or str(getattr(configurable, "query_generator_model", "") or "").strip()
)
```

**Potential Impact:**
- Configuration.from_runnable_config() doesn't actually read the environment variables used by the system
- Setting `LLM_PROVIDER` in env won't affect Configuration object
- Inconsistency between config loading and LLM initialization

**Fix Suggestion:**
Define explicit environment variable names for each field:
```python
class Configuration(BaseModel):
    llm_provider: str = Field(
        default="gemini",
        env="LLM_PROVIDER",  # Add explicit env var name
        ...
    )
```

---

### Bug #5: Duplicate .env Loading Across Multiple Files (HIGH)

**Files:**
- `/home/user/LQ/B_Signal/PHMGA/src/utils/__init__.py` (lines 18-25)
- `/home/user/LQ/B_Signal/PHMGA/src/cases/case1.py` (lines 10-13)
- `/home/user/LQ/B_Signal/PHMGA/src/model/__init__.py` (lines 17-68)
- `/home/user/LQ/B_Signal/PHMGA/src/utils.py` (lines 18-20)

**Problem Description:**
The `.env` file is loaded in multiple places with slightly different logic:
1. `src/utils/__init__.py` loads it at module import time
2. `src/cases/case1.py` loads it again with `Path.cwd() / ".env"`
3. `src/model/__init__.py` has a complex `_best_effort_load_dotenv()` with multiple fallback paths
4. `src/utils.py` (legacy) loads it without specifying a path

Each of these uses different path resolution logic and some use `override=False` while others don't specify.

**Potential Impact:**
- Unpredictable which `.env` file is actually used
- Potential for loading different `.env` files in different parts of the code
- Performance overhead from multiple file reads
- Confusing behavior when `.env` is in different locations

**Fix Suggestion:**
1. Create a single module responsible for `.env` loading
2. Call it once at application entry point
3. Have all other code rely on environment variables being already loaded

---

### Bug #6: Missing Config Field Validation (HIGH)

**File:** `/home/user/LQ/B_Signal/PHMGA/src/cases/case1.py` (lines 143-143, 360-361)

**Problem Description:**
The code accesses `config['state_save_path']` and `config['report_path']` without checking if these keys exist. If a YAML config is missing these fields, a `KeyError` will be raised.

**Code Locations:**
```python
# Line 143 - No validation before access
state_save_path = config["state_save_path"]

# Line 360 - No validation before access
generate_final_report(final_state, config['report_path'])
```

**Potential Impact:**
- `KeyError` when configs are missing required fields
- Unclear error message for users
- No early validation to catch config errors

**Fix Suggestion:**
```python
# Add validation after loading config:
required_fields = ['state_save_path', 'report_path', 'name', 'user_instruction']
missing = [f for f in required_fields if f not in config]
if missing:
    raise ValueError(f"Config missing required fields: {missing}")
```

---

### Bug #7: Model Config Schema Allows Inconsistent num_classes vs out_channels (MEDIUM)

**File:** `/home/user/LQ/B_Signal/PHMGA/config/model_tspn_basic.yaml` (lines 8-12)

**Problem Description:**
The model config has `num_classes: 5` but `out_channels: 3`. For a classification task, `out_channels` typically should equal `num_classes` (or be related to the number of classes). This inconsistency could cause training issues.

**Code Snippet:**
```yaml
model:
  name: tspn
  device: cpu
  num_classes: 5
  in_dim: 4096
  in_channels: 2

  out_channels: 3  # This doesn't match num_classes!
```

**Potential Impact:**
- Model may produce incorrect output dimensions
- Classification may fail or produce wrong results
- Confusing for users who expect `num_classes` to match output

**Fix Suggestion:**
1. Either set `out_channels: 5` to match `num_classes`
2. OR add validation in `ModelConfig` to warn when these don't match
3. OR clarify the relationship in documentation

---

### Bug #8: Case Configs Lacking train_backend Field (MEDIUM)

**Files:**
- `/home/user/LQ/B_Signal/PHMGA/config/case1.yaml`
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp2.yaml`
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp2.5.yaml`
- `/home/user/LQ/B_Signal/PHMGA/config/case_exp_ottawa.yaml`

**Problem Description:**
Legacy case configs don't specify `train_backend`. The code defaults to different values in different places:
- Line 206 of `src/cases/case1.py`: defaults to `"tspn"` for vibench mode
- Line 223 of `src/cases/case1.py`: defaults to `"shallow"` for fixed_ids mode

This inconsistent default behavior can confuse users.

**Code Locations:**
```python
# Line 206 - vibench mode defaults to "tspn"
train_backend=str(config.get("train_backend", "tspn")),

# Line 223 - fixed_ids mode defaults to "shallow"
train_backend=str(config.get("train_backend", "shallow")),
```

**Potential Impact:**
- Users get different training backends depending on data mode
- Unexpected behavior when migrating between configs
- Hard to reproduce results

**Fix Suggestion:**
1. Add explicit `train_backend` to all configs
2. OR use a single consistent default with clear documentation

---

## Low Priority Issues

### Issue #9: Relative Path Handling Inconsistency

Some configs use relative paths (e.g., `config/model_tspn_basic.yaml`) while others use absolute paths. There's no consistent resolution strategy, leading to potential path resolution failures depending on CWD.

### Issue #10: Missing builder.min_depth vs builder.max_depth Validation

The code doesn't validate that `min_depth <= max_depth`. A config with reversed values would cause infinite loops or other issues.

### Issue #11: API Key Fields Not Masked

When Configuration objects are logged or serialized, API key fields are not masked, potentially exposing sensitive credentials in logs.

### Issue #12: preflight.strict Defaults Not Documented

The `preflight.strict` setting defaults to `True` but this isn't documented in the config files. Users may not understand why their cases fail preflight checks.

### Issue #13: Duplicate Model Profile Map

The `_MODEL_PROFILE_MAP` in `case1.py` is hardcoded and duplicated logic. It should be in a central configuration or discovered dynamically.

---

## Recommendations

1. **Create a Config Schema:** Use Pydantic to validate all case configs against a schema
2. **Centralize Path Resolution:** Use a single function to resolve all config paths
3. **Add Config Linter:** Create a pre-commit hook that validates configs
4. **Document Environment Variables:** Maintain a single source of truth for all env vars
5. **Add Config Migration:** Tool to update old configs to new format

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 3 |
| Medium | 2 |
| Low | 5 |
| **Total** | **13** |

---

## End of Report

**Reviewed by:** reviewer-8
**Configuration files analyzed:** 6
**Python files analyzed:** 4
**Lines of code reviewed:** ~500
