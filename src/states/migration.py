"""
Migration utilities for transitioning to the unified state management system.

This module provides tools to help migrate from the old configuration patterns
to the new UnifiedStateManager system while maintaining backward compatibility.
"""

from __future__ import annotations
import warnings
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from .phm_states import get_unified_state, UnifiedStateManager


class StateMigrationHelper:
    """Helper class for migrating to unified state management."""
    
    def __init__(self):
        self.unified_state = get_unified_state()
        self.migration_warnings: List[str] = []
    
    def migrate_environment_variables(self) -> Dict[str, str]:
        """
        Migrate old environment variable patterns to new unified structure.
        
        Returns
        -------
        Dict[str, str]
            Mapping of old variable names to new unified paths
        """
        migration_map = {
            # Legacy environment variables -> unified paths
            'PHM_MODEL': 'llm.model',
            'QUERY_GENERATOR_MODEL': 'llm.query_generator_model',
            'REFLECTION_MODEL': 'llm.reflection_model',
            'ANSWER_MODEL': 'llm.answer_model',
            'GEMINI_API_KEY': 'llm.gemini_api_key',
            'OPENAI_API_KEY': 'llm.openai_api_key',
            'DATA_DIR': 'paths.data_dir',
            'SAVE_DIR': 'paths.save_dir',
            'MAX_DEPTH': 'processing.max_depth',
            'MIN_DEPTH': 'processing.min_depth',
            'MAX_STEPS': 'processing.max_steps',
        }
        
        migrated = {}
        for old_var, new_path in migration_map.items():
            if old_var in os.environ:
                old_value = os.environ[old_var]
                current_value = self.unified_state.get(new_path)
                
                if current_value != old_value:
                    self.unified_state.set(new_path, old_value, category='env')
                    migrated[old_var] = new_path
                    
                    self.migration_warnings.append(
                        f"Migrated {old_var}={old_value} to {new_path}"
                    )
        
        return migrated
    
    def migrate_configuration_object(self, config_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Migrate old Configuration object values to unified state.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Dictionary representation of old Configuration object
        
        Returns
        -------
        Dict[str, str]
            Mapping of old config keys to new unified paths
        """
        migration_map = {
            'phm_model': 'llm.model',
            'query_generator_model': 'llm.query_generator_model',
            'reflection_model': 'llm.reflection_model',
            'answer_model': 'llm.answer_model',
        }
        
        migrated = {}
        for old_key, new_path in migration_map.items():
            if old_key in config_dict:
                old_value = config_dict[old_key]
                self.unified_state.set(new_path, old_value, category='config')
                migrated[old_key] = new_path
                
                self.migration_warnings.append(
                    f"Migrated config.{old_key}={old_value} to {new_path}"
                )
        
        return migrated
    
    def migrate_yaml_config(self, yaml_path: str) -> None:
        """
        Migrate YAML configuration file to unified state.
        
        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file
        """
        import yaml
        
        if not Path(yaml_path).exists():
            self.migration_warnings.append(f"YAML file not found: {yaml_path}")
            return
        
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Migrate YAML structure to unified state
        self.unified_state.update_from_yaml(yaml_path)
        
        self.migration_warnings.append(f"Migrated YAML config from {yaml_path}")
    
    def check_deprecated_usage(self, code_content: str) -> List[str]:
        """
        Check code content for deprecated usage patterns.
        
        Parameters
        ----------
        code_content : str
            Source code content to check
        
        Returns
        -------
        List[str]
            List of deprecated patterns found
        """
        deprecated_patterns = [
            ('Configuration()', 'get_unified_state()'),
            ('config.phm_model', 'state.get_config("llm.model")'),
            ('config.query_generator_model', 'state.get_config("llm.query_generator_model")'),
            ('os.environ["PHM_MODEL"]', 'state.get_config("llm.model")'),
            ('os.getenv("PHM_MODEL")', 'state.get_config("llm.model")'),
            ('from src.configuration import Configuration', 'from src.states.phm_states import get_unified_state'),
        ]
        
        found_patterns = []
        for old_pattern, new_pattern in deprecated_patterns:
            if old_pattern in code_content:
                found_patterns.append(f"Replace '{old_pattern}' with '{new_pattern}'")
        
        return found_patterns
    
    def generate_migration_report(self) -> str:
        """
        Generate a comprehensive migration report.
        
        Returns
        -------
        str
            Markdown-formatted migration report
        """
        report = ["# State Management Migration Report\n"]
        
        if self.migration_warnings:
            report.append("## Migrations Performed\n")
            for warning in self.migration_warnings:
                report.append(f"- {warning}")
            report.append("")
        
        # Current unified state configuration
        report.append("## Current Unified State Configuration\n")
        config_export = self.unified_state.export_config()
        
        for category, values in config_export.items():
            if category != 'deprecated_access' and values:
                report.append(f"### {category.title()}\n")
                for key, value in values.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            report.append(f"- `{key}.{subkey}`: {subvalue}")
                    else:
                        report.append(f"- `{key}`: {value}")
                report.append("")
        
        # Deprecated access tracking
        deprecated_access = config_export.get('deprecated_access', {})
        if deprecated_access:
            report.append("## Deprecated Access Patterns\n")
            for pattern, count in deprecated_access.items():
                report.append(f"- `{pattern}`: accessed {count} times")
            report.append("")
        
        # Validation results
        validation_errors = self.unified_state.validate_required_config()
        if validation_errors:
            report.append("## Configuration Validation Errors\n")
            for error in validation_errors:
                report.append(f"- ❌ {error}")
        else:
            report.append("## Configuration Validation\n")
            report.append("- ✅ All required configuration is present")
        
        report.append("")
        
        # Migration recommendations
        report.append("## Migration Recommendations\n")
        report.append("1. Update import statements:")
        report.append("   ```python")
        report.append("   # Old")
        report.append("   from src.configuration import Configuration")
        report.append("   config = Configuration()")
        report.append("")
        report.append("   # New")
        report.append("   from src.states.phm_states import get_unified_state")
        report.append("   unified_state = get_unified_state()")
        report.append("   ```")
        report.append("")
        
        report.append("2. Update configuration access:")
        report.append("   ```python")
        report.append("   # Old")
        report.append("   model_name = config.phm_model")
        report.append("")
        report.append("   # New")
        report.append("   model_name = unified_state.get('llm.model')")
        report.append("   # Or in PHMState")
        report.append("   model_name = state.get_config('llm.model')")
        report.append("   ```")
        report.append("")
        
        report.append("3. Update environment variable usage:")
        report.append("   ```python")
        report.append("   # Old")
        report.append("   data_dir = os.getenv('PHM_DATA_DIR', '/default/path')")
        report.append("")
        report.append("   # New")
        report.append("   data_dir = unified_state.get('paths.data_dir', '/default/path')")
        report.append("   ```")
        
        return "\n".join(report)
    
    def save_migration_report(self, output_path: str) -> None:
        """
        Save migration report to file.
        
        Parameters
        ----------
        output_path : str
            Path to save the migration report
        """
        report = self.generate_migration_report()
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Migration report saved to: {output_path}")


def migrate_to_unified_state(
    config_dict: Optional[Dict[str, Any]] = None,
    yaml_path: Optional[str] = None,
    check_code_files: Optional[List[str]] = None,
    output_report: Optional[str] = None
) -> StateMigrationHelper:
    """
    Comprehensive migration to unified state management.
    
    Parameters
    ----------
    config_dict : Dict[str, Any], optional
        Old Configuration object as dictionary
    yaml_path : str, optional
        Path to YAML configuration file
    check_code_files : List[str], optional
        List of Python files to check for deprecated patterns
    output_report : str, optional
        Path to save migration report
    
    Returns
    -------
    StateMigrationHelper
        Migration helper with results
    """
    helper = StateMigrationHelper()
    
    # Migrate environment variables
    helper.migrate_environment_variables()
    
    # Migrate configuration object
    if config_dict:
        helper.migrate_configuration_object(config_dict)
    
    # Migrate YAML configuration
    if yaml_path:
        helper.migrate_yaml_config(yaml_path)
    
    # Check code files for deprecated patterns
    if check_code_files:
        for file_path in check_code_files:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                deprecated = helper.check_deprecated_usage(content)
                if deprecated:
                    helper.migration_warnings.extend([
                        f"In {file_path}: {pattern}" for pattern in deprecated
                    ])
    
    # Save migration report
    if output_report:
        helper.save_migration_report(output_report)
    
    # Print summary
    if helper.migration_warnings:
        print(f"Migration completed with {len(helper.migration_warnings)} items migrated.")
        print("Run with output_report parameter to get detailed migration report.")
    else:
        print("No migration needed - configuration is already up to date.")
    
    return helper


if __name__ == "__main__":
    # Example migration
    helper = migrate_to_unified_state(
        yaml_path="config/case1.yaml",
        output_report="migration_report.md"
    )
    
    print("Migration completed!")
    print(f"Unified state configuration: {helper.unified_state.export_config()}")
