"""
Agent Factory and Configuration Management for PHM Research Agents.

This module provides centralized agent creation, configuration management,
and dependency injection for the PHM research agent system.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Type, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from pathlib import Path

from .research_base import ResearchAgentBase, InputValidator, SignalValidator, StateValidator

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of available agent types."""
    DATA_ANALYST = "data_analyst"
    ALGORITHM_RESEARCHER = "algorithm_researcher"
    DOMAIN_EXPERT = "domain_expert"
    INTEGRATION = "integration"


@dataclass
class AgentConfig:
    """Configuration container for research agents."""
    agent_type: AgentType
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    validators: Dict[str, str] = field(default_factory=dict)
    performance_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        
        # Set default performance limits
        if not self.performance_limits:
            self.performance_limits = {
                "max_execution_time": 300,  # 5 minutes
                "max_memory_mb": 1024,      # 1 GB
                "max_failures": 3
            }


@dataclass
class AgentPreset:
    """Predefined agent configuration presets."""
    name: str
    description: str
    agents: List[AgentConfig]
    workflow_config: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Manages agent configurations and presets."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.presets: Dict[str, AgentPreset] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self._load_default_presets()
        
        if config_path:
            self.load_from_file(config_path)
    
    def _load_default_presets(self):
        """Load default agent presets."""
        # Quick Diagnosis preset
        quick_diagnosis = AgentPreset(
            name="QuickDiagnosis",
            description="Fast diagnosis with essential agents only",
            agents=[
                AgentConfig(
                    agent_type=AgentType.DATA_ANALYST,
                    name="quick_data_analyst",
                    config={
                        "enable_advanced_features": False,
                        "max_features": 5,
                        "quick_mode": True
                    }
                ),
                AgentConfig(
                    agent_type=AgentType.DOMAIN_EXPERT,
                    name="quick_domain_expert",
                    config={
                        "enable_physics_validation": True,
                        "enable_bearing_analysis": True,
                        "quick_mode": True
                    }
                )
            ],
            workflow_config={
                "max_iterations": 1,
                "quality_threshold": 0.6,
                "parallel_execution": False
            }
        )
        
        # Comprehensive Research preset
        comprehensive_research = AgentPreset(
            name="ComprehensiveResearch",
            description="Full research analysis with all agents",
            agents=[
                AgentConfig(
                    agent_type=AgentType.DATA_ANALYST,
                    name="comprehensive_data_analyst",
                    config={
                        "enable_advanced_features": True,
                        "enable_pca_analysis": True,
                        "enable_clustering": True
                    }
                ),
                AgentConfig(
                    agent_type=AgentType.ALGORITHM_RESEARCHER,
                    name="comprehensive_algorithm_researcher",
                    config={
                        "enable_hyperparameter_optimization": True,
                        "enable_cross_validation": True,
                        "max_algorithms": 10
                    }
                ),
                AgentConfig(
                    agent_type=AgentType.DOMAIN_EXPERT,
                    name="comprehensive_domain_expert",
                    config={
                        "enable_physics_validation": True,
                        "enable_bearing_analysis": True,
                        "enable_frequency_analysis": True
                    }
                ),
                AgentConfig(
                    agent_type=AgentType.INTEGRATION,
                    name="comprehensive_integration",
                    config={
                        "enable_conflict_resolution": True,
                        "enable_consensus_building": True
                    }
                )
            ],
            workflow_config={
                "max_iterations": 5,
                "quality_threshold": 0.85,
                "parallel_execution": True
            }
        )
        
        # Predictive Maintenance preset
        predictive_maintenance = AgentPreset(
            name="PredictiveMaintenance",
            description="Optimized for predictive maintenance analysis",
            agents=[
                AgentConfig(
                    agent_type=AgentType.DATA_ANALYST,
                    name="predictive_data_analyst",
                    config={
                        "focus_on_degradation": True,
                        "enable_trend_analysis": True,
                        "health_features": ["rms", "kurtosis", "crest_factor"]
                    }
                ),
                AgentConfig(
                    agent_type=AgentType.DOMAIN_EXPERT,
                    name="predictive_domain_expert",
                    config={
                        "enable_rul_estimation": True,
                        "enable_maintenance_scheduling": True,
                        "degradation_models": ["linear", "exponential", "weibull"]
                    }
                ),
                AgentConfig(
                    agent_type=AgentType.INTEGRATION,
                    name="predictive_integration",
                    config={
                        "focus_on_maintenance": True,
                        "enable_risk_assessment": True
                    }
                )
            ],
            workflow_config={
                "max_iterations": 3,
                "quality_threshold": 0.8,
                "parallel_execution": True,
                "enable_predictive_features": True
            }
        )
        
        self.presets = {
            "QuickDiagnosis": quick_diagnosis,
            "ComprehensiveResearch": comprehensive_research,
            "PredictiveMaintenance": predictive_maintenance
        }
    
    def get_preset(self, preset_name: str) -> Optional[AgentPreset]:
        """Get a predefined agent preset."""
        return self.presets.get(preset_name)
    
    def list_presets(self) -> List[str]:
        """List available presets."""
        return list(self.presets.keys())
    
    def validate_config(self, config: AgentConfig) -> Tuple[bool, List[str]]:
        """Validate agent configuration."""
        errors = []
        
        if not config.name:
            errors.append("Agent name is required")
        
        if not isinstance(config.agent_type, AgentType):
            errors.append("Invalid agent type")
        
        # Validate performance limits
        if config.performance_limits:
            if "max_execution_time" in config.performance_limits:
                if config.performance_limits["max_execution_time"] <= 0:
                    errors.append("max_execution_time must be positive")
            
            if "max_memory_mb" in config.performance_limits:
                if config.performance_limits["max_memory_mb"] <= 0:
                    errors.append("max_memory_mb must be positive")
        
        return len(errors) == 0, errors
    
    def load_from_file(self, config_path: str):
        """Load configuration from file."""
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return
            
            with open(path, 'r') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Load agent configurations
            if "agents" in data:
                for agent_data in data["agents"]:
                    config = AgentConfig(**agent_data)
                    is_valid, errors = self.validate_config(config)
                    if is_valid:
                        self.agent_configs[config.name] = config
                    else:
                        logger.error(f"Invalid agent config {config.name}: {errors}")
            
            # Load custom presets
            if "presets" in data:
                for preset_data in data["presets"]:
                    preset = AgentPreset(**preset_data)
                    self.presets[preset.name] = preset
                    
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
    
    def save_to_file(self, config_path: str):
        """Save current configuration to file."""
        try:
            data = {
                "agents": [config.__dict__ for config in self.agent_configs.values()],
                "presets": [preset.__dict__ for preset in self.presets.values()]
            }
            
            path = Path(config_path)
            with open(path, 'w') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    yaml.dump(data, f, default_flow_style=False)
                else:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")


class DependencyInjector:
    """Manages dependency injection for agents."""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.factories: Dict[str, callable] = {}
    
    def register_service(self, name: str, service: Any):
        """Register a service instance."""
        self.services[name] = service
    
    def register_factory(self, name: str, factory: callable):
        """Register a service factory."""
        self.factories[name] = factory
    
    def get_service(self, name: str) -> Any:
        """Get a service instance."""
        if name in self.services:
            return self.services[name]
        
        if name in self.factories:
            service = self.factories[name]()
            self.services[name] = service  # Cache the instance
            return service
        
        raise ValueError(f"Service not found: {name}")
    
    def create_dependencies(self, config: AgentConfig) -> Dict[str, Any]:
        """Create dependencies for an agent based on its configuration."""
        dependencies = {}
        
        for dep_name, dep_config in config.dependencies.items():
            if isinstance(dep_config, str):
                # Simple service reference
                dependencies[dep_name] = self.get_service(dep_config)
            elif isinstance(dep_config, dict):
                # Service with configuration
                service_name = dep_config.get("service")
                if service_name:
                    service = self.get_service(service_name)
                    # Apply configuration to service if supported
                    if hasattr(service, "configure"):
                        service.configure(dep_config.get("config", {}))
                    dependencies[dep_name] = service
        
        return dependencies


class AgentFactory:
    """Factory for creating and configuring research agents."""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.dependency_injector = DependencyInjector()
        self.agent_registry: Dict[AgentType, Type[ResearchAgentBase]] = {}
        self._register_default_agents()
        self._register_default_services()
    
    def _register_default_agents(self):
        """Register default agent types."""
        # Import here to avoid circular imports
        try:
            from .data_analyst_agent import DataAnalystAgent
            from .algorithm_researcher_agent import AlgorithmResearcherAgent
            from .domain_expert_agent import DomainExpertAgent
            from .integration_agent import IntegrationAgent
            
            self.agent_registry[AgentType.DATA_ANALYST] = DataAnalystAgent
            self.agent_registry[AgentType.ALGORITHM_RESEARCHER] = AlgorithmResearcherAgent
            self.agent_registry[AgentType.DOMAIN_EXPERT] = DomainExpertAgent
            self.agent_registry[AgentType.INTEGRATION] = IntegrationAgent
            
        except ImportError as e:
            logger.warning(f"Could not import default agents: {e}")
    
    def _register_default_services(self):
        """Register default services."""
        # Register validators
        self.dependency_injector.register_factory("signal_validator", SignalValidator)
        self.dependency_injector.register_factory("state_validator", StateValidator)
        
        # Register other common services
        self.dependency_injector.register_factory("logger", lambda: logging.getLogger("phm_agents"))
    
    def register_agent_type(self, agent_type: AgentType, agent_class: Type[ResearchAgentBase]):
        """Register a custom agent type."""
        self.agent_registry[agent_type] = agent_class
    
    def create_agent(self, config: AgentConfig) -> ResearchAgentBase:
        """Create an agent instance from configuration."""
        if config.agent_type not in self.agent_registry:
            raise ValueError(f"Unknown agent type: {config.agent_type}")
        
        agent_class = self.agent_registry[config.agent_type]
        
        # Create validators
        validators = {}
        for validator_name, validator_type in config.validators.items():
            if validator_type == "signal":
                validators[validator_name] = SignalValidator()
            elif validator_type == "state":
                validators[validator_name] = StateValidator()
        
        # Create dependencies
        dependencies = self.dependency_injector.create_dependencies(config)
        
        # Create agent instance
        agent = agent_class(
            agent_name=config.name,
            config=config.config,
            validators=validators,
            dependencies=dependencies
        )
        
        return agent
    
    def create_agent_from_preset(self, preset_name: str) -> List[ResearchAgentBase]:
        """Create agents from a preset configuration."""
        preset = self.config_manager.get_preset(preset_name)
        if not preset:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        agents = []
        for agent_config in preset.agents:
            if agent_config.enabled:
                agent = self.create_agent(agent_config)
                agents.append(agent)
        
        return agents
    
    def create_agent_builder(self, agent_type: AgentType) -> 'AgentBuilder':
        """Create a fluent builder for agent configuration."""
        return AgentBuilder(self, agent_type)


class AgentBuilder:
    """Fluent builder for agent configuration."""
    
    def __init__(self, factory: AgentFactory, agent_type: AgentType):
        self.factory = factory
        self.config = AgentConfig(agent_type=agent_type, name=f"{agent_type.value}_agent")
    
    def with_name(self, name: str) -> 'AgentBuilder':
        """Set agent name."""
        self.config.name = name
        return self
    
    def with_config(self, **kwargs) -> 'AgentBuilder':
        """Add configuration parameters."""
        self.config.config.update(kwargs)
        return self
    
    def with_dependency(self, name: str, service: str) -> 'AgentBuilder':
        """Add a dependency."""
        self.config.dependencies[name] = service
        return self
    
    def with_validator(self, name: str, validator_type: str) -> 'AgentBuilder':
        """Add a validator."""
        self.config.validators[name] = validator_type
        return self
    
    def with_performance_limit(self, **kwargs) -> 'AgentBuilder':
        """Set performance limits."""
        self.config.performance_limits.update(kwargs)
        return self
    
    def build(self) -> ResearchAgentBase:
        """Build the agent."""
        is_valid, errors = self.factory.config_manager.validate_config(self.config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
        
        return self.factory.create_agent(self.config)


if __name__ == "__main__":
    # Example usage
    factory = AgentFactory()
    
    # Create agent using builder pattern
    agent = (factory.create_agent_builder(AgentType.DATA_ANALYST)
             .with_name("my_data_analyst")
             .with_config(enable_advanced_features=True)
             .with_validator("state", "state")
             .build())
    
    print(f"Created agent: {agent.agent_name}")
    
    # Create agents from preset
    agents = factory.create_agent_from_preset("QuickDiagnosis")
    print(f"Created {len(agents)} agents from preset")
