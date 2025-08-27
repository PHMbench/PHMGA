"""
Tutorial Bridge Module

Connects tutorial concepts from Parts 1-4 with the production PHMGA system.
Provides mapping and translation between educational concepts and real system components.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Type, Callable
from dataclasses import dataclass
from enum import Enum

# Add src/ to path for production system imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Production system imports
from tools.signal_processing_schemas import OP_REGISTRY, PHMOperator, ExpandOp, TransformOp, AggregateOp, DecisionOp, MultiVariableOp
from agents import plan_agent, execute_agent, reflect_agent, inquirer_agent, dataset_preparer_agent, shallow_ml_agent, report_agent
from states.phm_states import PHMState, DAGState, InputData, ProcessedData, SimilarityNode, DataSetNode, OutputNode


class TutorialConcept(Enum):
    """Tutorial concepts from Parts 1-4"""
    
    # Part 1: LLM Foundations
    LLM_PROVIDERS = "llm_providers"
    RESEARCH_LLM = "research_llm"
    PROMPT_ENGINEERING = "prompt_engineering"
    
    # Part 2: Multi-Agent Router  
    AGENT_ROUTER = "agent_router"
    RESEARCH_AGENTS = "research_agents"
    PARALLEL_EXECUTION = "parallel_execution"
    
    # Part 3: Research Integration
    RESEARCH_WORKFLOW = "research_workflow"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    ITERATIVE_RESEARCH = "iterative_research"
    
    # Part 4: DAG Architecture
    DAG_STRUCTURE = "dag_structure" 
    SIGNAL_PROCESSING = "signal_processing"
    NODE_RELATIONSHIPS = "node_relationships"


class ProductionComponent(Enum):
    """Production PHMGA components"""
    
    # LangGraph workflows
    BUILDER_GRAPH = "builder_graph"
    EXECUTOR_GRAPH = "executor_graph"
    
    # Production agents
    PLAN_AGENT = "plan_agent"
    EXECUTE_AGENT = "execute_agent"
    REFLECT_AGENT = "reflect_agent"
    INQUIRER_AGENT = "inquirer_agent"
    DATASET_PREPARER = "dataset_preparer"
    ML_AGENT = "ml_agent"
    REPORT_AGENT = "report_agent"
    
    # Signal processing operators
    EXPAND_OPERATORS = "expand_operators"
    TRANSFORM_OPERATORS = "transform_operators"
    AGGREGATE_OPERATORS = "aggregate_operators"
    DECISION_OPERATORS = "decision_operators"
    MULTIVARIABLE_OPERATORS = "multivariable_operators"
    
    # State management
    PHM_STATE = "phm_state"
    DAG_STATE = "dag_state"
    NODE_TYPES = "node_types"


@dataclass
class ConceptMapping:
    """Maps tutorial concepts to production components"""
    tutorial_concept: TutorialConcept
    production_component: ProductionComponent
    description: str
    learning_objectives: List[str]
    production_details: Dict[str, Any]


class TutorialBridge:
    """
    Bridge between tutorial concepts and production PHMGA system.
    Provides educational context for production components.
    """
    
    def __init__(self):
        self._concept_mappings = self._build_concept_mappings()
        self._production_registry = self._build_production_registry()
        
    def _build_concept_mappings(self) -> List[ConceptMapping]:
        """Build mappings between tutorial concepts and production components"""
        
        return [
            # Part 1: LLM Foundations → Production LLM Integration
            ConceptMapping(
                tutorial_concept=TutorialConcept.LLM_PROVIDERS,
                production_component=ProductionComponent.PLAN_AGENT,
                description="LLM provider configuration maps to intelligent planning agent",
                learning_objectives=[
                    "Understand LLM integration in production systems",
                    "See how LLM providers enable intelligent planning",
                    "Learn about model selection for different tasks"
                ],
                production_details={
                    "component_path": "src.agents.plan_agent",
                    "llm_usage": "Generates signal processing plans using configured LLM",
                    "configuration": "PHMState.llm_provider and PHMState.llm_model",
                    "tutorial_connection": "Part 1 LLM configuration directly controls agent behavior"
                }
            ),
            
            ConceptMapping(
                tutorial_concept=TutorialConcept.RESEARCH_LLM,
                production_component=ProductionComponent.INQUIRER_AGENT,
                description="Research LLM capabilities manifest in similarity analysis",
                learning_objectives=[
                    "See research LLM applied to signal analysis",
                    "Understand intelligent similarity computation",
                    "Learn about LLM-guided feature extraction"
                ],
                production_details={
                    "component_path": "src.agents.inquirer_agent",
                    "llm_usage": "Analyzes signal similarities using research-grade LLM",
                    "metrics": ["cosine", "euclidean", "dtw"],
                    "tutorial_connection": "Part 1 research LLM enables sophisticated signal understanding"
                }
            ),
            
            # Part 2: Multi-Agent Router → Production Multi-Agent Workflow
            ConceptMapping(
                tutorial_concept=TutorialConcept.AGENT_ROUTER,
                production_component=ProductionComponent.BUILDER_GRAPH,
                description="Agent routing concepts realized in LangGraph workflow",
                learning_objectives=[
                    "See agent routing in real workflow orchestration",
                    "Understand state-based agent coordination", 
                    "Learn about conditional agent execution"
                ],
                production_details={
                    "component_path": "src.phm_outer_graph.build_builder_graph",
                    "workflow": "plan → execute → reflect (conditional loop)",
                    "routing_logic": "Conditional edges based on needs_revision flag",
                    "tutorial_connection": "Part 2 routing logic implemented as LangGraph conditions"
                }
            ),
            
            ConceptMapping(
                tutorial_concept=TutorialConcept.PARALLEL_EXECUTION,
                production_component=ProductionComponent.EXECUTOR_GRAPH,
                description="Parallel execution patterns in analysis pipeline", 
                learning_objectives=[
                    "Understand parallel processing in signal analysis",
                    "Learn about pipeline parallelization",
                    "See concurrent agent operations"
                ],
                production_details={
                    "component_path": "src.phm_outer_graph.build_executor_graph",
                    "workflow": "inquire → prepare → train → report",
                    "parallel_aspects": "Multiple signal processing, ML model training",
                    "tutorial_connection": "Part 2 parallel concepts enable efficient analysis pipeline"
                }
            ),
            
            # Part 3: Research Integration → Production Knowledge Integration
            ConceptMapping(
                tutorial_concept=TutorialConcept.RESEARCH_WORKFLOW,
                production_component=ProductionComponent.REFLECT_AGENT,
                description="Research workflow principles guide reflection and quality assessment",
                learning_objectives=[
                    "See research methodology in quality assessment",
                    "Understand iterative refinement",
                    "Learn about evidence-based decision making"
                ],
                production_details={
                    "component_path": "src.agents.reflect_agent",
                    "research_aspects": "Quality assessment, depth analysis, refinement recommendations",
                    "decision_making": "Evidence-based continuation or stopping",
                    "tutorial_connection": "Part 3 research iteration enables DAG quality control"
                }
            ),
            
            ConceptMapping(
                tutorial_concept=TutorialConcept.KNOWLEDGE_SYNTHESIS,
                production_component=ProductionComponent.REPORT_AGENT,
                description="Knowledge synthesis generates comprehensive analysis reports",
                learning_objectives=[
                    "Understand knowledge integration in reporting",
                    "See synthesis of analysis results",
                    "Learn about structured knowledge presentation"
                ],
                production_details={
                    "component_path": "src.agents.report_agent",
                    "synthesis_inputs": "DAG results, ML outputs, similarity analysis",
                    "output_format": "Structured reports with insights and recommendations",
                    "tutorial_connection": "Part 3 synthesis principles create comprehensive reports"
                }
            ),
            
            # Part 4: DAG Architecture → Production Signal Processing DAG
            ConceptMapping(
                tutorial_concept=TutorialConcept.DAG_STRUCTURE,
                production_component=ProductionComponent.DAG_STATE,
                description="DAG architectural concepts implemented in production topology",
                learning_objectives=[
                    "See DAG theory in production implementation", 
                    "Understand node relationships and dependencies",
                    "Learn about DAG evolution and optimization"
                ],
                production_details={
                    "component_path": "src.states.phm_states.DAGState",
                    "topology_management": "Nodes, edges, leaves tracking",
                    "immutable_updates": "Safe state transitions",
                    "tutorial_connection": "Part 4 DAG concepts enable reliable signal processing"
                }
            ),
            
            ConceptMapping(
                tutorial_concept=TutorialConcept.SIGNAL_PROCESSING,
                production_component=ProductionComponent.TRANSFORM_OPERATORS,
                description="Signal processing concepts realized as registered operators",
                learning_objectives=[
                    "Understand operator-based signal processing",
                    "See modular processing pipeline design",
                    "Learn about operator composition and chaining"
                ],
                production_details={
                    "component_path": "src.tools.signal_processing_schemas",
                    "operator_registry": "Dynamic operator discovery and instantiation", 
                    "operator_types": "EXPAND, TRANSFORM, AGGREGATE, DECISION, MultiVariable",
                    "tutorial_connection": "Part 4 signal processing enables intelligent operator selection"
                }
            ),
            
            ConceptMapping(
                tutorial_concept=TutorialConcept.NODE_RELATIONSHIPS,
                production_component=ProductionComponent.NODE_TYPES,
                description="Node relationship concepts in production node hierarchy",
                learning_objectives=[
                    "Understand typed node relationships",
                    "See parent-child dependency management", 
                    "Learn about node metadata and results"
                ],
                production_details={
                    "component_path": "src.states.phm_states (InputData, ProcessedData, etc.)",
                    "node_hierarchy": "InputData → ProcessedData → SimilarityNode → DataSetNode → OutputNode",
                    "relationship_tracking": "Parent IDs, dependency validation",
                    "tutorial_connection": "Part 4 relationships enable complex signal processing flows"
                }
            )
        ]
    
    def _build_production_registry(self) -> Dict[ProductionComponent, Dict[str, Any]]:
        """Build registry of production components with their details"""
        
        return {
            ProductionComponent.BUILDER_GRAPH: {
                "function": "src.phm_outer_graph.build_builder_graph",
                "purpose": "Constructs signal processing DAG through iterative planning",
                "workflow": ["plan", "execute", "reflect"],
                "agents_involved": ["plan_agent", "execute_agent", "reflect_agent"]
            },
            
            ProductionComponent.EXECUTOR_GRAPH: {
                "function": "src.phm_outer_graph.build_executor_graph", 
                "purpose": "Executes analysis on completed DAG",
                "workflow": ["inquire", "prepare", "train", "report"],
                "agents_involved": ["inquirer_agent", "dataset_preparer_agent", "shallow_ml_agent", "report_agent"]
            },
            
            ProductionComponent.EXPAND_OPERATORS: {
                "registry_category": "EXPAND",
                "examples": self._get_operators_by_type(ExpandOp),
                "purpose": "Expand signal representations (windowing, segmentation)",
                "tutorial_connection": "Enable detailed signal analysis"
            },
            
            ProductionComponent.TRANSFORM_OPERATORS: {
                "registry_category": "TRANSFORM",
                "examples": self._get_operators_by_type(TransformOp),
                "purpose": "Transform signals (FFT, wavelets, filters)",
                "tutorial_connection": "Core signal processing transformations"
            },
            
            ProductionComponent.AGGREGATE_OPERATORS: {
                "registry_category": "AGGREGATE",
                "examples": self._get_operators_by_type(AggregateOp),
                "purpose": "Aggregate features (statistics, summaries)",
                "tutorial_connection": "Feature extraction for ML"
            },
            
            ProductionComponent.DECISION_OPERATORS: {
                "registry_category": "DECISION", 
                "examples": self._get_operators_by_type(DecisionOp),
                "purpose": "Make processing decisions (thresholding, classification)",
                "tutorial_connection": "Intelligent processing decisions"
            }
        }
    
    def _get_operators_by_type(self, operator_type: Type[PHMOperator]) -> List[str]:
        """Get operator names by type from registry"""
        return [
            name for name, op_class in OP_REGISTRY.items()
            if issubclass(op_class, operator_type)
        ]
    
    def get_concept_mapping(self, tutorial_concept: TutorialConcept) -> Optional[ConceptMapping]:
        """Get mapping for a specific tutorial concept"""
        for mapping in self._concept_mappings:
            if mapping.tutorial_concept == tutorial_concept:
                return mapping
        return None
    
    def get_production_details(self, production_component: ProductionComponent) -> Dict[str, Any]:
        """Get details for a production component"""
        return self._production_registry.get(production_component, {})
    
    def get_learning_path(self) -> List[Dict[str, Any]]:
        """Get structured learning path from tutorial concepts to production"""
        
        learning_path = []
        
        # Group by tutorial parts
        part_mappings = {
            "Part 1 - LLM Foundations": [m for m in self._concept_mappings if "LLM" in m.tutorial_concept.value or "RESEARCH" in m.tutorial_concept.value],
            "Part 2 - Multi-Agent Router": [m for m in self._concept_mappings if "AGENT" in m.tutorial_concept.value or "PARALLEL" in m.tutorial_concept.value],
            "Part 3 - Research Integration": [m for m in self._concept_mappings if "RESEARCH_WORKFLOW" in m.tutorial_concept.value or "KNOWLEDGE" in m.tutorial_concept.value],
            "Part 4 - DAG Architecture": [m for m in self._concept_mappings if "DAG" in m.tutorial_concept.value or "SIGNAL" in m.tutorial_concept.value or "NODE" in m.tutorial_concept.value]
        }
        
        for part_name, mappings in part_mappings.items():
            part_info = {
                "part": part_name,
                "concepts": [],
                "production_components": [],
                "key_learnings": []
            }
            
            for mapping in mappings:
                part_info["concepts"].append({
                    "tutorial_concept": mapping.tutorial_concept.value,
                    "production_component": mapping.production_component.value,
                    "description": mapping.description,
                    "objectives": mapping.learning_objectives
                })
                
                part_info["production_components"].append(mapping.production_component.value)
                part_info["key_learnings"].extend(mapping.learning_objectives)
            
            learning_path.append(part_info)
        
        return learning_path
    
    def generate_concept_explanation(self, tutorial_concept: TutorialConcept) -> str:
        """Generate educational explanation connecting tutorial to production"""
        
        mapping = self.get_concept_mapping(tutorial_concept)
        if not mapping:
            return f"No mapping found for {tutorial_concept.value}"
        
        production_details = self.get_production_details(mapping.production_component)
        
        explanation = f"""
🎓 Tutorial Concept: {tutorial_concept.value.replace('_', ' ').title()}
🏭 Production Component: {mapping.production_component.value.replace('_', ' ').title()}

📖 Connection:
{mapping.description}

🎯 Learning Objectives:
{chr(10).join(f"   • {obj}" for obj in mapping.learning_objectives)}

🔧 Production Details:
   • Component: {mapping.production_details.get('component_path', 'N/A')}
   • Purpose: {production_details.get('purpose', mapping.production_details.get('llm_usage', 'N/A'))}
   • Tutorial Connection: {mapping.production_details.get('tutorial_connection', 'Direct mapping')}

💡 Key Insight:
The {tutorial_concept.value} concept from the tutorial directly enables the {mapping.production_component.value} in the production system, demonstrating how educational concepts translate to real-world implementations.
        """.strip()
        
        return explanation
    
    def get_operator_tutorial_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed mapping of signal processing operators to tutorial concepts"""
        
        operator_mapping = {}
        
        for op_name, op_class in OP_REGISTRY.items():
            # Determine operator category
            if issubclass(op_class, ExpandOp):
                category = "EXPAND"
                tutorial_concept = "Signal segmentation and windowing from Part 4"
            elif issubclass(op_class, TransformOp):
                category = "TRANSFORM" 
                tutorial_concept = "Signal transformation and feature extraction from Part 4"
            elif issubclass(op_class, AggregateOp):
                category = "AGGREGATE"
                tutorial_concept = "Feature aggregation and summarization from Part 4"
            elif issubclass(op_class, DecisionOp):
                category = "DECISION"
                tutorial_concept = "Intelligent decision making from Parts 2-3"
            elif issubclass(op_class, MultiVariableOp):
                category = "MULTIVARIABLE"
                tutorial_concept = "Multi-signal analysis from Part 4"
            else:
                category = "UNKNOWN"
                tutorial_concept = "General signal processing"
            
            operator_mapping[op_name] = {
                "category": category,
                "class": op_class.__name__,
                "tutorial_concept": tutorial_concept,
                "description": getattr(op_class, 'description', 'Signal processing operator'),
                "educational_purpose": f"Demonstrates {category.lower()} operations in production system"
            }
        
        return operator_mapping


def create_tutorial_bridge() -> TutorialBridge:
    """Create and return a tutorial bridge instance"""
    return TutorialBridge()


def demonstrate_concept_mapping():
    """Demonstrate the tutorial concept mapping"""
    
    print("🌉 Tutorial Bridge - Concept Mapping Demonstration")
    print("=" * 60)
    
    bridge = create_tutorial_bridge()
    
    print("\n📚 Learning Path Overview:")
    learning_path = bridge.get_learning_path()
    
    for i, part in enumerate(learning_path, 1):
        print(f"\n{i}. {part['part']}")
        print(f"   Concepts: {len(part['concepts'])}")
        print(f"   Production Components: {len(set(part['production_components']))}")
        print(f"   Key Learnings: {len(set(part['key_learnings']))}")
    
    print(f"\n🔧 Available Signal Processing Operators: {len(OP_REGISTRY)}")
    operator_mapping = bridge.get_operator_tutorial_mapping()
    
    categories = {}
    for op_name, op_info in operator_mapping.items():
        category = op_info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(op_name)
    
    for category, operators in categories.items():
        print(f"   {category}: {len(operators)} operators")
    
    print(f"\n💡 Tutorial Bridge successfully connects {len(bridge._concept_mappings)} tutorial concepts to production components!")
    
    return bridge


if __name__ == "__main__":
    demonstrate_concept_mapping()