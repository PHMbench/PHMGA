"""
Part 5: Complete PHMGA System - Production Integration Tutorial

Educational integration of all tutorial components with the production PHMGA system.
"""

# Import only core tutorial components that don't require external dependencies
try:
    from .tutorial_bridge import TutorialBridge, TutorialConcept, ProductionComponent, create_tutorial_bridge
    from .educational_wrappers import EducationalLogger, EducationalConfig, create_educational_system
    from .demo_configurations import DemoConfigurationManager, LearningLevel, AnalysisType, create_demo_manager
    from .visualization_tools import IntegratedVisualizer, DAGVisualizer, SignalVisualizer, PerformanceVisualizer
    from .operator_playground import OperatorPlayground, SignalGenerator, create_operator_playground
except ImportError as e:
    # Graceful fallback if some dependencies are missing
    print(f"⚠️ Some tutorial components couldn't be imported: {e}")

# Production-integrated components (may require external dependencies)
_production_integrated = []
try:
    from .phmga_system import PHMGASystem, PHMGAConfig, create_tutorial_system
    _production_integrated.extend(['PHMGASystem', 'PHMGAConfig', 'create_tutorial_system'])
except ImportError:
    pass

try:
    from .case1_tutorial import Case1Tutorial, run_case1_tutorial, demonstrate_case1_concepts  
    _production_integrated.extend(['Case1Tutorial', 'run_case1_tutorial', 'demonstrate_case1_concepts'])
except ImportError:
    pass

# Core tutorial components (minimal dependencies)
__all__ = [
    'TutorialBridge', 'TutorialConcept', 'ProductionComponent', 'create_tutorial_bridge',
    'EducationalLogger', 'EducationalConfig', 'create_educational_system',
    'DemoConfigurationManager', 'LearningLevel', 'AnalysisType', 'create_demo_manager',
    'IntegratedVisualizer', 'DAGVisualizer', 'SignalVisualizer', 'PerformanceVisualizer',
    'OperatorPlayground', 'SignalGenerator', 'create_operator_playground'
] + _production_integrated