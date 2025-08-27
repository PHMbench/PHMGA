"""
Agent Fundamentals for Research Applications

This module demonstrates core agent concepts using research scenarios,
comparing traditional hardcoded approaches with intelligent LLM-based agents.
"""

import re
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class TaskComplexity(Enum):
    """Classification of task complexity for agent selection"""
    SIMPLE = "simple"      # Rule-based approaches work well
    MODERATE = "moderate"  # Hybrid approach recommended
    COMPLEX = "complex"    # LLM agents necessary


@dataclass
class ProcessingResult:
    """Standard result structure for both hardcoded and agent approaches"""
    success: bool
    output: str
    confidence: float  # 0.0 to 1.0
    processing_time: float
    approach_used: str
    metadata: Dict[str, Any]


class BaseProcessor(ABC):
    """Abstract base class for all processing approaches"""
    
    def __init__(self, name: str):
        self.name = name
        self.processing_history: List[ProcessingResult] = []
    
    @abstractmethod
    def process(self, input_text: str, **kwargs) -> ProcessingResult:
        """Process input and return structured result"""
        pass
    
    def get_success_rate(self) -> float:
        """Calculate success rate from processing history"""
        if not self.processing_history:
            return 0.0
        successful = sum(1 for result in self.processing_history if result.success)
        return successful / len(self.processing_history)
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence from successful processes"""
        successful_results = [r for r in self.processing_history if r.success]
        if not successful_results:
            return 0.0
        return sum(r.confidence for r in successful_results) / len(successful_results)


class HardcodedMathProcessor(BaseProcessor):
    """
    Traditional hardcoded approach using regex patterns.
    
    Pros: Fast, predictable, no API dependencies
    Cons: Limited patterns, brittle, requires manual updates
    """
    
    def __init__(self):
        super().__init__("Hardcoded Math Processor")
        
        # Define conversion patterns
        self.patterns = {
            # Basic mathematical functions
            r'np\.sqrt\(([^)]+)\)': r'\\sqrt{\1}',
            r'np\.pow\(([^,]+),\s*([^)]+)\)': r'\1^{\2}',
            r'np\.exp\(([^)]+)\)': r'e^{\1}',
            r'np\.log\(([^)]+)\)': r'\\ln(\1)',
            r'np\.sin\(([^)]+)\)': r'\\sin(\1)',
            r'np\.cos\(([^)]+)\)': r'\\cos(\1)',
            r'np\.tan\(([^)]+)\)': r'\\tan(\1)',
            
            # Power operations
            r'([a-zA-Z_][a-zA-Z0-9_]*)\*\*(\d+)': r'\1^{\2}',
            r'\*\*': '^',
            
            # Fractions (simple cases)
            r'([a-zA-Z0-9_]+)\s*/\s*([a-zA-Z0-9_]+)': r'\\frac{\1}{\2}',
            
            # Greek letters (common variable names)
            r'\\balpha\\b': r'\\alpha',
            r'\\bbeta\\b': r'\\beta',
            r'\\bgamma\\b': r'\\gamma',
            r'\\btheta\\b': r'\\theta',
            r'\\bsigma\\b': r'\\sigma',
            r'\\bmu\\b': r'\\mu',
            
            # Arrays/matrices (very basic)
            r'np\.array\(\[\[([^\]]+)\]\]\)': r'\\begin{pmatrix}\1\\end{pmatrix}',
        }
        
        # Known limitations
        self.limitations = [
            "Cannot handle nested function calls",
            "Limited to predefined patterns",
            "Struggles with complex expressions",
            "No contextual understanding",
            "Requires manual pattern updates"
        ]
    
    def process(self, input_text: str, **kwargs) -> ProcessingResult:
        """Process mathematical expressions using regex patterns"""
        import time
        start_time = time.time()
        
        original_text = input_text
        converted_text = input_text
        patterns_matched = 0
        
        try:
            # Apply all patterns sequentially
            for pattern, replacement in self.patterns.items():
                matches = re.findall(pattern, converted_text)
                if matches:
                    patterns_matched += len(matches)
                    converted_text = re.sub(pattern, replacement, converted_text)
            
            # Simple confidence calculation based on patterns matched
            confidence = min(1.0, patterns_matched / 3)  # Arbitrary scaling
            
            # Success if any patterns matched
            success = patterns_matched > 0
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                success=success,
                output=f"${converted_text}$" if success else original_text,
                confidence=confidence,
                processing_time=processing_time,
                approach_used="hardcoded_regex",
                metadata={
                    "patterns_matched": patterns_matched,
                    "limitations": self.limitations,
                    "original_input": original_text
                }
            )
            
            self.processing_history.append(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            result = ProcessingResult(
                success=False,
                output=original_text,
                confidence=0.0,
                processing_time=processing_time,
                approach_used="hardcoded_regex",
                metadata={
                    "error": str(e),
                    "patterns_matched": 0,
                    "original_input": original_text
                }
            )
            
            self.processing_history.append(result)
            return result


class LLMAgent(BaseProcessor):
    """
    LLM-based intelligent agent for research tasks.
    
    Pros: Contextual understanding, handles complex cases, adaptable
    Cons: API dependencies, variable response time, requires prompt engineering
    """
    
    def __init__(self, llm, task_description: str = "mathematical expression conversion"):
        super().__init__("LLM Agent")
        self.llm = llm
        self.task_description = task_description
        
        # Core capabilities
        self.capabilities = [
            "Contextual understanding",
            "Handles complex nested expressions", 
            "Adapts to new patterns",
            "Provides explanations",
            "Learns from context"
        ]
        
        # System prompt for mathematical conversion
        self.system_prompt = """You are a research assistant specialized in converting code expressions to LaTeX format.

Your task: Convert programming code (especially Python/NumPy) to publication-ready LaTeX mathematical notation.

Guidelines:
1. Convert mathematical functions to proper LaTeX commands
2. Use appropriate mathematical symbols and notation
3. Handle nested expressions correctly
4. Maintain mathematical accuracy
5. Return only the LaTeX expression without explanations

Examples:
- np.sqrt(x**2 + y**2) → \\sqrt{x^2 + y^2}
- np.sin(theta) * np.cos(phi) → \\sin(\\theta) \\cos(\\phi)
- x / (y + z) → \\frac{x}{y + z}

Respond with only the LaTeX expression, enclosed in dollar signs for inline math."""
    
    def process(self, input_text: str, **kwargs) -> ProcessingResult:
        """Process using LLM with structured prompting"""
        import time
        start_time = time.time()
        
        try:
            # Construct the prompt
            user_prompt = f"Convert this expression to LaTeX: {input_text}"
            
            # Create messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM
            response = self.llm.invoke(messages)
            output_text = response.content.strip()
            
            # Calculate confidence based on response characteristics
            confidence = self._assess_confidence(input_text, output_text)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                success=True,
                output=output_text,
                confidence=confidence,
                processing_time=processing_time,
                approach_used="llm_agent",
                metadata={
                    "model": getattr(self.llm, 'model', 'unknown'),
                    "capabilities": self.capabilities,
                    "original_input": input_text,
                    "system_prompt_length": len(self.system_prompt)
                }
            )
            
            self.processing_history.append(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            result = ProcessingResult(
                success=False,
                output=input_text,
                confidence=0.0,
                processing_time=processing_time,
                approach_used="llm_agent",
                metadata={
                    "error": str(e),
                    "original_input": input_text
                }
            )
            
            self.processing_history.append(result)
            return result
    
    def _assess_confidence(self, input_text: str, output_text: str) -> float:
        """Assess confidence based on output characteristics"""
        confidence = 0.5  # Base confidence
        
        # Check if output contains LaTeX
        if '\\' in output_text and ('{' in output_text or '}' in output_text):
            confidence += 0.3
        
        # Check if properly formatted with dollar signs
        if output_text.startswith('$') and output_text.endswith('$'):
            confidence += 0.1
        
        # Check if input length is reasonable (not too complex)
        if len(input_text) < 100:
            confidence += 0.1
        
        return min(1.0, confidence)


class HybridProcessor(BaseProcessor):
    """
    Hybrid approach combining hardcoded patterns with LLM intelligence.
    
    Strategy: Try hardcoded first for speed, fall back to LLM for complex cases
    """
    
    def __init__(self, llm):
        super().__init__("Hybrid Processor")
        self.hardcoded = HardcodedMathProcessor()
        self.llm_agent = LLMAgent(llm)
    
    def process(self, input_text: str, **kwargs) -> ProcessingResult:
        """Process using hybrid approach"""
        import time
        start_time = time.time()
        
        # First, try hardcoded approach
        hardcoded_result = self.hardcoded.process(input_text)
        
        # If hardcoded succeeds with reasonable confidence, use it
        if hardcoded_result.success and hardcoded_result.confidence > 0.6:
            # Update metadata to indicate hybrid approach
            hardcoded_result.approach_used = "hybrid_hardcoded"
            hardcoded_result.metadata["fallback_available"] = True
            
            self.processing_history.append(hardcoded_result)
            return hardcoded_result
        
        # Otherwise, use LLM agent
        llm_result = self.llm_agent.process(input_text)
        llm_result.approach_used = "hybrid_llm"
        llm_result.metadata["hardcoded_attempted"] = True
        llm_result.metadata["hardcoded_confidence"] = hardcoded_result.confidence
        
        total_time = time.time() - start_time
        llm_result.processing_time = total_time
        
        self.processing_history.append(llm_result)
        return llm_result


def compare_approaches(test_cases: List[str], llm) -> Dict[str, Any]:
    """
    Compare hardcoded, LLM, and hybrid approaches on test cases.
    
    Args:
        test_cases: List of mathematical expressions to test
        llm: LLM instance for agent-based processing
        
    Returns:
        Comparison results with performance metrics
    """
    processors = {
        "hardcoded": HardcodedMathProcessor(),
        "llm_agent": LLMAgent(llm),
        "hybrid": HybridProcessor(llm)
    }
    
    results = {}
    
    for name, processor in processors.items():
        print(f"\n🧪 Testing {name.upper()} approach...")
        
        case_results = []
        for i, test_case in enumerate(test_cases):
            print(f"  Case {i+1}: {test_case[:50]}...")
            result = processor.process(test_case)
            case_results.append(result)
        
        # Calculate metrics
        success_rate = processor.get_success_rate()
        avg_confidence = processor.get_average_confidence()
        avg_time = sum(r.processing_time for r in case_results) / len(case_results)
        
        results[name] = {
            "processor": processor,
            "results": case_results,
            "success_rate": success_rate,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_time
        }
        
        print(f"    Success Rate: {success_rate:.2%}")
        print(f"    Avg Confidence: {avg_confidence:.2f}")
        print(f"    Avg Time: {avg_time:.3f}s")
    
    return results


def demonstrate_agent_evolution():
    """Demonstrate the evolution from hardcoded to intelligent agents"""
    
    print("🎓 AGENT EVOLUTION DEMONSTRATION")
    print("=" * 50)
    
    print("\n1. HARDCODED APPROACH (Traditional)")
    print("   - Fast execution")
    print("   - Predictable patterns")
    print("   - Limited to known cases")
    print("   - Requires manual updates")
    
    print("\n2. LLM AGENT APPROACH (Modern)")
    print("   - Contextual understanding")
    print("   - Handles complex cases")
    print("   - Learns from examples")
    print("   - API dependency")
    
    print("\n3. HYBRID APPROACH (Optimal)")
    print("   - Best of both worlds")
    print("   - Fast for simple cases")
    print("   - Intelligent for complex cases")
    print("   - Graceful fallback strategy")
    
    print("\n🎯 Research Application:")
    print("Choose approach based on your specific needs:")
    print("• Speed critical + simple patterns → Hardcoded")
    print("• Complex understanding required → LLM Agent")
    print("• Production system → Hybrid")


if __name__ == "__main__":
    # Demonstration
    demonstrate_agent_evolution()
    
    # Test cases for comparison
    test_cases = [
        "np.sqrt(x**2 + y**2)",
        "np.sin(theta) * np.cos(phi)",
        "x / (y + z)",
        "np.exp(-0.5 * ((x - mu) / sigma)**2)",
        "np.linalg.norm(vector)",  # Complex case
    ]
    
    print("\n" + "="*50)
    print("📊 COMPARISON RESULTS")
    print("="*50)
    
    # Note: Actual comparison would require LLM instance
    print("To run full comparison, provide an LLM instance:")
    print("```python")
    print("from llm_providers import create_research_llm")
    print("llm = create_research_llm('google')")
    print("results = compare_approaches(test_cases, llm)")
    print("```")