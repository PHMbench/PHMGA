"""
Code to LaTeX Conversion Agent

Specialized agent for converting programming code (Python, JavaScript, etc.)
to publication-ready LaTeX mathematical notation for research papers.
"""

import re
import ast
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from agent_basics import LLMAgent, ProcessingResult


class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    MATLAB = "matlab"
    R = "r"
    AUTO = "auto"  # Attempt to detect


class ConversionType(Enum):
    """Types of LaTeX conversion"""
    INLINE_MATH = "inline"      # $...$
    DISPLAY_MATH = "display"    # $$...$$
    EQUATION = "equation"       # \begin{equation}
    ALGORITHM = "algorithm"     # \begin{algorithm}
    MATRIX = "matrix"          # \begin{pmatrix}


@dataclass
class ConversionConfig:
    """Configuration for code-to-LaTeX conversion"""
    language: CodeLanguage = CodeLanguage.AUTO
    conversion_type: ConversionType = ConversionType.INLINE_MATH
    include_comments: bool = True
    preserve_variable_names: bool = True
    use_greek_letters: bool = True  # Convert alpha, beta, etc. to Greek
    format_matrices: bool = True
    include_algorithmic: bool = False  # For algorithm blocks


class CodeToLatexAgent(LLMAgent):
    """
    Specialized agent for converting code to LaTeX mathematical notation.
    
    This agent extends the base LLM agent with specific knowledge about
    mathematical notation conventions used in academic publications.
    """
    
    def __init__(self, llm, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        
        # Initialize with specialized prompt
        task_desc = f"code to LaTeX conversion ({self.config.language.value})"
        super().__init__(llm, task_desc)
        
        # Update system prompt for code conversion
        self.system_prompt = self._build_system_prompt()
        
        # Common mathematical functions mapping
        self.function_mappings = {
            # Trigonometric
            'sin': '\\sin',
            'cos': '\\cos', 
            'tan': '\\tan',
            'asin': '\\arcsin',
            'acos': '\\arccos',
            'atan': '\\arctan',
            
            # Hyperbolic
            'sinh': '\\sinh',
            'cosh': '\\cosh',
            'tanh': '\\tanh',
            
            # Logarithmic
            'log': '\\log',
            'ln': '\\ln',
            'log10': '\\log_{10}',
            'log2': '\\log_2',
            
            # Exponential
            'exp': '\\exp',
            'sqrt': '\\sqrt',
            'pow': '^',
            
            # Statistical
            'mean': '\\mu',
            'std': '\\sigma',
            'var': '\\sigma^2',
            
            # Linear algebra
            'dot': '\\cdot',
            'cross': '\\times',
            'norm': '\\|\\cdot\\|',
            'transpose': '^T',
            'inverse': '^{-1}',
        }
        
        # Greek letter variable mapping
        self.greek_mapping = {
            'alpha': '\\alpha',
            'beta': '\\beta', 
            'gamma': '\\gamma',
            'delta': '\\delta',
            'epsilon': '\\epsilon',
            'theta': '\\theta',
            'lambda': '\\lambda',
            'mu': '\\mu',
            'sigma': '\\sigma',
            'phi': '\\phi',
            'psi': '\\psi',
            'omega': '\\omega',
            'pi': '\\pi',
            'tau': '\\tau'
        }
    
    def _build_system_prompt(self) -> str:
        """Build specialized system prompt for code conversion"""
        
        base_prompt = """You are a research assistant specialized in converting programming code to LaTeX mathematical notation for academic publications.

Your expertise includes:
- Mathematical function translation
- Proper LaTeX formatting conventions
- Variable notation standards
- Matrix and vector representations
- Algorithm pseudocode formatting

Guidelines:
1. Convert code expressions to mathematically correct LaTeX
2. Use appropriate mathematical symbols and operators
3. Handle nested expressions and function calls
4. Maintain variable meaning and context
5. Follow academic publication standards
6. Use proper LaTeX environments when needed"""

        # Add language-specific instructions
        if self.config.language == CodeLanguage.PYTHON:
            base_prompt += """

Python-specific conversions:
- np.sqrt(x) → \\sqrt{x}
- np.sin(theta) → \\sin(\\theta)
- x**2 → x^2
- x/y → \\frac{x}{y}
- np.array([[a,b],[c,d]]) → \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}
- np.dot(a,b) → a \\cdot b
- np.linalg.norm(x) → \\|x\\|"""

        elif self.config.language == CodeLanguage.MATLAB:
            base_prompt += """

MATLAB-specific conversions:
- sqrt(x) → \\sqrt{x}
- sin(theta) → \\sin(\\theta)
- x.^2 → x^2
- x./y → \\frac{x}{y}
- [a b; c d] → \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}"""

        # Add conversion type instructions
        if self.config.conversion_type == ConversionType.DISPLAY_MATH:
            base_prompt += "\n\nFormat as display math: $$...$$"
        elif self.config.conversion_type == ConversionType.EQUATION:
            base_prompt += "\n\nFormat as numbered equation: \\begin{equation}...\\end{equation}"
        elif self.config.conversion_type == ConversionType.ALGORITHM:
            base_prompt += "\n\nFormat as algorithm block using algorithmic environment"
        else:
            base_prompt += "\n\nFormat as inline math: $...$"
        
        base_prompt += "\n\nRespond with only the LaTeX expression, properly formatted."
        
        return base_prompt
    
    def convert_expression(self, code: str, **kwargs) -> ProcessingResult:
        """
        Convert a single code expression to LaTeX.
        
        Args:
            code: Code expression to convert
            **kwargs: Additional conversion parameters
            
        Returns:
            ProcessingResult with LaTeX conversion
        """
        return self.process(code, **kwargs)
    
    def convert_algorithm(self, code: str, function_name: str = "") -> ProcessingResult:
        """
        Convert a code function/algorithm to LaTeX algorithmic format.
        
        Args:
            code: Function or algorithm code
            function_name: Name of the function/algorithm
            
        Returns:
            ProcessingResult with algorithmic LaTeX format
        """
        # Update config for algorithm conversion
        original_type = self.config.conversion_type
        self.config.conversion_type = ConversionType.ALGORITHM
        self.system_prompt = self._build_system_prompt()
        
        # Add context for algorithm conversion
        prompt_prefix = f"Convert this {self.config.language.value} function to LaTeX algorithmic format"
        if function_name:
            prompt_prefix += f" (Algorithm: {function_name})"
        prompt_prefix += ":\n\n"
        
        result = self.process(prompt_prefix + code)
        
        # Restore original config
        self.config.conversion_type = original_type
        self.system_prompt = self._build_system_prompt()
        
        return result
    
    def convert_matrix(self, code: str) -> ProcessingResult:
        """
        Convert matrix/array code to LaTeX matrix format.
        
        Args:
            code: Matrix or array definition
            
        Returns:
            ProcessingResult with matrix LaTeX format
        """
        original_type = self.config.conversion_type
        self.config.conversion_type = ConversionType.MATRIX
        
        result = self.process(f"Convert this matrix/array to LaTeX matrix format: {code}")
        
        self.config.conversion_type = original_type
        return result
    
    def batch_convert(self, 
                     code_snippets: List[str], 
                     conversion_types: Optional[List[ConversionType]] = None) -> List[ProcessingResult]:
        """
        Convert multiple code snippets to LaTeX.
        
        Args:
            code_snippets: List of code expressions
            conversion_types: Optional list of conversion types for each snippet
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for i, code in enumerate(code_snippets):
            # Use specific conversion type if provided
            if conversion_types and i < len(conversion_types):
                original_type = self.config.conversion_type
                self.config.conversion_type = conversion_types[i]
                self.system_prompt = self._build_system_prompt()
                
                result = self.convert_expression(code)
                
                # Restore original type
                self.config.conversion_type = original_type
                self.system_prompt = self._build_system_prompt()
            else:
                result = self.convert_expression(code)
            
            results.append(result)
        
        return results
    
    def detect_language(self, code: str) -> CodeLanguage:
        """
        Attempt to detect the programming language of the code.
        
        Args:
            code: Code snippet to analyze
            
        Returns:
            Detected CodeLanguage
        """
        code_lower = code.lower()
        
        # Python indicators
        if ('np.' in code or 'numpy' in code or 
            'import ' in code or 'def ' in code):
            return CodeLanguage.PYTHON
        
        # MATLAB indicators
        elif ('.m' in code or code.count(';') > code.count('\n') or
              any(func in code for func in ['function', 'end', 'fprintf'])):
            return CodeLanguage.MATLAB
        
        # JavaScript indicators
        elif ('function(' in code or 'var ' in code or 
              'let ' in code or 'const ' in code):
            return CodeLanguage.JAVASCRIPT
        
        # R indicators
        elif ('<-' in code or 'library(' in code or 
              any(func in code for func in ['c(', 'data.frame', 'summary'])):
            return CodeLanguage.R
        
        # Default to Python if unsure
        return CodeLanguage.PYTHON


def create_research_converter(llm, 
                            language: CodeLanguage = CodeLanguage.AUTO,
                            conversion_type: ConversionType = ConversionType.INLINE_MATH,
                            **config_kwargs) -> CodeToLatexAgent:
    """
    Factory function to create a code-to-LaTeX converter for research use.
    
    Args:
        llm: LLM instance
        language: Target programming language
        conversion_type: Type of LaTeX output desired
        **config_kwargs: Additional configuration options
        
    Returns:
        Configured CodeToLatexAgent
        
    Example:
        >>> llm = create_research_llm('google')
        >>> converter = create_research_converter(llm, language=CodeLanguage.PYTHON)
        >>> result = converter.convert_expression("np.sqrt(x**2 + y**2)")
    """
    config = ConversionConfig(
        language=language,
        conversion_type=conversion_type,
        **config_kwargs
    )
    
    return CodeToLatexAgent(llm, config)


def demonstrate_conversions():
    """Demonstrate various code-to-LaTeX conversion scenarios"""
    
    print("🔬 CODE-TO-LATEX CONVERSION EXAMPLES")
    print("=" * 50)
    
    examples = [
        {
            "category": "Mathematical Functions",
            "cases": [
                ("np.sqrt(x**2 + y**2)", "\\sqrt{x^2 + y^2}"),
                ("np.sin(theta) * np.cos(phi)", "\\sin(\\theta) \\cos(\\phi)"),
                ("np.exp(-0.5 * (x/sigma)**2)", "\\exp\\left(-\\frac{1}{2}\\left(\\frac{x}{\\sigma}\\right)^2\\right)")
            ]
        },
        {
            "category": "Linear Algebra",
            "cases": [
                ("np.dot(A, x)", "A \\cdot x"),
                ("np.linalg.norm(vector)", "\\|\\text{vector}\\|"),
                ("A.T", "A^T"),
                ("np.linalg.inv(A)", "A^{-1}")
            ]
        },
        {
            "category": "Statistical Functions",
            "cases": [
                ("np.mean(data)", "\\mu_{\\text{data}}"),
                ("np.std(samples)", "\\sigma_{\\text{samples}}"),
                ("scipy.stats.norm.pdf(x, mu, sigma)", "\\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}")
            ]
        }
    ]
    
    for example_group in examples:
        print(f"\n📊 {example_group['category']}:")
        for code, expected_latex in example_group['cases']:
            print(f"  Code: {code}")
            print(f"  LaTeX: ${expected_latex}$")
            print()
    
    print("🎯 Usage in Research Papers:")
    print("• Convert algorithm implementations to mathematical notation")
    print("• Document experimental procedures with proper formatting")
    print("• Transform statistical computations to publication standards")
    print("• Generate consistent mathematical notation across papers")


if __name__ == "__main__":
    demonstrate_conversions()
    
    print("\n" + "="*50)
    print("🧪 To test the converter with your LLM:")
    print("="*50)
    print("""
from llm_providers import create_research_llm
from code_to_latex import create_research_converter

# Create LLM and converter
llm = create_research_llm('google')  # or your preferred provider
converter = create_research_converter(llm, language=CodeLanguage.PYTHON)

# Convert expressions
result = converter.convert_expression("np.sqrt(x**2 + y**2)")
print(f"Result: {result.output}")

# Convert algorithm
algorithm_code = '''
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
'''
result = converter.convert_algorithm(algorithm_code, "Euclidean Distance")
print(f"Algorithm: {result.output}")
""")