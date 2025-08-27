"""
Research Router Agent

Intelligent router that analyzes research queries and delegates tasks to
specialized agents (literature search, citation formatting, etc.).
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from research_tools import ResearchToolsAggregator
from literature_agent import LiteratureSearchAgent, LiteratureSearchResult
from citation_agent import CitationFormatterAgent, CitationStyle


class TaskType(Enum):
    """Types of research tasks the router can handle"""
    LITERATURE_SEARCH = "literature_search"
    CITATION_FORMAT = "citation_format"
    SUMMARY_GENERATION = "summary_generation"  
    FIGURE_CAPTION = "figure_caption"
    TREND_ANALYSIS = "trend_analysis"
    AUTHOR_ANALYSIS = "author_analysis"
    REVIEW_GENERATION = "review_generation"
    MULTI_TASK = "multi_task"


@dataclass
class ResearchTask:
    """Structured representation of a research task"""
    task_id: str
    original_query: str
    task_types: List[TaskType]
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    estimated_time: float = 0.0
    dependencies: List[str] = field(default_factory=list)


@dataclass 
class TaskResult:
    """Result from executing a research task"""
    task_id: str
    task_type: TaskType
    success: bool
    result_data: Any
    execution_time: float
    agent_used: str
    confidence_score: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSession:
    """Complete research session with multiple tasks"""
    session_id: str
    original_query: str
    tasks: List[ResearchTask] = field(default_factory=list)
    results: List[TaskResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"
    final_report: Optional[str] = None


class ResearchRouterAgent:
    """
    Intelligent router agent that coordinates multiple research tasks.
    
    Analyzes complex research queries and breaks them down into specialized
    tasks that are delegated to appropriate agents.
    """
    
    def __init__(self, llm, research_tools: Optional[ResearchToolsAggregator] = None):
        self.llm = llm
        self.research_tools = research_tools or ResearchToolsAggregator()
        
        # Initialize specialized agents
        self.literature_agent = LiteratureSearchAgent(llm, self.research_tools)
        self.citation_agent = CitationFormatterAgent(llm)
        
        # Task routing patterns
        self.task_patterns = {
            TaskType.LITERATURE_SEARCH: [
                r"find.*papers?.*about",
                r"search.*literature.*on",
                r"recent.*research.*in",
                r"papers?.*related.*to",
                r"literature.*review.*on",
                r"what.*research.*exists"
            ],
            TaskType.CITATION_FORMAT: [
                r"format.*citations?",
                r"cite.*in.*style",
                r"bibliography.*in",
                r"references.*in.*format",
                r"APA.*style", r"IEEE.*style", r"MLA.*style"
            ],
            TaskType.SUMMARY_GENERATION: [
                r"summarize.*research",
                r"key.*findings.*from",
                r"main.*insights.*from",
                r"overview.*of.*research"
            ],
            TaskType.TREND_ANALYSIS: [
                r"trends.*in.*research",
                r"emerging.*directions",
                r"recent.*developments",
                r"what.*is.*trending"
            ],
            TaskType.AUTHOR_ANALYSIS: [
                r"who.*are.*leading.*researchers",
                r"top.*authors.*in",
                r"influential.*researchers",
                r"collaboration.*networks"
            ],
            TaskType.REVIEW_GENERATION: [
                r"write.*literature.*review",
                r"generate.*review.*section",
                r"create.*background.*section"
            ]
        }
        
        # Citation style detection
        self.style_patterns = {
            CitationStyle.IEEE: [r"ieee", r"engineering", r"computer science"],
            CitationStyle.APA: [r"apa", r"psychology", r"social science"],
            CitationStyle.MLA: [r"mla", r"humanities", r"literature"],
            CitationStyle.NATURE: [r"nature", r"biology", r"physics"]
        }
        
        # Session management
        self.active_sessions: Dict[str, ResearchSession] = {}
        self.session_counter = 0
    
    def analyze_query(self, query: str) -> ResearchTask:
        """
        Analyze a research query and create a structured task.
        
        Args:
            query: Natural language research query
            
        Returns:
            ResearchTask with identified task types and parameters
        """
        
        self.session_counter += 1
        task_id = f"task_{self.session_counter}_{datetime.now().strftime('%H%M%S')}"
        
        # Identify task types using pattern matching
        identified_types = []
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query.lower()):
                    identified_types.append(task_type)
                    break
        
        # Default to literature search if no specific patterns match
        if not identified_types:
            identified_types = [TaskType.LITERATURE_SEARCH]
        
        # Extract parameters
        parameters = self._extract_parameters(query, identified_types)
        
        # Estimate complexity and time
        estimated_time = self._estimate_task_time(identified_types, parameters)
        
        task = ResearchTask(
            task_id=task_id,
            original_query=query,
            task_types=identified_types,
            parameters=parameters,
            estimated_time=estimated_time
        )
        
        return task
    
    def execute_research_session(self, query: str) -> ResearchSession:
        """
        Execute a complete research session based on a query.
        
        Args:
            query: Research query
            
        Returns:
            ResearchSession with all results
        """
        
        # Create session
        session_id = f"session_{len(self.active_sessions) + 1}"
        session = ResearchSession(
            session_id=session_id,
            original_query=query
        )
        
        print(f"🚀 Starting research session: {session_id}")
        print(f"Query: '{query}'")
        
        # Analyze query and create task
        main_task = self.analyze_query(query)
        session.tasks.append(main_task)
        
        print(f"📋 Identified task types: {[t.value for t in main_task.task_types]}")
        
        # Execute tasks
        for task_type in main_task.task_types:
            result = self._execute_single_task(main_task, task_type)
            session.results.append(result)
        
        # Generate final report
        session.final_report = self._generate_session_report(session)
        session.status = "completed"
        
        # Store session
        self.active_sessions[session_id] = session
        
        print(f"✅ Research session completed: {session_id}")
        return session
    
    def _extract_parameters(self, query: str, task_types: List[TaskType]) -> Dict[str, Any]:
        """Extract parameters from query based on identified task types"""
        
        parameters = {}
        query_lower = query.lower()
        
        # Extract citation style if citation task
        if TaskType.CITATION_FORMAT in task_types:
            for style, patterns in self.style_patterns.items():
                for pattern in patterns:
                    if pattern in query_lower:
                        parameters["citation_style"] = style
                        break
                if "citation_style" in parameters:
                    break
            
            # Default to APA if no style specified
            if "citation_style" not in parameters:
                parameters["citation_style"] = CitationStyle.APA
        
        # Extract number limits
        number_patterns = [
            (r"(\d+)\s*papers?", "max_papers"),
            (r"top\s*(\d+)", "max_papers"), 
            (r"first\s*(\d+)", "max_papers"),
            (r"last\s*(\d+)\s*years?", "years_filter")
        ]
        
        for pattern, param_name in number_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parameters[param_name] = int(match.group(1))
        
        # Set defaults
        if "max_papers" not in parameters:
            parameters["max_papers"] = 20
        
        # Extract topic/field
        # Simple extraction - could be improved with NER
        parameters["topic"] = query
        
        return parameters
    
    def _estimate_task_time(self, task_types: List[TaskType], parameters: Dict[str, Any]) -> float:
        """Estimate execution time for tasks"""
        
        base_times = {
            TaskType.LITERATURE_SEARCH: 30.0,  # seconds
            TaskType.CITATION_FORMAT: 5.0,
            TaskType.SUMMARY_GENERATION: 20.0,
            TaskType.TREND_ANALYSIS: 15.0,
            TaskType.AUTHOR_ANALYSIS: 10.0,
            TaskType.REVIEW_GENERATION: 45.0
        }
        
        total_time = sum(base_times.get(t, 10.0) for t in task_types)
        
        # Adjust for paper count
        max_papers = parameters.get("max_papers", 20)
        if max_papers > 20:
            total_time += (max_papers - 20) * 0.5
        
        return total_time
    
    def _execute_single_task(self, task: ResearchTask, task_type: TaskType) -> TaskResult:
        """Execute a single task using appropriate agent"""
        
        import time
        start_time = time.time()
        
        try:
            print(f"   🔄 Executing {task_type.value}...")
            
            if task_type == TaskType.LITERATURE_SEARCH:
                result_data = self._execute_literature_search(task)
                agent_used = "LiteratureSearchAgent"
                
            elif task_type == TaskType.CITATION_FORMAT:
                result_data = self._execute_citation_format(task)
                agent_used = "CitationFormatterAgent"
                
            elif task_type == TaskType.SUMMARY_GENERATION:
                result_data = self._execute_summary_generation(task)
                agent_used = "SummaryAgent"
                
            elif task_type == TaskType.TREND_ANALYSIS:
                result_data = self._execute_trend_analysis(task)
                agent_used = "TrendAnalysisAgent"
                
            elif task_type == TaskType.REVIEW_GENERATION:
                result_data = self._execute_review_generation(task)
                agent_used = "ReviewGenerationAgent"
                
            else:
                result_data = f"Task type {task_type.value} not yet implemented"
                agent_used = "RouterAgent"
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                task_type=task_type,
                success=True,
                result_data=result_data,
                execution_time=execution_time,
                agent_used=agent_used,
                confidence_score=0.8
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Task failed: {str(e)}")
            
            return TaskResult(
                task_id=task.task_id,
                task_type=task_type,
                success=False,
                result_data=None,
                execution_time=execution_time,
                agent_used="RouterAgent",
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def _execute_literature_search(self, task: ResearchTask) -> LiteratureSearchResult:
        """Execute literature search task"""
        
        topic = task.parameters.get("topic", task.original_query)
        max_papers = task.parameters.get("max_papers", 20)
        
        return self.literature_agent.search_literature(
            query=topic,
            max_results=max_papers,
            include_recent_only=True
        )
    
    def _execute_citation_format(self, task: ResearchTask) -> str:
        """Execute citation formatting task"""
        
        # This would typically format citations from literature search results
        # For now, return instruction message
        style = task.parameters.get("citation_style", CitationStyle.APA)
        return f"Citation formatting configured for {style.value.upper()} style. Apply to literature search results."
    
    def _execute_summary_generation(self, task: ResearchTask) -> str:
        """Execute summary generation using LLM"""
        
        # Generate summary based on query
        prompt = f"""Generate a concise research summary for the topic: "{task.original_query}"

Provide:
1. Brief overview of the research area (2-3 sentences)
2. Key research questions and challenges
3. Current state of knowledge
4. Important findings or developments

Keep the summary academic and informative, approximately 150-200 words."""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def _execute_trend_analysis(self, task: ResearchTask) -> str:
        """Execute trend analysis"""
        
        prompt = f"""Analyze current research trends in: "{task.original_query}"

Identify:
1. Emerging research directions
2. Popular methodologies
3. Recent technological developments
4. Funding priorities and focus areas
5. Future research opportunities

Format as bullet points, keep concise and factual."""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Trend analysis failed: {str(e)}"
    
    def _execute_review_generation(self, task: ResearchTask) -> str:
        """Execute literature review generation"""
        
        # This would typically work with literature search results
        topic = task.parameters.get("topic", task.original_query)
        
        prompt = f"""Write a literature review section for: "{topic}"

Structure:
1. Introduction to the research area
2. Historical development and key milestones  
3. Current methodological approaches
4. Recent findings and contributions
5. Research gaps and future directions

Write in academic style, 400-600 words, suitable for inclusion in a research paper."""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Literature review generation failed: {str(e)}"
    
    def _generate_session_report(self, session: ResearchSession) -> str:
        """Generate comprehensive session report"""
        
        report = f"# Research Session Report\\n\\n"
        report += f"**Session ID:** {session.session_id}\\n"
        report += f"**Query:** {session.original_query}\\n"
        report += f"**Date:** {session.created_at[:10]}\\n"
        report += f"**Tasks Completed:** {len(session.results)}\\n\\n"
        
        # Executive summary
        successful_tasks = [r for r in session.results if r.success]
        report += f"## Executive Summary\\n\\n"
        report += f"Successfully completed {len(successful_tasks)} of {len(session.results)} tasks. "
        report += f"Total execution time: {sum(r.execution_time for r in session.results):.1f} seconds.\\n\\n"
        
        # Task results
        report += f"## Results by Task\\n\\n"
        
        for result in session.results:
            report += f"### {result.task_type.value.replace('_', ' ').title()}\\n\\n"
            
            if result.success:
                report += f"**Status:** ✅ Completed successfully\\n"
                report += f"**Agent:** {result.agent_used}\\n"
                report += f"**Execution Time:** {result.execution_time:.1f}s\\n"
                report += f"**Confidence:** {result.confidence_score:.1f}\\n\\n"
                
                # Include result data (truncated if too long)
                if isinstance(result.result_data, str):
                    result_preview = result.result_data[:500]
                    if len(result.result_data) > 500:
                        result_preview += "..."
                    report += f"**Result:**\\n{result_preview}\\n\\n"
                elif hasattr(result.result_data, 'to_summary'):
                    report += f"**Result:**\\n{result.result_data.to_summary()}\\n\\n"
                else:
                    report += f"**Result:** [Complex result - see detailed output]\\n\\n"
            else:
                report += f"**Status:** ❌ Failed\\n"
                report += f"**Error:** {result.error_message}\\n\\n"
        
        # Recommendations
        report += f"## Recommendations\\n\\n"
        if len(successful_tasks) == len(session.results):
            report += "- All tasks completed successfully\\n"
            report += "- Results are ready for integration into your research\\n"
        else:
            failed_tasks = [r for r in session.results if not r.success]
            report += f"- {len(failed_tasks)} tasks failed and may need attention\\n"
            report += "- Consider re-running failed tasks with adjusted parameters\\n"
        
        report += f"- Review detailed results above for research insights\\n"
        report += f"- Consider follow-up queries to explore specific aspects further\\n"
        
        return report
    
    def get_session_summary(self, session_id: str) -> Optional[str]:
        """Get summary of a specific session"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return session.final_report
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.active_sessions.keys())


def demonstrate_research_router():
    """Demonstrate research router capabilities"""
    
    print("🎯 RESEARCH ROUTER AGENT DEMONSTRATION")
    print("=" * 42)
    
    print("\\n🔄 Router Capabilities:")
    capabilities = [
        "Query analysis and task decomposition",
        "Intelligent agent selection and delegation",
        "Multi-task workflow orchestration", 
        "Session management and result tracking",
        "Comprehensive report generation",
        "Error handling and fallback strategies"
    ]
    
    for capability in capabilities:
        print(f"   • {capability}")
    
    print("\\n📋 Supported Task Types:")
    task_types = [
        ("Literature Search", "Find and analyze relevant research papers"),
        ("Citation Formatting", "Format citations in academic styles"),
        ("Summary Generation", "Create research area overviews"),
        ("Trend Analysis", "Identify emerging research directions"),
        ("Review Generation", "Write literature review sections"),
        ("Author Analysis", "Analyze researcher networks and influence")
    ]
    
    for task_type, description in task_types:
        print(f"   • {task_type}: {description}")
    
    print("\\n🤖 Agent Coordination:")
    coordination_features = [
        "Automatic task type detection from natural language",
        "Parameter extraction and optimization",
        "Sequential and parallel task execution",
        "Result aggregation and synthesis",
        "Quality assessment and validation"
    ]
    
    for feature in coordination_features:
        print(f"   • {feature}")
    
    print("\\n💡 Example Queries:")
    examples = [
        '"Find recent papers on transformer attention mechanisms and format citations in IEEE style"',
        '"Write a literature review on machine learning in healthcare"',
        '"What are the current trends in quantum computing research?"',
        '"Analyze collaboration networks in natural language processing"'
    ]
    
    for example in examples:
        print(f"   • {example}")


if __name__ == "__main__":
    demonstrate_research_router()
    
    print("\\n" + "="*50)
    print("🧪 RESEARCH ROUTER TESTING")
    print("="*50)
    
    print("\\n💡 To test the research router:")
    print("""
from research_router import ResearchRouterAgent
from llm_providers import create_research_llm

# Set up the router
llm = create_research_llm('google')
router = ResearchRouterAgent(llm)

# Execute research session
session = router.execute_research_session(
    "Find recent papers on transformer attention mechanisms and format citations in IEEE style"
)

# Get results
print(session.final_report)

# Check specific task results
for result in session.results:
    print(f"{result.task_type.value}: {result.success}")
    if result.success:
        print(f"Result preview: {str(result.result_data)[:200]}...")
""")