"""
Reflection Agent for Research Workflows

Intelligent reflection agent that identifies knowledge gaps, assesses research
completeness, and generates follow-up strategies for comprehensive coverage.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from state_schemas import ReflectionResult, ResearchSource, ResearchConfiguration


class ResearchReflectionAgent:
    """
    Reflection agent for identifying knowledge gaps in research.
    
    Uses LLM reasoning to analyze research findings, identify missing
    information, and suggest follow-up research strategies.
    """
    
    def __init__(self, llm, config: Optional[ResearchConfiguration] = None):
        self.llm = llm
        self.config = config or ResearchConfiguration()
        
        # Knowledge gap categories
        self.gap_categories = {
            "methodological": [
                "implementation details", "algorithm specifics", "technical approaches",
                "experimental methodology", "validation techniques"
            ],
            "empirical": [
                "performance metrics", "benchmark results", "comparative studies",
                "experimental evidence", "case studies"
            ],
            "contextual": [
                "applications", "use cases", "real-world deployment",
                "industry adoption", "practical considerations"
            ],
            "temporal": [
                "recent developments", "current trends", "future directions",
                "historical context", "evolution over time"
            ],
            "scope": [
                "related fields", "interdisciplinary connections", "broader implications",
                "alternative approaches", "competing paradigms"
            ]
        }
        
        # Completeness assessment criteria
        self.completeness_criteria = {
            "coverage_breadth": 0.25,    # How many aspects are covered
            "depth_adequacy": 0.25,      # How thoroughly each aspect is covered
            "source_diversity": 0.20,    # Variety of source types and perspectives
            "recency_relevance": 0.15,   # How current the information is
            "authority_credibility": 0.15 # Credibility of sources
        }
        
        # Research quality indicators
        self.quality_indicators = [
            "multiple perspectives presented",
            "recent and relevant sources",
            "authoritative and credible sources",
            "comprehensive topic coverage",
            "clear methodology discussion",
            "practical applications included",
            "limitations and challenges addressed"
        ]
    
    def analyze_research_completeness(self,
                                   original_question: str,
                                   research_findings: List[str],
                                   sources: List[ResearchSource],
                                   current_iteration: int) -> ReflectionResult:
        """
        Analyze research completeness and identify knowledge gaps.
        
        Args:
            original_question: Original research question
            research_findings: Current research findings
            sources: List of research sources
            current_iteration: Current research iteration number
            
        Returns:
            ReflectionResult with gap analysis and follow-up suggestions
        """
        
        print(f"🤔 Reflecting on research completeness (iteration {current_iteration})...")
        
        # Analyze current coverage
        coverage_analysis = self._analyze_coverage(original_question, research_findings, sources)
        
        # Generate reflection using LLM
        reflection_result = self._generate_llm_reflection(
            original_question, 
            research_findings, 
            coverage_analysis
        )
        
        # Enhance with systematic gap analysis
        systematic_gaps = self._identify_systematic_gaps(
            original_question, 
            research_findings, 
            sources
        )
        
        # Combine and finalize reflection
        final_reflection = self._combine_reflections(reflection_result, systematic_gaps)
        
        # Add quality scores
        final_reflection.confidence_score = self._calculate_confidence_score(
            research_findings, sources, coverage_analysis
        )
        
        print(f"   📊 Coverage assessment: {coverage_analysis['overall_score']:.1%}")
        print(f"   🎯 Confidence: {final_reflection.confidence_score:.2f}")
        print(f"   🔍 Gaps identified: {len(final_reflection.missing_aspects)}")
        
        return final_reflection
    
    def _analyze_coverage(self, 
                         question: str, 
                         findings: List[str], 
                         sources: List[ResearchSource]) -> Dict[str, Any]:
        """Analyze coverage quality across different dimensions"""
        
        coverage_analysis = {
            "total_findings": len(findings),
            "total_sources": len(sources),
            "source_diversity": self._calculate_source_diversity(sources),
            "temporal_coverage": self._assess_temporal_coverage(sources),
            "topic_breadth": self._assess_topic_breadth(question, findings),
            "overall_score": 0.0
        }
        
        # Calculate overall coverage score
        scores = []
        
        # Source quantity score
        source_score = min(len(sources) / self.config.minimum_sources, 1.0)
        scores.append(source_score)
        
        # Source diversity score  
        scores.append(coverage_analysis["source_diversity"])
        
        # Temporal coverage score
        scores.append(coverage_analysis["temporal_coverage"])
        
        # Topic breadth score
        scores.append(coverage_analysis["topic_breadth"])
        
        coverage_analysis["overall_score"] = sum(scores) / len(scores)
        
        return coverage_analysis
    
    def _generate_llm_reflection(self,
                               question: str,
                               findings: List[str],
                               coverage_analysis: Dict[str, Any]) -> ReflectionResult:
        """Generate reflection using LLM analysis"""
        
        # Prepare research summary for LLM
        findings_summary = self._prepare_findings_summary(findings)
        coverage_summary = self._prepare_coverage_summary(coverage_analysis)
        
        reflection_prompt = f"""You are a research analyst evaluating the completeness of research on a specific topic.

ORIGINAL RESEARCH QUESTION: "{question}"

CURRENT RESEARCH FINDINGS:
{findings_summary}

COVERAGE ANALYSIS:
{coverage_summary}

Please analyze whether this research is sufficient to comprehensively answer the original question. Consider:

1. COMPLETENESS: Are all major aspects of the topic covered?
2. DEPTH: Is there sufficient detail and evidence for each aspect?
3. RECENCY: Are recent developments and current state included?
4. PERSPECTIVES: Are different viewpoints and approaches represented?
5. APPLICATIONS: Are practical applications and real-world examples included?

Respond in the following format:

SUFFICIENT: [YES/NO]

KNOWLEDGE_GAPS: [If not sufficient, describe what specific information is missing or needs clarification. Be specific about aspects that need more coverage.]

MISSING_ASPECTS: [List 2-4 specific aspects that need more research, one per line]

FOLLOW_UP_QUERIES: [Suggest 2-3 specific search queries that would help fill the identified gaps, one per line]

Be concise but specific in your analysis."""

        try:
            response = self.llm.invoke(reflection_prompt)
            return self._parse_llm_reflection(response.content)
            
        except Exception as e:
            print(f"   ⚠️ LLM reflection failed: {e}")
            return self._create_fallback_reflection(question, findings)
    
    def _identify_systematic_gaps(self,
                                question: str,
                                findings: List[str],
                                sources: List[ResearchSource]) -> Dict[str, List[str]]:
        """Identify systematic gaps using predefined categories"""
        
        systematic_gaps = {category: [] for category in self.gap_categories.keys()}
        
        # Analyze question to determine expected gap categories
        question_lower = question.lower()
        
        # Check for methodological gaps
        if not self._has_methodological_content(findings):
            if any(word in question_lower for word in ["how", "method", "approach", "technique"]):
                systematic_gaps["methodological"].append("implementation methodology")
        
        # Check for empirical gaps
        if not self._has_empirical_content(findings):
            if any(word in question_lower for word in ["performance", "results", "effectiveness"]):
                systematic_gaps["empirical"].append("performance evaluation")
        
        # Check for contextual gaps
        if not self._has_contextual_content(findings):
            if any(word in question_lower for word in ["applications", "use", "real-world"]):
                systematic_gaps["contextual"].append("practical applications")
        
        # Check for temporal gaps
        if not self._has_recent_content(sources):
            if any(word in question_lower for word in ["recent", "current", "latest", "new"]):
                systematic_gaps["temporal"].append("recent developments")
        
        return systematic_gaps
    
    def _has_methodological_content(self, findings: List[str]) -> bool:
        """Check if findings contain methodological information"""
        methodological_terms = ["method", "approach", "algorithm", "technique", "procedure", "process"]
        findings_text = " ".join(findings).lower()
        return any(term in findings_text for term in methodological_terms)
    
    def _has_empirical_content(self, findings: List[str]) -> bool:
        """Check if findings contain empirical information"""
        empirical_terms = ["result", "performance", "accuracy", "evaluation", "experiment", "study", "analysis"]
        findings_text = " ".join(findings).lower()
        return any(term in findings_text for term in empirical_terms)
    
    def _has_contextual_content(self, findings: List[str]) -> bool:
        """Check if findings contain contextual information"""
        contextual_terms = ["application", "use case", "deployment", "industry", "practical", "real-world"]
        findings_text = " ".join(findings).lower()
        return any(term in findings_text for term in contextual_terms)
    
    def _has_recent_content(self, sources: List[ResearchSource]) -> bool:
        """Check if sources contain recent information"""
        current_year = datetime.now().year
        recent_count = 0
        
        for source in sources:
            if source.publication_date:
                try:
                    pub_year = int(source.publication_date[:4])
                    if pub_year >= current_year - 2:
                        recent_count += 1
                except:
                    pass
        
        return recent_count >= len(sources) * 0.3  # At least 30% recent
    
    def _calculate_source_diversity(self, sources: List[ResearchSource]) -> float:
        """Calculate diversity of source types"""
        if not sources:
            return 0.0
        
        source_types = set(source.source_type for source in sources)
        max_diversity = len(["academic", "institutional", "technical", "web"])
        
        return len(source_types) / max_diversity
    
    def _assess_temporal_coverage(self, sources: List[ResearchSource]) -> float:
        """Assess temporal coverage of sources"""
        if not sources:
            return 0.0
        
        current_year = datetime.now().year
        dated_sources = [s for s in sources if s.publication_date]
        
        if not dated_sources:
            return 0.5  # Neutral score for undated sources
        
        # Check for recent sources (last 3 years)
        recent_count = 0
        for source in dated_sources:
            try:
                pub_year = int(source.publication_date[:4])
                if pub_year >= current_year - 3:
                    recent_count += 1
            except:
                pass
        
        return recent_count / len(dated_sources)
    
    def _assess_topic_breadth(self, question: str, findings: List[str]) -> float:
        """Assess breadth of topic coverage"""
        if not findings:
            return 0.0
        
        # Extract key terms from question
        question_terms = set(re.findall(r'\\b\\w{4,}\\b', question.lower()))
        
        # Count unique concepts in findings
        findings_text = " ".join(findings).lower()
        findings_terms = set(re.findall(r'\\b\\w{4,}\\b', findings_text))
        
        # Calculate coverage
        if not question_terms:
            return 0.5
        
        overlap = len(question_terms.intersection(findings_terms))
        coverage = overlap / len(question_terms)
        
        # Bonus for additional relevant terms
        additional_terms = len(findings_terms - question_terms)
        breadth_bonus = min(additional_terms / 20, 0.3)  # Up to 30% bonus
        
        return min(1.0, coverage + breadth_bonus)
    
    def _parse_llm_reflection(self, llm_response: str) -> ReflectionResult:
        """Parse LLM reflection response into structured format"""
        
        try:
            # Extract sections
            is_sufficient = self._extract_sufficiency(llm_response)
            knowledge_gap = self._extract_knowledge_gap(llm_response)
            missing_aspects = self._extract_missing_aspects(llm_response)
            follow_up_queries = self._extract_follow_up_queries(llm_response)
            
            return ReflectionResult(
                is_sufficient=is_sufficient,
                knowledge_gap=knowledge_gap,
                follow_up_queries=follow_up_queries,
                missing_aspects=missing_aspects
            )
            
        except Exception as e:
            print(f"   ⚠️ Error parsing LLM reflection: {e}")
            return ReflectionResult(
                is_sufficient=False,
                knowledge_gap="Unable to assess completeness due to parsing error",
                follow_up_queries=["additional research needed"],
                missing_aspects=["comprehensive analysis required"]
            )
    
    def _extract_sufficiency(self, response: str) -> bool:
        """Extract sufficiency assessment from LLM response"""
        sufficient_match = re.search(r'SUFFICIENT:\\s*(YES|NO)', response, re.IGNORECASE)
        if sufficient_match:
            return sufficient_match.group(1).upper() == 'YES'
        
        # Fallback: look for keywords
        response_lower = response.lower()
        if 'sufficient' in response_lower and 'yes' in response_lower:
            return True
        elif 'not sufficient' in response_lower or ('no' in response_lower and 'sufficient' in response_lower):
            return False
        
        return False  # Default to not sufficient
    
    def _extract_knowledge_gap(self, response: str) -> str:
        """Extract knowledge gap description from LLM response"""
        gap_match = re.search(r'KNOWLEDGE_GAPS?:\\s*\\[([^\\]]+)\\]', response, re.IGNORECASE | re.DOTALL)
        if gap_match:
            return gap_match.group(1).strip()
        
        # Fallback: look for gap-related content
        lines = response.split('\\n')
        for i, line in enumerate(lines):
            if 'gap' in line.lower() or 'missing' in line.lower():
                # Return this line and possibly the next few lines
                gap_lines = lines[i:i+3]
                return " ".join(gap_lines).strip()
        
        return "Additional research needed to ensure comprehensive coverage"
    
    def _extract_missing_aspects(self, response: str) -> List[str]:
        """Extract missing aspects list from LLM response"""
        aspects_match = re.search(r'MISSING_ASPECTS?:\\s*\\[([^\\]]+)\\]', response, re.IGNORECASE | re.DOTALL)
        if aspects_match:
            aspects_text = aspects_match.group(1).strip()
            aspects = [aspect.strip() for aspect in aspects_text.split('\\n') if aspect.strip()]
            return aspects[:4]  # Limit to 4 aspects
        
        # Fallback: extract bullet points or numbered items
        aspects = []
        lines = response.split('\\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^[\\d\\-\\*\\+•]', line):
                clean_aspect = re.sub(r'^[\\d\\.\\-\\*\\+•\\s]+', '', line).strip()
                if clean_aspect and len(clean_aspect) > 5:
                    aspects.append(clean_aspect)
        
        return aspects[:4] if aspects else ["comprehensive analysis", "detailed methodology"]
    
    def _extract_follow_up_queries(self, response: str) -> List[str]:
        """Extract follow-up queries from LLM response"""
        queries_match = re.search(r'FOLLOW[_\\s]?UP[_\\s]?QUERIES?:\\s*\\[([^\\]]+)\\]', response, re.IGNORECASE | re.DOTALL)
        if queries_match:
            queries_text = queries_match.group(1).strip()
            queries = [query.strip() for query in queries_text.split('\\n') if query.strip()]
            return queries[:3]  # Limit to 3 queries
        
        # Fallback: extract quoted strings or bullet points
        queries = []
        lines = response.split('\\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^[\\d\\-\\*\\+•]', line):
                clean_query = re.sub(r'^[\\d\\.\\-\\*\\+•\\s]+', '', line).strip()
                if clean_query and len(clean_query) > 10:
                    queries.append(clean_query)
        
        return queries[:3] if queries else ["additional specific research needed"]
    
    def _create_fallback_reflection(self, question: str, findings: List[str]) -> ReflectionResult:
        """Create fallback reflection when LLM analysis fails"""
        
        # Simple heuristic-based assessment
        is_sufficient = len(findings) >= self.config.minimum_sources
        
        knowledge_gap = "Unable to perform detailed analysis. Consider gathering more diverse sources."
        
        missing_aspects = [
            "comprehensive coverage verification",
            "source diversity assessment", 
            "recent developments analysis"
        ]
        
        follow_up_queries = [
            f"{question} recent research",
            f"{question} comprehensive review"
        ]
        
        return ReflectionResult(
            is_sufficient=is_sufficient,
            knowledge_gap=knowledge_gap,
            follow_up_queries=follow_up_queries,
            missing_aspects=missing_aspects
        )
    
    def _combine_reflections(self, 
                           llm_reflection: ReflectionResult, 
                           systematic_gaps: Dict[str, List[str]]) -> ReflectionResult:
        """Combine LLM reflection with systematic gap analysis"""
        
        # Merge missing aspects
        all_missing = list(llm_reflection.missing_aspects)
        for category, gaps in systematic_gaps.items():
            all_missing.extend(gaps)
        
        # Remove duplicates and limit
        unique_missing = list(dict.fromkeys(all_missing))[:6]
        
        # Enhance follow-up queries with systematic suggestions
        all_queries = list(llm_reflection.follow_up_queries)
        for category, gaps in systematic_gaps.items():
            for gap in gaps[:1]:  # One query per category
                all_queries.append(f"{gap} research methods")
        
        unique_queries = list(dict.fromkeys(all_queries))[:4]
        
        return ReflectionResult(
            is_sufficient=llm_reflection.is_sufficient,
            knowledge_gap=llm_reflection.knowledge_gap,
            follow_up_queries=unique_queries,
            missing_aspects=unique_missing
        )
    
    def _calculate_confidence_score(self,
                                  findings: List[str],
                                  sources: List[ResearchSource],
                                  coverage_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score in the reflection assessment"""
        
        confidence_factors = []
        
        # Source quantity factor
        source_factor = min(len(sources) / 10, 1.0)  # Normalize to 10 sources
        confidence_factors.append(source_factor)
        
        # Source diversity factor
        confidence_factors.append(coverage_analysis["source_diversity"])
        
        # Coverage score factor
        confidence_factors.append(coverage_analysis["overall_score"])
        
        # Quality indicator factor
        quality_score = self._assess_quality_indicators(findings, sources)
        confidence_factors.append(quality_score)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _assess_quality_indicators(self, findings: List[str], sources: List[ResearchSource]) -> float:
        """Assess research quality indicators"""
        
        if not findings:
            return 0.0
        
        quality_score = 0.0
        findings_text = " ".join(findings).lower()
        
        # Check for multiple perspectives
        if len(set(s.source_type for s in sources)) > 2:
            quality_score += 0.2
        
        # Check for authoritative sources
        academic_sources = [s for s in sources if s.source_type == 'academic']
        if len(academic_sources) >= len(sources) * 0.3:
            quality_score += 0.3
        
        # Check for methodological content
        method_terms = ["method", "approach", "technique", "procedure"]
        if any(term in findings_text for term in method_terms):
            quality_score += 0.2
        
        # Check for practical content
        practical_terms = ["application", "use case", "implementation", "deployment"]
        if any(term in findings_text for term in practical_terms):
            quality_score += 0.2
        
        # Check for limitations discussion
        limitation_terms = ["limitation", "challenge", "drawback", "problem"]
        if any(term in findings_text for term in limitation_terms):
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _prepare_findings_summary(self, findings: List[str]) -> str:
        """Prepare findings summary for LLM analysis"""
        if not findings:
            return "No findings available yet."
        
        # Limit and format findings
        summary_findings = findings[:8]  # Limit for LLM context
        formatted = []
        
        for i, finding in enumerate(summary_findings, 1):
            # Truncate long findings
            truncated = finding[:300] + "..." if len(finding) > 300 else finding
            formatted.append(f"{i}. {truncated}")
        
        return "\\n".join(formatted)
    
    def _prepare_coverage_summary(self, coverage_analysis: Dict[str, Any]) -> str:
        """Prepare coverage analysis summary for LLM"""
        return f"""Total Sources: {coverage_analysis['total_sources']}
Total Findings: {coverage_analysis['total_findings']}
Source Diversity: {coverage_analysis['source_diversity']:.1%}
Temporal Coverage: {coverage_analysis['temporal_coverage']:.1%}
Topic Breadth: {coverage_analysis['topic_breadth']:.1%}
Overall Coverage Score: {coverage_analysis['overall_score']:.1%}"""


def demonstrate_reflection_agent():
    """Demonstrate reflection agent capabilities"""
    
    print("🤔 REFLECTION AGENT DEMONSTRATION")
    print("=" * 32)
    
    print("\\n🎯 Reflection Capabilities:")
    capabilities = [
        "Research completeness assessment",
        "Knowledge gap identification",
        "Follow-up query generation", 
        "Multi-dimensional coverage analysis",
        "Systematic gap categorization",
        "Quality indicator evaluation"
    ]
    
    for capability in capabilities:
        print(f"   • {capability}")
    
    print("\\n📊 Gap Categories:")
    gap_categories = [
        ("Methodological", "Implementation details, algorithms, techniques"),
        ("Empirical", "Performance metrics, benchmarks, experimental evidence"),
        ("Contextual", "Applications, use cases, real-world deployment"),
        ("Temporal", "Recent developments, current trends, future directions"),
        ("Scope", "Related fields, alternative approaches, broader implications")
    ]
    
    for category, description in gap_categories:
        print(f"   • {category}: {description}")
    
    print("\\n🔍 Assessment Dimensions:")
    dimensions = [
        "Coverage breadth (how many aspects covered)",
        "Coverage depth (how thoroughly each aspect covered)",
        "Source diversity (variety of perspectives and types)",
        "Temporal relevance (recency and currentness)",
        "Authority credibility (trustworthiness of sources)"
    ]
    
    for dimension in dimensions:
        print(f"   • {dimension}")


if __name__ == "__main__":
    demonstrate_reflection_agent()
    
    print("\\n" + "="*50)
    print("🧪 REFLECTION AGENT TESTING")
    print("="*50)
    
    print("\\n💡 To test reflection agent:")
    print("""
# Import from Part 1
sys.path.append('../Part1_Foundations/modules')
from llm_providers import create_research_llm

from reflection_agent import ResearchReflectionAgent
from state_schemas import ResearchConfiguration, ResearchSource

# Create reflection agent
llm = create_research_llm()
config = ResearchConfiguration.for_academic_research()
reflection_agent = ResearchReflectionAgent(llm, config)

# Sample data
question = "What are recent advances in quantum error correction?"
findings = [
    "Surface codes are the leading approach for fault-tolerant quantum computing",
    "Recent work focuses on reducing overhead and improving error thresholds",
    "Color codes provide an alternative with different trade-offs"
]

sources = [
    ResearchSource(url="https://arxiv.org/abs/2023.12345", title="Recent Advances in Surface Codes", 
                  snippet="This paper reviews...", source_type="academic"),
    # Add more sources...
]

# Analyze completeness
reflection = reflection_agent.analyze_research_completeness(
    question, findings, sources, current_iteration=1
)

print(f"Sufficient: {reflection.is_sufficient}")
print(f"Knowledge Gap: {reflection.knowledge_gap}")
print(f"Follow-up Queries: {reflection.follow_up_queries}")
print(f"Missing Aspects: {reflection.missing_aspects}")
""")