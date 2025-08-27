"""
Query Generator for Research Workflows

Intelligent query generation that transforms research questions into
optimal search strategies using multi-provider LLM support.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from state_schemas import SearchQueryList, SearchQuery, ResearchConfiguration


class ResearchQueryGenerator:
    """
    Intelligent query generator for research workflows.
    
    Transforms research questions into optimal search strategies
    using configurable LLM providers from Part 1.
    """
    
    def __init__(self, llm, config: Optional[ResearchConfiguration] = None):
        self.llm = llm
        self.config = config or ResearchConfiguration()
        
        # Query generation templates
        self.query_templates = {
            "academic": [
                "{topic} recent research {year}",
                "{topic} systematic review",
                "{topic} methodology advances",
                "{topic} applications and limitations"
            ],
            "technical": [
                "{topic} implementation approaches",
                "{topic} performance evaluation",
                "{topic} comparison study",
                "{topic} best practices"
            ],
            "trends": [
                "{topic} trends {year}",
                "{topic} emerging directions",
                "{topic} future research",
                "{topic} state of the art"
            ],
            "comprehensive": [
                "{topic} overview",
                "{topic} recent advances",
                "{topic} challenges and solutions",
                "{topic} practical applications"
            ]
        }
        
        # Domain-specific query patterns
        self.domain_patterns = {
            "machine_learning": [
                "deep learning", "neural networks", "training methods", "architectures",
                "optimization", "regularization", "performance metrics"
            ],
            "quantum_computing": [
                "quantum algorithms", "quantum hardware", "error correction", "NISQ devices",
                "quantum supremacy", "quantum software", "applications"
            ],
            "biotechnology": [
                "genomics", "proteomics", "drug discovery", "biomarkers",
                "clinical trials", "regulatory approval", "therapeutic targets"
            ],
            "climate_science": [
                "climate modeling", "greenhouse gases", "carbon capture", "renewable energy",
                "climate adaptation", "extreme weather", "environmental policy"
            ]
        }
        
        # Time-based modifiers
        self.temporal_modifiers = {
            "recent": ["2024", "2023", "recent", "latest", "new"],
            "emerging": ["emerging", "novel", "innovative", "cutting-edge"],
            "established": ["established", "traditional", "proven", "standard"]
        }
    
    def generate_initial_queries(self, research_question: str) -> SearchQueryList:
        """
        Generate initial search queries from a research question.
        
        Args:
            research_question: The research question to analyze
            
        Returns:
            SearchQueryList with generated queries and strategy
        """
        
        # Analyze the research question
        question_analysis = self._analyze_research_question(research_question)
        
        # Generate queries using LLM
        llm_queries = self._generate_llm_queries(research_question, question_analysis)
        
        # Enhance with template-based queries
        template_queries = self._generate_template_queries(research_question, question_analysis)
        
        # Combine and optimize
        all_queries = llm_queries + template_queries
        optimized_queries = self._optimize_query_list(all_queries)
        
        # Create rationale and strategy
        rationale = self._create_query_rationale(research_question, optimized_queries)
        strategy = self._determine_research_strategy(question_analysis)
        
        return SearchQueryList(
            queries=optimized_queries,
            rationale=rationale,
            research_strategy=strategy
        )
    
    def generate_follow_up_queries(self,
                                 original_question: str,
                                 current_findings: List[str],
                                 knowledge_gaps: List[str]) -> List[str]:
        """
        Generate follow-up queries to address identified knowledge gaps.
        
        Args:
            original_question: Original research question
            current_findings: Current research findings
            knowledge_gaps: Identified gaps in knowledge
            
        Returns:
            List of follow-up search queries
        """
        
        # Use LLM to generate targeted follow-up queries
        follow_up_prompt = f"""Based on this research question: "{original_question}"

Current findings include:
{self._format_findings(current_findings[:5])}

The following knowledge gaps have been identified:
{self._format_gaps(knowledge_gaps)}

Generate 2-4 specific search queries that would help fill these knowledge gaps. 
Focus on the missing aspects and provide queries that would find complementary information.

Format your response as a simple list of queries, one per line."""

        try:
            response = self.llm.invoke(follow_up_prompt)
            follow_up_queries = self._parse_llm_queries(response.content)
            
            # Add template-based follow-ups for gaps
            template_follow_ups = self._generate_gap_specific_queries(knowledge_gaps)
            
            # Combine and optimize
            all_follow_ups = follow_up_queries + template_follow_ups
            optimized_follow_ups = self._optimize_query_list(all_follow_ups)
            
            return optimized_follow_ups[:4]  # Limit to 4 follow-ups
            
        except Exception as e:
            print(f"⚠️ LLM follow-up generation failed: {e}")
            return self._generate_gap_specific_queries(knowledge_gaps)
    
    def _analyze_research_question(self, question: str) -> Dict[str, Any]:
        """Analyze research question to determine optimal query strategy"""
        
        question_lower = question.lower()
        
        analysis = {
            "type": "general",
            "domain": "general",
            "temporal_focus": "recent",
            "complexity": "moderate",
            "scope": "comprehensive"
        }
        
        # Determine question type
        if any(word in question_lower for word in ["recent", "latest", "new", "current"]):
            analysis["type"] = "trends"
            analysis["temporal_focus"] = "recent"
        elif any(word in question_lower for word in ["compare", "difference", "versus", "vs"]):
            analysis["type"] = "comparison"
        elif any(word in question_lower for word in ["how", "method", "approach", "technique"]):
            analysis["type"] = "technical"
        elif any(word in question_lower for word in ["review", "overview", "survey", "state"]):
            analysis["type"] = "academic"
            analysis["scope"] = "broad"
        
        # Identify domain
        for domain, keywords in self.domain_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                analysis["domain"] = domain
                break
        
        # Assess complexity
        if len(question.split()) > 15 or "and" in question_lower or "," in question:
            analysis["complexity"] = "complex"
        elif len(question.split()) < 8:
            analysis["complexity"] = "simple"
        
        return analysis
    
    def _generate_llm_queries(self, question: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate queries using LLM reasoning"""
        
        # Customize prompt based on analysis
        domain_context = ""
        if analysis["domain"] != "general":
            domain_context = f" in the {analysis['domain'].replace('_', ' ')} domain"
        
        temporal_context = ""
        if analysis["temporal_focus"] == "recent":
            temporal_context = f" Focus on recent work from {datetime.now().year-1}-{datetime.now().year}."
        
        llm_prompt = f"""You are a research assistant helping to create search queries for comprehensive research{domain_context}.

Research Question: "{question}"

Generate {self.config.initial_query_count} diverse search queries that would help answer this research question comprehensively. Each query should:
1. Target different aspects of the topic
2. Use varied terminology and synonyms
3. Be specific enough to find relevant results
4. Cover both theoretical and practical perspectives{temporal_context}

Provide only the search queries, one per line, without numbers or bullets."""

        try:
            response = self.llm.invoke(llm_prompt)
            return self._parse_llm_queries(response.content)
        except Exception as e:
            print(f"⚠️ LLM query generation failed: {e}")
            return []
    
    def _generate_template_queries(self, question: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate queries using template-based approach"""
        
        # Extract main topic from question
        topic = self._extract_topic(question)
        current_year = str(datetime.now().year)
        
        # Select appropriate template set
        template_set = self.query_templates.get(analysis["type"], self.query_templates["comprehensive"])
        
        # Generate queries from templates
        template_queries = []
        for template in template_set:
            query = template.format(topic=topic, year=current_year)
            template_queries.append(query)
        
        # Add domain-specific queries if applicable
        if analysis["domain"] in self.domain_patterns:
            domain_keywords = self.domain_patterns[analysis["domain"]][:2]  # Use top 2
            for keyword in domain_keywords:
                template_queries.append(f"{topic} {keyword}")
        
        return template_queries[:3]  # Limit template queries
    
    def _extract_topic(self, question: str) -> str:
        """Extract main topic from research question"""
        
        # Remove question words
        question_words = ["what", "how", "why", "when", "where", "which", "who", 
                         "are", "is", "can", "does", "do", "will", "would", "should"]
        
        words = question.lower().split()
        topic_words = [w for w in words if w not in question_words and len(w) > 2]
        
        # Take key terms (remove common words)
        common_words = ["the", "and", "or", "but", "for", "with", "from", "about", "into", "through"]
        key_words = [w for w in topic_words if w not in common_words]
        
        # Return first few key words as topic
        return " ".join(key_words[:4])
    
    def _optimize_query_list(self, queries: List[str]) -> List[str]:
        """Optimize query list by removing duplicates and poor queries"""
        
        if not queries:
            return []
        
        # Remove duplicates and very similar queries
        unique_queries = []
        for query in queries:
            query_clean = query.strip().lower()
            if query_clean and len(query_clean) > 5:  # Minimum length check
                # Check for similarity with existing queries
                is_duplicate = False
                for existing in unique_queries:
                    if self._queries_too_similar(query_clean, existing.lower()):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_queries.append(query.strip())
        
        # Limit to configured number
        max_queries = min(self.config.initial_query_count, 6)  # Max 6 queries
        return unique_queries[:max_queries]
    
    def _queries_too_similar(self, query1: str, query2: str, threshold: float = 0.7) -> bool:
        """Check if two queries are too similar"""
        
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold
    
    def _parse_llm_queries(self, llm_response: str) -> List[str]:
        """Parse LLM response into query list"""
        
        queries = []
        lines = llm_response.strip().split('\\n')
        
        for line in lines:
            # Remove common prefixes and clean up
            clean_line = re.sub(r'^[\\d\\.\\-\\*\\+]\\s*', '', line.strip())
            clean_line = re.sub(r'^["""]', '', clean_line)
            clean_line = re.sub(r'["""]$', '', clean_line)
            
            if clean_line and len(clean_line) > 5:
                queries.append(clean_line)
        
        return queries
    
    def _create_query_rationale(self, question: str, queries: List[str]) -> str:
        """Create rationale for the generated queries"""
        
        return f"""These {len(queries)} queries are designed to comprehensively research "{question[:50]}..." by:
1. Covering different aspects and perspectives of the topic
2. Using varied terminology to capture diverse sources
3. Balancing broad overview with specific technical details
4. Including recent developments and established knowledge
5. Targeting both academic and practical applications"""
    
    def _determine_research_strategy(self, analysis: Dict[str, Any]) -> str:
        """Determine research strategy based on question analysis"""
        
        strategies = {
            "academic": "Systematic academic literature review approach",
            "technical": "Technical implementation and methodology focus",
            "trends": "Current trends and emerging developments analysis", 
            "comparison": "Comparative analysis of different approaches",
            "general": "Comprehensive multi-perspective research strategy"
        }
        
        base_strategy = strategies.get(analysis["type"], strategies["general"])
        
        if analysis["complexity"] == "complex":
            base_strategy += " with iterative refinement for complex topic coverage"
        elif analysis["scope"] == "broad":
            base_strategy += " emphasizing breadth of coverage across subdomains"
        
        return base_strategy
    
    def _generate_gap_specific_queries(self, knowledge_gaps: List[str]) -> List[str]:
        """Generate queries specifically targeting knowledge gaps"""
        
        gap_queries = []
        for gap in knowledge_gaps[:3]:  # Limit to 3 gaps
            # Extract key terms from gap description
            gap_terms = self._extract_key_terms(gap)
            if gap_terms:
                gap_queries.append(" ".join(gap_terms))
        
        return gap_queries
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from gap description"""
        
        # Simple keyword extraction
        words = text.lower().split()
        key_terms = [w for w in words if len(w) > 3 and w.isalpha()]
        return key_terms[:3]  # Top 3 terms
    
    def _format_findings(self, findings: List[str]) -> str:
        """Format findings for LLM prompts"""
        if not findings:
            return "No specific findings available yet."
        
        formatted = []
        for i, finding in enumerate(findings[:5], 1):
            formatted.append(f"{i}. {finding[:100]}...")
        return "\\n".join(formatted)
    
    def _format_gaps(self, gaps: List[str]) -> str:
        """Format knowledge gaps for LLM prompts"""
        if not gaps:
            return "No specific gaps identified."
        
        formatted = []
        for i, gap in enumerate(gaps[:3], 1):
            formatted.append(f"{i}. {gap}")
        return "\\n".join(formatted)


def demonstrate_query_generator():
    """Demonstrate query generation capabilities"""
    
    print("🔍 QUERY GENERATOR DEMONSTRATION")
    print("=" * 32)
    
    print("\\n🎯 Query Generation Capabilities:")
    capabilities = [
        "Multi-perspective query generation from research questions",
        "Domain-specific query optimization",
        "Temporal focus adaptation (recent vs established work)",
        "Follow-up query generation for knowledge gaps",
        "Template-based and LLM-based query synthesis",
        "Query deduplication and optimization"
    ]
    
    for capability in capabilities:
        print(f"   • {capability}")
    
    print("\\n📋 Supported Question Types:")
    question_types = [
        ("Trends", "What are recent advances in quantum computing?"),
        ("Technical", "How do transformer attention mechanisms work?"),
        ("Academic", "Review of deep learning in medical imaging"),
        ("Comparison", "Compare CNN vs Transformer architectures"),
        ("Comprehensive", "Overview of climate change modeling approaches")
    ]
    
    for q_type, example in question_types:
        print(f"   • {q_type}: '{example}'")
    
    print("\\n🔄 Query Generation Process:")
    process_steps = [
        "1. Analyze research question (type, domain, complexity)",
        "2. Generate LLM-based queries using reasoning",
        "3. Create template-based queries for coverage",
        "4. Optimize and deduplicate query list",
        "5. Create rationale and research strategy"
    ]
    
    for step in process_steps:
        print(f"   {step}")


if __name__ == "__main__":
    demonstrate_query_generator()
    
    print("\\n" + "="*50)
    print("🧪 QUERY GENERATOR TESTING")
    print("="*50)
    
    print("\\n💡 To test with LLM:")
    print("""
# Import from Part 1
sys.path.append('../Part1_Foundations/modules')
from llm_providers import create_research_llm

# Create LLM and query generator
llm = create_research_llm()
config = ResearchConfiguration.for_academic_research()
generator = ResearchQueryGenerator(llm, config)

# Generate queries
research_question = "What are recent advances in quantum error correction?"
queries = generator.generate_initial_queries(research_question)

print(f"Generated {len(queries.queries)} queries:")
for i, query in enumerate(queries.queries, 1):
    print(f"{i}. {query}")

print(f"Strategy: {queries.research_strategy}")
""")