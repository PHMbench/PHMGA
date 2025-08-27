"""
Research Pipeline DAG Implementation

Advanced DAG patterns for academic research workflows including
literature review, systematic analysis, and multi-stage research processes.
"""

import sys
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from our foundation
sys.path.append('../../Part1_Foundations/modules')
from dag_fundamentals import ResearchDAG, DAGNode, NodeType, ExecutionStatus


class ResearchPhase(Enum):
    """Phases of academic research process"""
    PLANNING = "planning"
    SEARCH = "search"
    SCREENING = "screening"
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REPORTING = "reporting"


class QualityGate(Enum):
    """Quality gates for research validation"""
    RELEVANCE_CHECK = "relevance_check"
    CREDIBILITY_ASSESSMENT = "credibility_assessment"
    COMPLETENESS_VALIDATION = "completeness_validation"
    BIAS_DETECTION = "bias_detection"


@dataclass
class ResearchCriteria:
    """Inclusion/exclusion criteria for systematic reviews"""
    inclusion_keywords: List[str]
    exclusion_keywords: List[str]
    min_publication_year: int = 2020
    required_study_types: List[str] = None
    minimum_citation_count: int = 0
    language_requirements: List[str] = None
    
    def __post_init__(self):
        if self.required_study_types is None:
            self.required_study_types = ["empirical", "systematic_review", "case_study"]
        if self.language_requirements is None:
            self.language_requirements = ["english"]


class LiteratureReviewDAG(ResearchDAG):
    """
    Specialized DAG for systematic literature review workflows.
    
    Implements the standard systematic review process:
    Planning → Search → Screening → Data Extraction → Analysis → Synthesis
    """
    
    def __init__(self, review_topic: str, criteria: ResearchCriteria):
        super().__init__(f"literature_review_{review_topic}", 
                        f"Systematic literature review on {review_topic}")
        self.review_topic = review_topic
        self.criteria = criteria
        self.search_databases = ["arxiv", "pubmed", "ieee", "acm"]
        self.parallel_enabled = True
        
        # Build the complete workflow
        self._build_review_pipeline()
    
    def _build_review_pipeline(self):
        """Construct the complete literature review DAG"""
        
        # Phase 1: Planning and Query Generation
        self._add_planning_phase()
        
        # Phase 2: Multi-database search (parallel)
        self._add_search_phase()
        
        # Phase 3: Title and abstract screening
        self._add_screening_phase()
        
        # Phase 4: Full-text review and data extraction
        self._add_extraction_phase()
        
        # Phase 5: Quality assessment and analysis
        self._add_analysis_phase()
        
        # Phase 6: Synthesis and reporting
        self._add_synthesis_phase()
    
    def _add_planning_phase(self):
        """Add research planning nodes"""
        
        def define_research_questions(inputs):
            """Define primary and secondary research questions"""
            return {
                "primary_question": f"What are the recent advances in {self.review_topic}?",
                "secondary_questions": [
                    f"What methodologies are used in {self.review_topic} research?",
                    f"What are the main challenges in {self.review_topic}?",
                    f"What future research directions are identified?"
                ],
                "search_strategy": "comprehensive"
            }
        
        def generate_search_queries(inputs):
            """Generate search queries for different databases"""
            base_terms = self.criteria.inclusion_keywords
            
            # Create Boolean search strategies
            queries = {
                "boolean_and": " AND ".join(base_terms),
                "boolean_or": " OR ".join(base_terms), 
                "phrase_search": f'"{" ".join(base_terms)}"',
                "wildcard_search": " AND ".join([f"{term}*" for term in base_terms])
            }
            
            return {
                "search_queries": queries,
                "database_specific_queries": {
                    db: self._adapt_query_for_database(queries["boolean_and"], db) 
                    for db in self.search_databases
                }
            }
        
        # Create planning nodes
        planning_node = DAGNode("planning", "Research Planning", NodeType.INPUT, define_research_questions)
        planning_node.research_phase = ResearchPhase.PLANNING.value
        
        query_gen_node = DAGNode("query_generation", "Search Query Generation", 
                                NodeType.PROCESSING, generate_search_queries)
        query_gen_node.research_phase = ResearchPhase.PLANNING.value
        
        self.add_node(planning_node)
        self.add_node(query_gen_node)
        self.add_edge("planning", "query_generation")
    
    def _add_search_phase(self):
        """Add parallel database search nodes"""
        
        search_nodes = []
        
        for database in self.search_databases:
            def create_search_operation(db_name):
                def search_database(inputs):
                    """Search specific database"""
                    queries = inputs.get("query_generation", {}).get("database_specific_queries", {})
                    query = queries.get(db_name, self.review_topic)
                    
                    # Simulate database search with realistic delays
                    time.sleep(0.3 + hash(db_name) % 3 * 0.1)  # Varied delays
                    
                    # Generate realistic result counts
                    base_results = 50 + hash(f"{db_name}{query}") % 200
                    
                    return {
                        "database": db_name,
                        "query_used": query,
                        "total_results": base_results,
                        "results": [
                            {
                                "id": f"{db_name}_{i}",
                                "title": f"Paper {i} from {db_name} on {self.review_topic}",
                                "authors": [f"Author {i}A", f"Author {i}B"],
                                "year": 2020 + (i % 5),
                                "abstract": f"This paper discusses {self.review_topic} with focus on aspect {i}...",
                                "citation_count": max(0, 100 - i * 2),
                                "database_source": db_name
                            }
                            for i in range(min(base_results, 20))  # Limit for demo
                        ]
                    }
                return search_database
            
            search_node = DAGNode(f"search_{database}", f"Search {database.upper()}", 
                                NodeType.PROCESSING, create_search_operation(database))
            search_node.research_phase = ResearchPhase.SEARCH.value
            search_nodes.append(search_node.node_id)
            
            self.add_node(search_node)
            self.add_edge("query_generation", search_node.node_id)
        
        # Add aggregation node to combine search results
        def aggregate_search_results(inputs):
            """Combine results from all databases"""
            all_results = []
            total_found = 0
            database_coverage = {}
            
            for node_id in search_nodes:
                if node_id in inputs:
                    search_data = inputs[node_id]
                    all_results.extend(search_data.get("results", []))
                    total_found += search_data.get("total_results", 0)
                    database_coverage[search_data.get("database")] = search_data.get("total_results", 0)
            
            # Remove duplicates based on title similarity
            unique_results = self._deduplicate_papers(all_results)
            
            return {
                "combined_results": unique_results,
                "total_papers_found": len(unique_results),
                "total_database_hits": total_found,
                "database_coverage": database_coverage,
                "deduplication_removed": len(all_results) - len(unique_results)
            }
        
        aggregation_node = DAGNode("search_aggregation", "Combine Search Results", 
                                  NodeType.AGGREGATION, aggregate_search_results)
        aggregation_node.research_phase = ResearchPhase.SEARCH.value
        
        self.add_node(aggregation_node)
        for search_node_id in search_nodes:
            self.add_edge(search_node_id, "search_aggregation")
    
    def _add_screening_phase(self):
        """Add title/abstract screening nodes"""
        
        def title_abstract_screening(inputs):
            """Screen papers based on title and abstract"""
            papers = inputs.get("search_aggregation", {}).get("combined_results", [])
            
            included_papers = []
            excluded_papers = []
            screening_reasons = {}
            
            for paper in papers:
                include_score = self._calculate_inclusion_score(paper)
                
                if include_score >= 0.6:  # Inclusion threshold
                    included_papers.append(paper)
                    screening_reasons[paper["id"]] = f"Included (score: {include_score:.2f})"
                else:
                    excluded_papers.append(paper)
                    screening_reasons[paper["id"]] = f"Excluded (score: {include_score:.2f})"
            
            return {
                "included_papers": included_papers,
                "excluded_papers": excluded_papers,
                "screening_reasons": screening_reasons,
                "inclusion_rate": len(included_papers) / len(papers) if papers else 0,
                "screening_statistics": {
                    "total_screened": len(papers),
                    "included": len(included_papers),
                    "excluded": len(excluded_papers)
                }
            }
        
        def quality_gate_check(inputs):
            """Quality gate for screening results"""
            screening_stats = inputs.get("title_screening", {}).get("screening_statistics", {})
            inclusion_rate = inputs.get("title_screening", {}).get("inclusion_rate", 0)
            
            # Quality checks
            quality_passed = True
            quality_issues = []
            
            if inclusion_rate < 0.1:
                quality_issues.append("Very low inclusion rate - criteria may be too strict")
                
            if inclusion_rate > 0.8:
                quality_issues.append("Very high inclusion rate - criteria may be too broad")
            
            if screening_stats.get("total_screened", 0) < 10:
                quality_issues.append("Insufficient papers for reliable analysis")
                quality_passed = False
            
            return {
                "quality_gate_passed": quality_passed,
                "quality_issues": quality_issues,
                "screening_quality_score": 1.0 - len(quality_issues) * 0.2,
                "recommendation": "proceed" if quality_passed else "revise_criteria"
            }
        
        # Create screening nodes
        screening_node = DAGNode("title_screening", "Title/Abstract Screening", 
                               NodeType.PROCESSING, title_abstract_screening)
        screening_node.research_phase = ResearchPhase.SCREENING.value
        
        quality_node = DAGNode("screening_quality", "Screening Quality Gate", 
                              NodeType.VALIDATION, quality_gate_check)
        quality_node.quality_requirements = {"min_papers": 10, "inclusion_rate_range": [0.1, 0.8]}
        
        self.add_node(screening_node)
        self.add_node(quality_node)
        self.add_edge("search_aggregation", "title_screening")
        self.add_edge("title_screening", "screening_quality")
    
    def _add_extraction_phase(self):
        """Add data extraction and full-text review nodes"""
        
        def full_text_review(inputs):
            """Conduct full-text review of included papers"""
            papers = inputs.get("title_screening", {}).get("included_papers", [])
            
            extracted_data = []
            exclusions_full_text = []
            
            for paper in papers[:10]:  # Limit for demo
                # Simulate full-text availability and review
                if hash(paper["id"]) % 4 != 0:  # 75% availability
                    extraction = {
                        "paper_id": paper["id"],
                        "title": paper["title"],
                        "methodology": f"Methodology for {paper['title'][:30]}",
                        "key_findings": [f"Finding 1 from {paper['id']}", f"Finding 2 from {paper['id']}"],
                        "limitations": f"Limitations discussed in {paper['id']}",
                        "quality_score": 0.6 + (hash(paper["id"]) % 40) / 100,
                        "data_extraction_complete": True
                    }
                    extracted_data.append(extraction)
                else:
                    exclusions_full_text.append({
                        "paper_id": paper["id"],
                        "exclusion_reason": "Full text not available"
                    })
            
            return {
                "extracted_data": extracted_data,
                "full_text_exclusions": exclusions_full_text,
                "extraction_success_rate": len(extracted_data) / len(papers) if papers else 0,
                "data_points_extracted": len(extracted_data) * 4  # methodology, findings, limitations, quality
            }
        
        extraction_node = DAGNode("data_extraction", "Full-text Review & Data Extraction", 
                                NodeType.PROCESSING, full_text_review)
        extraction_node.research_phase = ResearchPhase.EXTRACTION.value
        
        self.add_node(extraction_node)
        self.add_edge("screening_quality", "data_extraction")
    
    def _add_analysis_phase(self):
        """Add analysis and quality assessment nodes"""
        
        def thematic_analysis(inputs):
            """Perform thematic analysis of extracted data"""
            extracted_data = inputs.get("data_extraction", {}).get("extracted_data", [])
            
            # Identify themes from findings
            themes = {
                "methodological_approaches": [],
                "key_challenges": [],
                "emerging_trends": [],
                "future_directions": []
            }
            
            theme_frequency = {}
            
            for data in extracted_data:
                findings = data.get("key_findings", [])
                for finding in findings:
                    # Simple theme classification based on keywords
                    if any(word in finding.lower() for word in ["method", "approach", "technique"]):
                        themes["methodological_approaches"].append(finding)
                    elif any(word in finding.lower() for word in ["challenge", "problem", "limitation"]):
                        themes["key_challenges"].append(finding)
                    elif any(word in finding.lower() for word in ["trend", "emerging", "novel"]):
                        themes["emerging_trends"].append(finding)
                    elif any(word in finding.lower() for word in ["future", "direction", "recommendation"]):
                        themes["future_directions"].append(finding)
                
                # Count theme occurrences
                for theme_name, theme_items in themes.items():
                    theme_frequency[theme_name] = len(theme_items)
            
            return {
                "identified_themes": themes,
                "theme_frequency": theme_frequency,
                "total_themes_found": sum(theme_frequency.values()),
                "dominant_theme": max(theme_frequency.keys(), key=lambda k: theme_frequency[k]) if theme_frequency else None
            }
        
        def quality_assessment(inputs):
            """Assess overall quality of the research corpus"""
            extracted_data = inputs.get("data_extraction", {}).get("extracted_data", [])
            
            quality_metrics = {
                "average_quality_score": 0.0,
                "high_quality_papers": 0,
                "quality_distribution": {"high": 0, "medium": 0, "low": 0},
                "reliability_score": 0.0
            }
            
            if extracted_data:
                quality_scores = [data.get("quality_score", 0.5) for data in extracted_data]
                quality_metrics["average_quality_score"] = sum(quality_scores) / len(quality_scores)
                
                for score in quality_scores:
                    if score >= 0.8:
                        quality_metrics["quality_distribution"]["high"] += 1
                    elif score >= 0.6:
                        quality_metrics["quality_distribution"]["medium"] += 1
                    else:
                        quality_metrics["quality_distribution"]["low"] += 1
                
                quality_metrics["high_quality_papers"] = quality_metrics["quality_distribution"]["high"]
                quality_metrics["reliability_score"] = quality_metrics["average_quality_score"] * 0.7 + (len(extracted_data) / 50) * 0.3
            
            return {
                "quality_assessment": quality_metrics,
                "corpus_reliability": "high" if quality_metrics["reliability_score"] > 0.7 else "medium" if quality_metrics["reliability_score"] > 0.5 else "low",
                "recommendation": "sufficient" if quality_metrics["average_quality_score"] > 0.6 else "expand_search"
            }
        
        # Create analysis nodes
        thematic_node = DAGNode("thematic_analysis", "Thematic Analysis", 
                              NodeType.PROCESSING, thematic_analysis)
        thematic_node.research_phase = ResearchPhase.ANALYSIS.value
        
        quality_node = DAGNode("quality_assessment", "Quality Assessment", 
                              NodeType.VALIDATION, quality_assessment)
        quality_node.research_phase = ResearchPhase.ANALYSIS.value
        
        self.add_node(thematic_node)
        self.add_node(quality_node)
        self.add_edge("data_extraction", "thematic_analysis")
        self.add_edge("data_extraction", "quality_assessment")
    
    def _add_synthesis_phase(self):
        """Add synthesis and reporting nodes"""
        
        def synthesize_findings(inputs):
            """Synthesize all findings into coherent insights"""
            themes = inputs.get("thematic_analysis", {}).get("identified_themes", {})
            quality_assessment = inputs.get("quality_assessment", {}).get("quality_assessment", {})
            
            synthesis = {
                "executive_summary": f"Systematic review of {self.review_topic} identified {len(themes)} major themes",
                "key_insights": [],
                "research_gaps": [],
                "methodological_summary": {},
                "confidence_level": "high" if quality_assessment.get("reliability_score", 0) > 0.7 else "medium"
            }
            
            # Generate insights from themes
            for theme_name, theme_items in themes.items():
                if theme_items:
                    insight = f"{theme_name.replace('_', ' ').title()}: {len(theme_items)} findings identified"
                    synthesis["key_insights"].append(insight)
            
            # Identify research gaps
            if themes.get("future_directions"):
                synthesis["research_gaps"] = [
                    "Need for standardized methodologies",
                    "Limited long-term studies available", 
                    "Geographical bias in current research"
                ]
            
            return synthesis
        
        def generate_report(inputs):
            """Generate final research report"""
            synthesis = inputs.get("synthesis", {})
            planning = inputs.get("planning", {})
            search_stats = inputs.get("search_aggregation", {})
            
            report = {
                "title": f"Systematic Literature Review: {self.review_topic}",
                "research_question": planning.get("primary_question", ""),
                "methodology": {
                    "databases_searched": len(self.search_databases),
                    "total_papers_found": search_stats.get("total_papers_found", 0),
                    "papers_included": len(inputs.get("data_extraction", {}).get("extracted_data", [])),
                    "screening_approach": "title_abstract_then_fulltext"
                },
                "key_findings": synthesis.get("key_insights", []),
                "research_gaps": synthesis.get("research_gaps", []),
                "limitations": [
                    "Limited to English language papers",
                    "Search restricted to selected databases",
                    "Publication bias may affect results"
                ],
                "recommendations": [
                    "Future research should address identified gaps",
                    "Standardization of methodologies needed",
                    "Longitudinal studies recommended"
                ],
                "confidence_assessment": synthesis.get("confidence_level", "medium")
            }
            
            return report
        
        # Create synthesis nodes
        synthesis_node = DAGNode("synthesis", "Findings Synthesis", 
                               NodeType.AGGREGATION, synthesize_findings)
        synthesis_node.research_phase = ResearchPhase.SYNTHESIS.value
        
        report_node = DAGNode("final_report", "Report Generation", 
                            NodeType.OUTPUT, generate_report)
        report_node.research_phase = ResearchPhase.REPORTING.value
        
        self.add_node(synthesis_node)
        self.add_node(report_node)
        self.add_edge("thematic_analysis", "synthesis")
        self.add_edge("quality_assessment", "synthesis")
        self.add_edge("synthesis", "final_report")
    
    def _adapt_query_for_database(self, base_query: str, database: str) -> str:
        """Adapt search query for specific database syntax"""
        adaptations = {
            "arxiv": f"({base_query}) AND cat:cs.*",  # Computer Science category
            "pubmed": f"({base_query})[Title/Abstract]",  # PubMed syntax
            "ieee": f'("{base_query}") AND (Document Type:Conference OR Document Type:Journal)',
            "acm": f"({base_query}) AND (acmdlTitle:({base_query}) OR acmdlAbstract:({base_query}))"
        }
        return adaptations.get(database, base_query)
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title_normalized = paper["title"].lower().strip()
            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _calculate_inclusion_score(self, paper: Dict) -> float:
        """Calculate inclusion score based on criteria"""
        score = 0.5  # Base score
        
        # Year check
        if paper.get("year", 2020) >= self.criteria.min_publication_year:
            score += 0.2
        
        # Keyword relevance
        title_abstract = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        inclusion_matches = sum(1 for keyword in self.criteria.inclusion_keywords 
                              if keyword.lower() in title_abstract)
        exclusion_matches = sum(1 for keyword in self.criteria.exclusion_keywords 
                               if keyword.lower() in title_abstract)
        
        score += (inclusion_matches * 0.1) - (exclusion_matches * 0.2)
        
        # Citation count
        if paper.get("citation_count", 0) >= self.criteria.minimum_citation_count:
            score += 0.1
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1


class ResearchPipeline:
    """
    High-level interface for creating and managing research pipelines.
    
    Simplifies the creation of complex research DAGs with common patterns.
    """
    
    @staticmethod
    def create_literature_review(topic: str, inclusion_keywords: List[str], 
                               exclusion_keywords: List[str] = None) -> LiteratureReviewDAG:
        """Create a standard literature review pipeline"""
        criteria = ResearchCriteria(
            inclusion_keywords=inclusion_keywords,
            exclusion_keywords=exclusion_keywords or [],
            min_publication_year=2020,
            minimum_citation_count=0
        )
        
        return LiteratureReviewDAG(topic, criteria)
    
    @staticmethod
    def create_meta_analysis(topic: str, effect_size_measure: str = "cohen_d") -> ResearchDAG:
        """Create a meta-analysis pipeline (simplified)"""
        dag = ResearchDAG(f"meta_analysis_{topic}", f"Meta-analysis of {topic}")
        
        # Add meta-analysis specific nodes
        # (Implementation simplified for demo)
        
        return dag
    
    @staticmethod
    def create_comparative_analysis(topics: List[str]) -> ResearchDAG:
        """Create a comparative analysis pipeline"""
        dag = ResearchDAG("comparative_analysis", f"Comparative analysis of {', '.join(topics)}")
        
        # Add comparative analysis nodes
        # (Implementation simplified for demo)
        
        return dag


def demonstrate_research_pipeline():
    """Demonstrate research pipeline DAG capabilities"""
    
    print("📚 RESEARCH PIPELINE DAG DEMONSTRATION")
    print("=" * 50)
    
    # Create a literature review pipeline
    print("\\n🔬 Creating Literature Review Pipeline...")
    
    criteria = ResearchCriteria(
        inclusion_keywords=["quantum", "error correction", "fault tolerance"],
        exclusion_keywords=["classical", "obsolete"],
        min_publication_year=2020
    )
    
    review_dag = LiteratureReviewDAG("Quantum Error Correction", criteria)
    
    print(f"✅ Created DAG with {len(review_dag.nodes)} nodes")
    print(f"📋 Research phases: {len(set(node.research_phase for node in review_dag.nodes.values()))}")
    
    # Show DAG structure
    print(review_dag.visualize_structure())
    
    print("\\n🚀 Executing Literature Review Pipeline...")
    try:
        start_time = time.time()
        results = review_dag.execute()
        execution_time = time.time() - start_time
        
        print(f"✅ Pipeline completed in {execution_time:.2f} seconds")
        
        # Show key results
        if "final_report" in results:
            report = results["final_report"]
            print(f"\\n📊 Review Results:")
            print(f"   • Title: {report.get('title', 'N/A')}")
            print(f"   • Papers found: {report.get('methodology', {}).get('total_papers_found', 0)}")
            print(f"   • Papers included: {report.get('methodology', {}).get('papers_included', 0)}")
            print(f"   • Key findings: {len(report.get('key_findings', []))}")
            print(f"   • Confidence: {report.get('confidence_assessment', 'N/A')}")
        
        # Show execution statistics
        stats = review_dag.get_statistics()
        print(f"\\n📈 Execution Statistics:")
        print(f"   • Success rate: {stats['success_rate']:.1%}")
        print(f"   • Total nodes: {stats['total_nodes']}")
        print(f"   • Node types: {stats['nodes_by_type']}")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_research_pipeline()