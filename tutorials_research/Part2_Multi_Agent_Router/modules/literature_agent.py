"""
Literature Search Agent

Specialized agent for conducting comprehensive literature searches,
ranking results, and extracting key insights for research applications.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from research_tools import ResearchToolsAggregator, ResearchPaper


@dataclass
class LiteratureSearchResult:
    """Structured result from literature search"""
    query: str
    papers: List[ResearchPaper]
    total_found: int
    search_time: float
    filters_applied: Dict[str, Any]
    key_insights: List[str] = field(default_factory=list)
    trending_topics: List[str] = field(default_factory=list)
    author_networks: List[str] = field(default_factory=list)
    
    def to_summary(self) -> str:
        """Generate a human-readable summary"""
        summary = f"Literature Search: '{self.query}'\\n"
        summary += f"Found {self.total_found} papers in {self.search_time:.1f}s\\n\\n"
        
        if self.papers:
            summary += "Top Papers:\\n"
            for i, paper in enumerate(self.papers[:5], 1):
                summary += f"{i}. {paper.title}\\n"
                summary += f"   Authors: {', '.join(paper.authors[:3])}\\n"
                summary += f"   Citations: {paper.citation_count} | Year: {paper.publication_date[:4]}\\n\\n"
        
        if self.key_insights:
            summary += "Key Insights:\\n"
            for insight in self.key_insights:
                summary += f"• {insight}\\n"
        
        return summary


class LiteratureSearchAgent:
    """
    Specialized agent for literature search and analysis.
    
    Capabilities:
    - Multi-source paper search (ArXiv, Semantic Scholar)
    - Query optimization and expansion
    - Result ranking and filtering
    - Key insight extraction
    - Trend analysis and author networks
    """
    
    def __init__(self, llm, research_tools: Optional[ResearchToolsAggregator] = None):
        self.llm = llm
        self.research_tools = research_tools or ResearchToolsAggregator()
        
        # Agent configuration
        self.max_papers_per_query = 50
        self.default_time_filter_years = 5
        
        # Search optimization patterns
        self.query_expansion_terms = {
            "machine learning": ["ML", "artificial intelligence", "deep learning", "neural networks"],
            "natural language processing": ["NLP", "text processing", "language models", "text mining"],
            "computer vision": ["image processing", "visual recognition", "computer graphics"],
            "transformer": ["attention mechanism", "BERT", "GPT", "language model"],
            "deep learning": ["neural networks", "CNN", "RNN", "LSTM", "GAN"]
        }
        
        # Quality assessment criteria
        self.quality_metrics = {
            "min_citations": 10,
            "preferred_venues": ["nature", "science", "proceedings", "journal", "conference"],
            "recency_weight": 0.3,
            "citation_weight": 0.4,
            "relevance_weight": 0.3
        }
    
    def search_literature(self,
                         query: str,
                         max_results: int = 20,
                         include_recent_only: bool = True,
                         expand_query: bool = True,
                         filter_by_citations: bool = True) -> LiteratureSearchResult:
        """
        Conduct comprehensive literature search.
        
        Args:
            query: Search query
            max_results: Maximum papers to return
            include_recent_only: Filter to recent papers (last 5 years)
            expand_query: Automatically expand query terms
            filter_by_citations: Filter low-citation papers
            
        Returns:
            LiteratureSearchResult with papers and analysis
        """
        import time
        start_time = time.time()
        
        print(f"📚 Starting literature search for: '{query}'")
        
        # Query expansion
        expanded_query = self._expand_query(query) if expand_query else query
        if expanded_query != query:
            print(f"   Expanded to: '{expanded_query}'")
        
        # Set up filters
        filters = {
            "original_query": query,
            "expanded_query": expanded_query,
            "max_results": max_results,
            "recent_only": include_recent_only,
            "citation_filter": filter_by_citations
        }
        
        # Search all sources
        try:
            papers = self.research_tools.get_comprehensive_results(
                expanded_query, 
                max_total_results=max_results * 2  # Get more for filtering
            )
            print(f"   Found {len(papers)} papers from all sources")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return LiteratureSearchResult(
                query=query,
                papers=[],
                total_found=0,
                search_time=time.time() - start_time,
                filters_applied=filters
            )
        
        # Apply filters
        filtered_papers = self._apply_filters(papers, filters)
        print(f"   {len(filtered_papers)} papers after filtering")
        
        # Final ranking and selection
        final_papers = self._rank_and_select(filtered_papers, query, max_results)
        print(f"   Selected top {len(final_papers)} papers")
        
        # Generate insights
        key_insights = self._extract_insights(final_papers, query)
        trending_topics = self._identify_trends(final_papers)
        
        search_time = time.time() - start_time
        
        result = LiteratureSearchResult(
            query=query,
            papers=final_papers,
            total_found=len(papers),
            search_time=search_time,
            filters_applied=filters,
            key_insights=key_insights,
            trending_topics=trending_topics
        )
        
        print(f"✅ Literature search completed in {search_time:.1f}s")
        return result
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        query_lower = query.lower()
        
        # Find matching expansion terms
        for key_term, expansions in self.query_expansion_terms.items():
            if key_term in query_lower:
                # Add most relevant expansion terms
                additional_terms = expansions[:2]  # Limit to avoid too broad search
                expanded = f"{query} OR {' OR '.join(additional_terms)}"
                return expanded
        
        return query
    
    def _apply_filters(self, papers: List[ResearchPaper], filters: Dict[str, Any]) -> List[ResearchPaper]:
        """Apply various filters to paper list"""
        filtered = papers.copy()
        
        # Recent papers filter
        if filters.get("recent_only", False):
            cutoff_year = datetime.now().year - self.default_time_filter_years
            filtered = [p for p in filtered if self._extract_year(p.publication_date) >= cutoff_year]
        
        # Citation filter
        if filters.get("citation_filter", False):
            min_citations = self.quality_metrics["min_citations"]
            filtered = [p for p in filtered if p.citation_count >= min_citations]
        
        return filtered
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from date string"""
        try:
            # Handle various date formats
            if date_str and len(date_str) >= 4:
                year_match = re.search(r'(19|20)\\d{2}', date_str)
                if year_match:
                    return int(year_match.group())
            return 1900  # Default for unknown dates
        except:
            return 1900
    
    def _rank_and_select(self, papers: List[ResearchPaper], query: str, max_results: int) -> List[ResearchPaper]:
        """Advanced ranking and selection of papers"""
        
        # Calculate comprehensive scores
        for paper in papers:
            score = 0
            
            # Citation score (normalized)
            citation_score = min(paper.citation_count / 100, 1.0)
            score += citation_score * self.quality_metrics["citation_weight"]
            
            # Recency score
            paper_year = self._extract_year(paper.publication_date)
            current_year = datetime.now().year
            recency_score = max(0, 1 - (current_year - paper_year) / 10)  # Decay over 10 years
            score += recency_score * self.quality_metrics["recency_weight"]
            
            # Relevance score (already calculated in research_tools)
            relevance_score = paper.confidence_score
            score += relevance_score * self.quality_metrics["relevance_weight"]
            
            # Venue quality bonus
            venue_lower = (paper.venue or "").lower()
            for preferred in self.quality_metrics["preferred_venues"]:
                if preferred in venue_lower:
                    score += 0.1
                    break
            
            paper.confidence_score = score
        
        # Sort and select top papers
        ranked_papers = sorted(papers, key=lambda p: p.confidence_score, reverse=True)
        return ranked_papers[:max_results]
    
    def _extract_insights(self, papers: List[ResearchPaper], query: str) -> List[str]:
        """Extract key insights from paper collection using LLM"""
        
        if not papers or len(papers) < 3:
            return ["Insufficient papers for insight extraction"]
        
        # Prepare paper summaries for LLM analysis
        paper_summaries = []
        for i, paper in enumerate(papers[:10], 1):  # Limit to top 10 for LLM processing
            summary = f"{i}. {paper.title}\\n"
            summary += f"   Authors: {', '.join(paper.authors[:3])}\\n"
            summary += f"   Year: {self._extract_year(paper.publication_date)}\\n"
            summary += f"   Citations: {paper.citation_count}\\n"
            summary += f"   Abstract: {paper.abstract[:300]}...\\n"
            paper_summaries.append(summary)
        
        # LLM prompt for insight extraction
        prompt = f"""Analyze these research papers related to '{query}' and extract key insights:

{chr(10).join(paper_summaries)}

Please provide 3-5 key insights about:
1. Main research trends and directions
2. Most influential authors and institutions
3. Emerging methodologies or approaches
4. Gaps or future research opportunities
5. Practical applications and impact

Format as bullet points, keep each insight concise (1-2 sentences)."""

        try:
            response = self.llm.invoke(prompt)
            insights_text = response.content
            
            # Parse bullet points
            insights = []
            for line in insights_text.split('\\n'):
                line = line.strip()
                if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                    clean_insight = re.sub(r'^[•\\-*0-9.\\s]+', '', line).strip()
                    if clean_insight:
                        insights.append(clean_insight)
            
            return insights[:5] if insights else ["Analysis completed successfully"]
            
        except Exception as e:
            return [f"Insight extraction failed: {str(e)}"]
    
    def _identify_trends(self, papers: List[ResearchPaper]) -> List[str]:
        """Identify trending topics from paper keywords and titles"""
        
        # Collect terms from titles and categories
        all_terms = []
        
        for paper in papers:
            # Extract terms from title
            title_terms = re.findall(r'\\b[a-zA-Z]{3,}\\b', paper.title.lower()) if paper.title else []
            all_terms.extend(title_terms)
            
            # Add categories
            all_terms.extend([cat.lower() for cat in paper.categories])
        
        # Count term frequency
        term_counts = {}
        for term in all_terms:
            if len(term) >= 4:  # Filter short terms
                term_counts[term] = term_counts.get(term, 0) + 1
        
        # Get top trending terms
        trending = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Filter and format
        filtered_trends = []
        stop_words = {'paper', 'study', 'research', 'analysis', 'approach', 'method', 'using', 'based'}
        
        for term, count in trending:
            if term not in stop_words and count >= 3:  # Must appear in at least 3 papers
                filtered_trends.append(f"{term.title()} ({count} papers)")
        
        return filtered_trends[:5]
    
    def generate_literature_review_section(self, 
                                         search_result: LiteratureSearchResult,
                                         section_focus: str = "overview") -> str:
        """
        Generate a literature review section using LLM.
        
        Args:
            search_result: Results from literature search
            section_focus: Focus of the review ("overview", "methods", "trends", "gaps")
            
        Returns:
            Formatted literature review text
        """
        
        if not search_result.papers:
            return "No papers found for literature review generation."
        
        # Prepare paper information
        papers_info = []
        for paper in search_result.papers[:15]:  # Limit for LLM context
            paper_info = {
                "title": paper.title,
                "authors": paper.authors[:3],  # Limit authors
                "year": self._extract_year(paper.publication_date),
                "citations": paper.citation_count,
                "abstract": paper.abstract[:200],  # Truncate for context
                "venue": paper.venue or "Unknown"
            }
            papers_info.append(paper_info)
        
        # Generate LLM prompt based on focus
        focus_prompts = {
            "overview": "Provide a comprehensive overview of the research landscape",
            "methods": "Focus on methodological approaches and techniques used",
            "trends": "Emphasize recent trends and emerging directions",
            "gaps": "Identify research gaps and future opportunities"
        }
        
        focus_instruction = focus_prompts.get(section_focus, focus_prompts["overview"])
        
        prompt = f"""Write a literature review section about '{search_result.query}' based on these papers:

{self._format_papers_for_llm(papers_info)}

Instructions:
- {focus_instruction}
- Write in academic style suitable for a research paper
- Include proper citations using author-year format
- Organize into coherent paragraphs
- Length: 300-500 words
- Highlight key contributions and findings

Structure the review with:
1. Opening statement about the research area
2. Discussion of major contributions and approaches
3. Current trends or methodological insights
4. Brief conclusion about the state of research"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Literature review generation failed: {str(e)}"
    
    def _format_papers_for_llm(self, papers_info: List[Dict]) -> str:
        """Format paper information for LLM processing"""
        formatted = []
        
        for i, paper in enumerate(papers_info, 1):
            authors_str = ", ".join(paper["authors"])
            if len(paper["authors"]) > 3:
                authors_str += " et al."
            
            paper_text = f"{i}. {paper['title']} ({paper['year']})\\n"
            paper_text += f"   Authors: {authors_str}\\n"
            paper_text += f"   Venue: {paper['venue']} | Citations: {paper['citations']}\\n"
            paper_text += f"   Abstract: {paper['abstract']}...\\n"
            
            formatted.append(paper_text)
        
        return "\\n".join(formatted)
    
    def get_author_collaboration_network(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze author collaboration networks from paper collection"""
        
        author_stats = {}
        collaborations = {}
        
        for paper in papers:
            authors = paper.authors
            
            # Update author stats
            for author in authors:
                if author not in author_stats:
                    author_stats[author] = {
                        "papers": 0,
                        "total_citations": 0,
                        "collaborators": set()
                    }
                
                author_stats[author]["papers"] += 1
                author_stats[author]["total_citations"] += paper.citation_count
                
                # Track collaborations
                for other_author in authors:
                    if other_author != author:
                        author_stats[author]["collaborators"].add(other_author)
        
        # Find most collaborative authors
        top_authors = sorted(
            author_stats.items(),
            key=lambda x: (x[1]["papers"], len(x[1]["collaborators"])),
            reverse=True
        )[:10]
        
        return {
            "top_authors": [(name, stats["papers"], len(stats["collaborators"])) 
                           for name, stats in top_authors],
            "total_unique_authors": len(author_stats),
            "average_collaborators": sum(len(stats["collaborators"]) 
                                       for stats in author_stats.values()) / len(author_stats)
        }


def demonstrate_literature_agent():
    """Demonstrate literature search agent capabilities"""
    
    print("📚 LITERATURE SEARCH AGENT DEMONSTRATION")
    print("=" * 45)
    
    print("\n🎯 Agent Capabilities:")
    capabilities = [
        "Multi-source literature search (ArXiv + Semantic Scholar)",
        "Query expansion and optimization",
        "Advanced paper ranking and filtering",
        "Key insight extraction using LLM analysis",
        "Trend identification and author network analysis",
        "Automated literature review section generation"
    ]
    
    for capability in capabilities:
        print(f"   • {capability}")
    
    print("\n📊 Search Features:")
    features = [
        "Recent paper filtering (last 5 years)",
        "Citation-based quality filtering",
        "Multi-criteria ranking (relevance, recency, citations)",
        "Venue quality assessment",
        "Duplicate detection and removal"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    print("\n🔍 Usage Examples:")
    examples = [
        'agent.search_literature("transformer attention mechanisms")',
        'agent.generate_literature_review_section(results, "trends")',
        'agent.get_author_collaboration_network(papers)',
    ]
    
    for example in examples:
        print(f"   • {example}")


if __name__ == "__main__":
    demonstrate_literature_agent()
    
    print("\\n" + "="*50)
    print("🧪 LITERATURE AGENT TESTING")
    print("="*50)
    
    print("\\n💡 To test the literature agent:")
    print("""
from research_tools import ResearchToolsAggregator
from literature_agent import LiteratureSearchAgent
from llm_providers import create_research_llm

# Set up components
llm = create_research_llm('google')  # or your preferred provider
research_tools = ResearchToolsAggregator()
literature_agent = LiteratureSearchAgent(llm, research_tools)

# Conduct literature search
results = literature_agent.search_literature(
    "transformer attention mechanisms",
    max_results=10,
    include_recent_only=True
)

print(results.to_summary())

# Generate literature review section
review = literature_agent.generate_literature_review_section(results, "trends")
print(review)
""")