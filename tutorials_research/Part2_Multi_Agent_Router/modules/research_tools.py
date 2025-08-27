"""
Research Tools API Integration

Real API wrappers for academic research databases and services.
Provides unified interface for ArXiv, Semantic Scholar, CrossRef, and other research tools.
"""

import os
import time
import requests
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import quote
import xml.etree.ElementTree as ET


@dataclass
class ResearchPaper:
    """Standardized paper representation across different APIs"""
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    venue: Optional[str] = None
    citation_count: int = 0
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    pdf_url: Optional[str] = None
    confidence_score: float = 0.0
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "publication_date": self.publication_date,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "categories": self.categories,
            "keywords": self.keywords,
            "pdf_url": self.pdf_url,
            "confidence_score": self.confidence_score,
            "source": self.source
        }


class ArXivClient:
    """
    ArXiv API client for academic paper search and retrieval.
    
    Provides access to arXiv.org preprint repository with over 2 million papers
    in physics, mathematics, computer science, and related fields.
    """
    
    def __init__(self, base_url: str = "http://export.arxiv.org/api/query"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ResearchAgent/1.0 (Academic Research Tool)'
        })
        
        # Rate limiting (arXiv recommends max 3 requests per second)
        self.min_request_interval = 0.5  # seconds
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting to respect arXiv guidelines"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def search_papers(self, 
                     query: str,
                     max_results: int = 20,
                     sort_by: str = "relevance",
                     start_date: Optional[str] = None,
                     categories: Optional[List[str]] = None) -> List[ResearchPaper]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search terms (supports boolean operators)
            max_results: Maximum papers to return (default 20, max 30000)
            sort_by: Sort order ("relevance", "lastUpdatedDate", "submittedDate")
            start_date: Filter papers after this date (YYYY-MM-DD format)
            categories: Filter by arXiv categories (e.g., ["cs.AI", "cs.LG"])
            
        Returns:
            List of ResearchPaper objects
        """
        self._rate_limit()
        
        # Build search query
        search_query = f"all:{query}"
        
        # Add category filters
        if categories:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query += f" AND ({category_filter})"
        
        # Add date filter
        if start_date:
            search_query += f" AND submittedDate:[{start_date}0101 TO {datetime.now().strftime('%Y%m%d')}]"
        
        # Prepare parameters
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 30000),  # arXiv limit
            "sortBy": sort_by,
            "sortOrder": "descending" if sort_by != "relevance" else "descending"
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            return self._parse_arxiv_response(response.text)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ ArXiv API error: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_response: str) -> List[ResearchPaper]:
        """Parse arXiv XML response into ResearchPaper objects"""
        papers = []
        
        try:
            root = ET.fromstring(xml_response)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Extract papers
            for entry in root.findall('atom:entry', namespaces):
                try:
                    # Basic information
                    title = entry.find('atom:title', namespaces).text.strip()
                    abstract = entry.find('atom:summary', namespaces).text.strip()
                    
                    # Authors
                    authors = []
                    for author in entry.findall('atom:author', namespaces):
                        name = author.find('atom:name', namespaces).text
                        authors.append(name)
                    
                    # Dates and IDs
                    published = entry.find('atom:published', namespaces).text[:10]  # YYYY-MM-DD
                    arxiv_id = entry.find('atom:id', namespaces).text.split('/')[-1]
                    
                    # Links
                    pdf_url = None
                    abs_url = None
                    for link in entry.findall('atom:link', namespaces):
                        if link.get('type') == 'application/pdf':
                            pdf_url = link.get('href')
                        elif link.get('type') == 'text/html':
                            abs_url = link.get('href')
                    
                    # Categories
                    categories = []
                    for category in entry.findall('atom:category', namespaces):
                        categories.append(category.get('term'))
                    
                    # Create paper object
                    paper = ResearchPaper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        publication_date=published,
                        arxiv_id=arxiv_id,
                        url=abs_url,
                        pdf_url=pdf_url,
                        categories=categories,
                        source="arxiv"
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    print(f"⚠️ Error parsing arXiv entry: {e}")
                    continue
                    
        except ET.ParseError as e:
            print(f"❌ Error parsing arXiv XML response: {e}")
            
        return papers
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[ResearchPaper]:
        """Retrieve specific paper by arXiv ID"""
        results = self.search_papers(f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None


class SemanticScholarClient:
    """
    Semantic Scholar API client for paper search and citation analysis.
    
    Provides access to Semantic Scholar's database with citation networks,
    author information, and paper influence metrics.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key or os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'x-api-key': self.api_key
            })
        
        # Rate limiting (free tier: 1 request per second, paid: 10/sec)
        self.min_request_interval = 1.0 if not self.api_key else 0.1
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting for Semantic Scholar API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def search_papers(self, 
                     query: str,
                     max_results: int = 20,
                     fields: Optional[List[str]] = None) -> List[ResearchPaper]:
        """
        Search Semantic Scholar for papers.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            fields: Specific fields to retrieve
            
        Returns:
            List of ResearchPaper objects
        """
        self._rate_limit()
        
        if not fields:
            fields = ["title", "authors", "abstract", "year", "citationCount", 
                     "url", "venue", "externalIds", "publicationDate"]
        
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API limit
            "fields": ",".join(fields)
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/paper/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_semantic_scholar_response(data)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Semantic Scholar API error: {e}")
            return []
    
    def _parse_semantic_scholar_response(self, data: Dict) -> List[ResearchPaper]:
        """Parse Semantic Scholar API response"""
        papers = []
        
        for item in data.get('data', []):
            try:
                # Basic information
                title = item.get('title', 'No title')
                abstract = item.get('abstract', 'No abstract available')
                
                # Authors
                authors = []
                for author_info in item.get('authors', []):
                    authors.append(author_info.get('name', 'Unknown'))
                
                # Publication info
                year = item.get('year', 'Unknown')
                pub_date = item.get('publicationDate', f"{year}-01-01" if year != 'Unknown' else 'Unknown')
                venue = item.get('venue', 'Unknown venue')
                citation_count = item.get('citationCount', 0)
                
                # External IDs
                external_ids = item.get('externalIds', {})
                doi = external_ids.get('DOI')
                arxiv_id = external_ids.get('ArXiv')
                
                # Create paper object
                paper = ResearchPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    publication_date=pub_date,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    url=item.get('url'),
                    venue=venue,
                    citation_count=citation_count,
                    source="semantic_scholar"
                )
                
                papers.append(paper)
                
            except Exception as e:
                print(f"⚠️ Error parsing Semantic Scholar entry: {e}")
                continue
        
        return papers
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Get detailed information about a specific paper"""
        self._rate_limit()
        
        fields = ["title", "authors", "abstract", "year", "citationCount", 
                 "citations", "references", "venue", "externalIds"]
        
        try:
            response = self.session.get(
                f"{self.base_url}/paper/{paper_id}",
                params={"fields": ",".join(fields)},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching paper details: {e}")
            return None


class CrossRefClient:
    """
    CrossRef API client for DOI resolution and citation metadata.
    
    Provides access to CrossRef's database for citation formatting
    and bibliographic metadata retrieval.
    """
    
    def __init__(self, mailto: Optional[str] = None):
        self.base_url = "https://api.crossref.org"
        self.mailto = mailto or os.getenv('CROSSREF_MAILTO')
        
        self.session = requests.Session()
        if self.mailto:
            self.session.headers.update({
                'User-Agent': f'ResearchAgent/1.0 (mailto:{self.mailto})'
            })
        
        # Rate limiting (polite usage)
        self.min_request_interval = 0.1
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement polite rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def resolve_doi(self, doi: str) -> Optional[Dict]:
        """Resolve DOI to get full bibliographic information"""
        self._rate_limit()
        
        try:
            response = self.session.get(
                f"{self.base_url}/works/{doi}",
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('message', {})
            
        except requests.exceptions.RequestException as e:
            print(f"❌ CrossRef DOI resolution error: {e}")
            return None
    
    def search_works(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search CrossRef works database"""
        self._rate_limit()
        
        params = {
            "query": query,
            "rows": min(max_results, 1000)  # API limit
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/works",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('message', {}).get('items', [])
            
        except requests.exceptions.RequestException as e:
            print(f"❌ CrossRef search error: {e}")
            return []


class ResearchToolsAggregator:
    """
    Unified interface for multiple research APIs.
    
    Combines results from ArXiv, Semantic Scholar, and CrossRef
    to provide comprehensive research capabilities.
    """
    
    def __init__(self, 
                 semantic_scholar_api_key: Optional[str] = None,
                 crossref_mailto: Optional[str] = None):
        self.arxiv = ArXivClient()
        self.semantic_scholar = SemanticScholarClient(semantic_scholar_api_key)
        self.crossref = CrossRefClient(crossref_mailto)
    
    def search_all_sources(self, 
                          query: str,
                          max_results_per_source: int = 10) -> Dict[str, List[ResearchPaper]]:
        """
        Search all available sources and return aggregated results.
        
        Args:
            query: Search query
            max_results_per_source: Max results from each source
            
        Returns:
            Dictionary mapping source names to paper lists
        """
        results = {}
        
        print(f"🔍 Searching ArXiv for: {query}")
        results['arxiv'] = self.arxiv.search_papers(query, max_results_per_source)
        
        print(f"🔍 Searching Semantic Scholar for: {query}")
        results['semantic_scholar'] = self.semantic_scholar.search_papers(query, max_results_per_source)
        
        return results
    
    def get_comprehensive_results(self, 
                                query: str,
                                max_total_results: int = 20) -> List[ResearchPaper]:
        """
        Get comprehensive results from all sources, deduplicated and ranked.
        
        Args:
            query: Search query
            max_total_results: Maximum total results to return
            
        Returns:
            Ranked and deduplicated list of papers
        """
        all_sources = self.search_all_sources(query, max_total_results // 2)
        
        # Combine all results
        all_papers = []
        for source, papers in all_sources.items():
            all_papers.extend(papers)
        
        # Deduplicate by title similarity
        deduplicated = self._deduplicate_papers(all_papers)
        
        # Rank by relevance (simple scoring)
        ranked = self._rank_papers(deduplicated, query)
        
        return ranked[:max_total_results]
    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Simple deduplication by title (could be improved with fuzzy matching)
            title_key = paper.title.lower().strip() if paper.title else ""
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _rank_papers(self, papers: List[ResearchPaper], query: str) -> List[ResearchPaper]:
        """Rank papers by relevance to query"""
        # Simple ranking based on citation count and query term presence
        query_terms = set(query.lower().split())
        
        for paper in papers:
            score = 0
            
            # Citation count (normalize to 0-1)
            score += min(paper.citation_count / 100, 1.0) * 0.3
            
            # Title relevance
            title_terms = set(paper.title.lower().split()) if paper.title else set()
            title_overlap = len(query_terms.intersection(title_terms)) / max(len(query_terms), 1)
            score += title_overlap * 0.4
            
            # Abstract relevance
            abstract_terms = set(paper.abstract.lower().split()) if paper.abstract else set()
            abstract_overlap = len(query_terms.intersection(abstract_terms)) / max(len(query_terms), 1)
            score += abstract_overlap * 0.3
            
            paper.confidence_score = score
        
        # Sort by confidence score
        return sorted(papers, key=lambda p: p.confidence_score, reverse=True)


def demonstrate_research_tools():
    """Demonstrate the research tools capabilities"""
    
    print("🔬 RESEARCH TOOLS DEMONSTRATION")
    print("=" * 35)
    
    print("\n📚 Available Tools:")
    tools = [
        ("ArXiv Client", "Preprint repository search (2M+ papers)"),
        ("Semantic Scholar", "Citation analysis and paper metrics"),
        ("CrossRef", "DOI resolution and bibliographic data"),
        ("Aggregator", "Unified search across all sources")
    ]
    
    for tool, description in tools:
        print(f"   • {tool}: {description}")
    
    print("\n🎯 Research Capabilities:")
    capabilities = [
        "Literature search across multiple databases",
        "Citation count and influence metrics",
        "Author collaboration networks",
        "Paper categorization and filtering",
        "Full-text and metadata access",
        "DOI resolution and validation"
    ]
    
    for capability in capabilities:
        print(f"   • {capability}")
    
    print("\n💡 Usage Examples:")
    examples = [
        'arxiv.search_papers("transformer attention", max_results=10)',
        'semantic_scholar.search_papers("neural networks", max_results=20)',
        'crossref.resolve_doi("10.1038/nature12373")',
        'aggregator.get_comprehensive_results("machine learning")'
    ]
    
    for example in examples:
        print(f"   • {example}")


if __name__ == "__main__":
    demonstrate_research_tools()
    
    print("\\n" + "="*50)
    print("🧪 TESTING RESEARCH TOOLS")
    print("="*50)
    
    # Test ArXiv (no API key required)
    try:
        print("\\n🔍 Testing ArXiv search...")
        arxiv_client = ArXivClient()
        results = arxiv_client.search_papers("attention mechanism", max_results=3)
        print(f"✅ Found {len(results)} papers from ArXiv")
        
        if results:
            print(f"   Sample: {results[0].title[:100]}...")
    except Exception as e:
        print(f"❌ ArXiv test failed: {e}")
    
    # Test aggregator
    try:
        print("\\n🔍 Testing aggregated search...")
        aggregator = ResearchToolsAggregator()
        results = aggregator.get_comprehensive_results("machine learning", max_total_results=5)
        print(f"✅ Found {len(results)} papers total")
        
        if results:
            print("\\n📊 Top result:")
            top_paper = results[0]
            print(f"   Title: {top_paper.title}")
            print(f"   Authors: {', '.join(top_paper.authors[:3])}")
            print(f"   Source: {top_paper.source}")
            print(f"   Confidence: {top_paper.confidence_score:.2f}")
    except Exception as e:
        print(f"❌ Aggregator test failed: {e}")
    
    print("\\n💡 To use with API keys:")
    print("Set environment variables:")
    print("   SEMANTIC_SCHOLAR_API_KEY=your_key_here")
    print("   CROSSREF_MAILTO=your_email@domain.com")