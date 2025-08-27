"""
Web Search Execution for Research Workflows

Executes web searches using Google Search API with parallel processing
and source validation for academic research applications.
"""

import os
import time
import asyncio
import aiohttp
import requests
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import re
from urllib.parse import quote, urlparse

from state_schemas import ResearchSource, ResearchConfiguration


class WebSearchExecutor:
    """
    Web search executor with Google Search API integration.
    
    Provides both free (limited) and API key (enhanced) search capabilities
    with parallel execution and source validation.
    """
    
    def __init__(self, config: Optional[ResearchConfiguration] = None):
        self.config = config or ResearchConfiguration()
        
        # Google Search API configuration
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        # Alternative: SerpAPI (if Google Custom Search not available)
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        
        # Search configuration
        self.base_google_url = "https://www.googleapis.com/customsearch/v1"
        self.base_serpapi_url = "https://serpapi.com/search.json"
        
        # Rate limiting
        self.requests_per_second = 10
        self.last_request_time = 0
        self.min_request_interval = 1.0 / self.requests_per_second
        
        # Source validation patterns
        self.trusted_domains = {
            'academic': [
                'arxiv.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov',
                'ieee.org', 'acm.org', 'nature.com', 'science.org',
                'springer.com', 'elsevier.com', 'wiley.com', 'cambridge.org'
            ],
            'institutional': [
                '.edu', '.gov', '.ac.uk', '.ac.in', '.org'
            ],
            'technical': [
                'github.com', 'stackoverflow.com', 'medium.com/@',
                'towardsdatascience.com', 'distill.pub'
            ]
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ResearchAgent/1.0 (Academic Research Tool)'
        })
    
    def execute_parallel_search(self, queries: List[str]) -> Dict[str, List[ResearchSource]]:
        """
        Execute multiple search queries in parallel.
        
        Args:
            queries: List of search queries to execute
            
        Returns:
            Dictionary mapping queries to their search results
        """
        print(f"🔍 Executing parallel search for {len(queries)} queries...")
        
        if self.config.parallel_search_enabled and len(queries) > 1:
            return self._execute_parallel_threaded(queries)
        else:
            return self._execute_sequential(queries)
    
    def search_single_query(self, query: str) -> List[ResearchSource]:
        """
        Execute a single search query.
        
        Args:
            query: Search query string
            
        Returns:
            List of ResearchSource objects
        """
        
        self._rate_limit()
        
        print(f"   🔎 Searching: '{query[:50]}...'")
        
        # Try different search methods in order of preference
        sources = []
        
        # Method 1: Google Custom Search API (if available)
        if self.google_api_key and self.search_engine_id:
            sources = self._search_google_custom(query)
        
        # Method 2: SerpAPI (if available)
        elif self.serpapi_key:
            sources = self._search_serpapi(query)
        
        # Method 3: Fallback to basic web search simulation
        else:
            sources = self._search_fallback(query)
        
        # Validate and score sources
        validated_sources = self._validate_sources(sources, query)
        
        print(f"   ✅ Found {len(validated_sources)} validated sources")
        return validated_sources[:self.config.max_sources_per_query]
    
    def _execute_parallel_threaded(self, queries: List[str]) -> Dict[str, List[ResearchSource]]:
        """Execute searches using thread pool for parallelization"""
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(self.search_single_query, query): query
                for query in queries
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    sources = future.result(timeout=self.config.search_timeout)
                    results[query] = sources
                except Exception as e:
                    print(f"   ❌ Search failed for '{query}': {e}")
                    results[query] = []
        
        return results
    
    def _execute_sequential(self, queries: List[str]) -> Dict[str, List[ResearchSource]]:
        """Execute searches sequentially"""
        
        results = {}
        for query in queries:
            try:
                sources = self.search_single_query(query)
                results[query] = sources
            except Exception as e:
                print(f"   ❌ Search failed for '{query}': {e}")
                results[query] = []
        
        return results
    
    def _search_google_custom(self, query: str) -> List[ResearchSource]:
        """Search using Google Custom Search API"""
        
        params = {
            'key': self.google_api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(self.config.max_sources_per_query, 10),  # API limit
            'safe': 'active',
            'fields': 'items(title,link,snippet,pagemap)'
        }
        
        try:
            response = self.session.get(
                self.base_google_url,
                params=params,
                timeout=self.config.search_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_google_results(data)
            
        except Exception as e:
            print(f"   ⚠️ Google Custom Search failed: {e}")
            return []
    
    def _search_serpapi(self, query: str) -> List[ResearchSource]:
        """Search using SerpAPI"""
        
        params = {
            'api_key': self.serpapi_key,
            'engine': 'google',
            'q': query,
            'num': self.config.max_sources_per_query,
            'safe': 'active'
        }
        
        try:
            response = self.session.get(
                self.base_serpapi_url,
                params=params,
                timeout=self.config.search_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_serpapi_results(data)
            
        except Exception as e:
            print(f"   ⚠️ SerpAPI search failed: {e}")
            return []
    
    def _search_fallback(self, query: str) -> List[ResearchSource]:
        """Fallback search method (simulated results for demo)"""
        
        print(f"   📝 Using fallback search method for demo")
        
        # Create simulated results based on query
        query_terms = query.lower().split()
        
        # Generate realistic-looking results
        fallback_results = []
        
        # Academic source simulation
        if any(term in query_terms for term in ['research', 'study', 'analysis', 'method']):
            fallback_results.append(ResearchSource(
                url=f"https://arxiv.org/abs/{datetime.now().year}.{len(query):04d}",
                title=f"Recent Advances in {query.title()}",
                snippet=f"This paper presents a comprehensive study of {query.lower()}, "
                       f"discussing recent developments and future directions...",
                source_type="academic",
                relevance_score=0.9,
                credibility_score=0.95
            ))
        
        # Technical source simulation  
        if any(term in query_terms for term in ['algorithm', 'implementation', 'code', 'software']):
            fallback_results.append(ResearchSource(
                url=f"https://github.com/research/{query.replace(' ', '-')}",
                title=f"{query.title()}: Implementation and Examples",
                snippet=f"Open source implementation of {query.lower()} with examples and documentation. "
                       f"Includes benchmarks and performance analysis...",
                source_type="technical",
                relevance_score=0.8,
                credibility_score=0.7
            ))
        
        # News/blog source simulation
        fallback_results.append(ResearchSource(
            url=f"https://towardsdatascience.com/{query.replace(' ', '-')}-guide",
            title=f"Complete Guide to {query.title()}",
            snippet=f"A comprehensive guide covering the fundamentals of {query.lower()}, "
                   f"including practical applications and real-world examples...",
            source_type="web",
            relevance_score=0.7,
            credibility_score=0.6,
            publication_date=datetime.now().strftime('%Y-%m-%d')
        ))
        
        return fallback_results
    
    def _parse_google_results(self, data: Dict[str, Any]) -> List[ResearchSource]:
        """Parse Google Custom Search API results"""
        
        sources = []
        items = data.get('items', [])
        
        for item in items:
            try:
                # Extract basic information
                title = item.get('title', 'No title')
                url = item.get('link', '')
                snippet = item.get('snippet', 'No description available')
                
                # Extract additional metadata from pagemap if available
                pagemap = item.get('pagemap', {})
                publication_date = None
                authors = []
                
                # Try to extract publication date
                if 'metatags' in pagemap:
                    for metatag in pagemap['metatags']:
                        if 'article:published_time' in metatag:
                            publication_date = metatag['article:published_time'][:10]
                        elif 'datePublished' in metatag:
                            publication_date = metatag['datePublished'][:10]
                
                # Determine source type
                source_type = self._classify_source_type(url)
                
                source = ResearchSource(
                    url=url,
                    title=title,
                    snippet=snippet,
                    source_type=source_type,
                    publication_date=publication_date,
                    authors=authors
                )
                
                sources.append(source)
                
            except Exception as e:
                print(f"   ⚠️ Error parsing search result: {e}")
                continue
        
        return sources
    
    def _parse_serpapi_results(self, data: Dict[str, Any]) -> List[ResearchSource]:
        """Parse SerpAPI results"""
        
        sources = []
        organic_results = data.get('organic_results', [])
        
        for result in organic_results:
            try:
                title = result.get('title', 'No title')
                url = result.get('link', '')
                snippet = result.get('snippet', 'No description available')
                
                source_type = self._classify_source_type(url)
                
                source = ResearchSource(
                    url=url,
                    title=title,
                    snippet=snippet,
                    source_type=source_type,
                    publication_date=result.get('date')
                )
                
                sources.append(source)
                
            except Exception as e:
                print(f"   ⚠️ Error parsing SerpAPI result: {e}")
                continue
        
        return sources
    
    def _classify_source_type(self, url: str) -> str:
        """Classify source type based on URL"""
        
        domain = urlparse(url).netloc.lower()
        
        # Check for academic sources
        for academic_domain in self.trusted_domains['academic']:
            if academic_domain in domain:
                return 'academic'
        
        # Check for institutional sources
        for institutional_suffix in self.trusted_domains['institutional']:
            if domain.endswith(institutional_suffix):
                return 'institutional'
        
        # Check for technical sources
        for tech_domain in self.trusted_domains['technical']:
            if tech_domain in domain:
                return 'technical'
        
        # Default to web
        return 'web'
    
    def _validate_sources(self, sources: List[ResearchSource], query: str) -> List[ResearchSource]:
        """Validate and score sources based on quality criteria"""
        
        validated_sources = []
        query_terms = set(query.lower().split())
        
        for source in sources:
            if self._is_valid_source(source):
                # Calculate relevance score
                source.relevance_score = self._calculate_relevance_score(source, query_terms)
                
                # Calculate credibility score
                source.credibility_score = self._calculate_credibility_score(source)
                
                # Only keep sources above minimum thresholds
                if source.relevance_score >= 0.3 and source.credibility_score >= 0.3:
                    validated_sources.append(source)
        
        # Sort by combined score
        validated_sources.sort(
            key=lambda s: (s.relevance_score + s.credibility_score) / 2,
            reverse=True
        )
        
        return validated_sources
    
    def _is_valid_source(self, source: ResearchSource) -> bool:
        """Check if source meets basic validity criteria"""
        
        # Must have URL and title
        if not source.url or not source.title:
            return False
        
        # URL must be valid
        parsed = urlparse(source.url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Title must be meaningful
        if len(source.title) < 10:
            return False
        
        return True
    
    def _calculate_relevance_score(self, source: ResearchSource, query_terms: set) -> float:
        """Calculate relevance score based on query terms"""
        
        # Combine title and snippet for analysis
        text = f"{source.title} {source.snippet}".lower()
        text_terms = set(text.split())
        
        # Calculate term overlap
        overlap = len(query_terms.intersection(text_terms))
        if len(query_terms) == 0:
            return 0.0
        
        base_score = overlap / len(query_terms)
        
        # Boost for exact phrase matches
        text_combined = " ".join(text_terms)
        query_phrase = " ".join(query_terms)
        if query_phrase in text_combined:
            base_score += 0.2
        
        # Boost for title matches
        title_terms = set(source.title.lower().split())
        title_overlap = len(query_terms.intersection(title_terms))
        if title_overlap > 0:
            base_score += (title_overlap / len(query_terms)) * 0.3
        
        return min(1.0, base_score)
    
    def _calculate_credibility_score(self, source: ResearchSource) -> float:
        """Calculate credibility score based on source characteristics"""
        
        base_score = 0.5  # Neutral baseline
        domain = urlparse(source.url).netloc.lower()
        
        # Academic sources get highest credibility
        if source.source_type == 'academic':
            base_score = 0.9
        elif source.source_type == 'institutional':
            base_score = 0.8
        elif source.source_type == 'technical':
            base_score = 0.7
        
        # Boost for trusted domains
        for domain_list in self.trusted_domains.values():
            for trusted in domain_list:
                if trusted in domain:
                    base_score += 0.1
                    break
        
        # Penalty for suspicious characteristics
        if 'click' in domain or 'ad' in domain or 'spam' in domain:
            base_score -= 0.3
        
        # Boost for recent publication dates
        if source.publication_date:
            try:
                pub_year = int(source.publication_date[:4])
                current_year = datetime.now().year
                if pub_year >= current_year - 2:
                    base_score += 0.1
                elif pub_year >= current_year - 5:
                    base_score += 0.05
            except:
                pass
        
        return max(0.0, min(1.0, base_score))
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_search_statistics(self, results: Dict[str, List[ResearchSource]]) -> Dict[str, Any]:
        """Generate statistics about search results"""
        
        total_sources = sum(len(sources) for sources in results.values())
        
        if total_sources == 0:
            return {"total_sources": 0, "average_relevance": 0, "source_types": {}}
        
        # Calculate statistics
        all_sources = [source for sources in results.values() for source in sources]
        
        avg_relevance = sum(s.relevance_score for s in all_sources) / len(all_sources)
        avg_credibility = sum(s.credibility_score for s in all_sources) / len(all_sources)
        
        # Source type distribution
        source_types = {}
        for source in all_sources:
            source_types[source.source_type] = source_types.get(source.source_type, 0) + 1
        
        return {
            "total_sources": total_sources,
            "unique_queries": len(results),
            "average_relevance": round(avg_relevance, 2),
            "average_credibility": round(avg_credibility, 2),
            "source_types": source_types,
            "top_domains": self._get_top_domains(all_sources)
        }
    
    def _get_top_domains(self, sources: List[ResearchSource]) -> Dict[str, int]:
        """Get top domains from search results"""
        
        domain_counts = {}
        for source in sources:
            domain = urlparse(source.url).netloc
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Return top 5 domains
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_domains[:5])


def demonstrate_web_searcher():
    """Demonstrate web search capabilities"""
    
    print("🌐 WEB SEARCH EXECUTOR DEMONSTRATION")
    print("=" * 35)
    
    print("\\n🔍 Search Capabilities:")
    capabilities = [
        "Google Custom Search API integration",
        "SerpAPI alternative support", 
        "Parallel search execution",
        "Source validation and scoring",
        "Academic source prioritization",
        "Rate limiting and error handling"
    ]
    
    for capability in capabilities:
        print(f"   • {capability}")
    
    print("\\n🏆 Source Quality Scoring:")
    scoring_factors = [
        "Relevance to search query (term overlap, phrase matching)",
        "Source credibility (academic, institutional, technical)",
        "Domain trust level (established academic/institutional domains)",
        "Publication recency (recent sources get boost)",
        "Content quality indicators"
    ]
    
    for factor in scoring_factors:
        print(f"   • {factor}")
    
    print("\\n🎯 Source Types:")
    source_types = [
        ("Academic", "arXiv, IEEE, Nature, Science, PubMed"),
        ("Institutional", ".edu, .gov, .org domains"),
        ("Technical", "GitHub, Stack Overflow, technical blogs"),
        ("Web", "General web sources with quality filtering")
    ]
    
    for s_type, examples in source_types:
        print(f"   • {s_type}: {examples}")


if __name__ == "__main__":
    demonstrate_web_searcher()
    
    print("\\n" + "="*50)
    print("🧪 WEB SEARCHER TESTING")
    print("="*50)
    
    print("\\n💡 To test web search:")
    print("""
from web_searcher import WebSearchExecutor
from state_schemas import ResearchConfiguration

# Create search executor
config = ResearchConfiguration.for_academic_research()
searcher = WebSearchExecutor(config)

# Execute search
queries = [
    "quantum error correction recent advances",
    "machine learning healthcare applications"
]

results = searcher.execute_parallel_search(queries)

# Display results
for query, sources in results.items():
    print(f"\\nQuery: {query}")
    print(f"Sources found: {len(sources)}")
    
    for source in sources[:3]:
        print(f"  • {source.title}")
        print(f"    URL: {source.url}")
        print(f"    Relevance: {source.relevance_score:.2f}")
        print(f"    Credibility: {source.credibility_score:.2f}")

# Get statistics
stats = searcher.get_search_statistics(results)
print(f"\\nSearch Statistics: {stats}")
""")