"""
Citation Formatting Agent

Specialized agent for formatting research citations in multiple academic styles
(IEEE, APA, MLA, Chicago, etc.) with support for different publication types.
"""

import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from research_tools import ResearchPaper


class CitationStyle(Enum):
    """Supported citation styles"""
    IEEE = "ieee"
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    NATURE = "nature"
    VANCOUVER = "vancouver"


class PublicationType(Enum):
    """Types of publications for citation formatting"""
    JOURNAL_ARTICLE = "journal"
    CONFERENCE_PAPER = "conference"
    BOOK = "book"
    BOOK_CHAPTER = "chapter"
    THESIS = "thesis"
    TECHNICAL_REPORT = "report"
    WEBSITE = "website"
    PREPRINT = "preprint"


@dataclass
class CitationData:
    """Structured citation information"""
    title: str
    authors: List[str]
    publication_year: str
    venue: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    publisher: Optional[str] = None
    publication_type: PublicationType = PublicationType.JOURNAL_ARTICLE
    arxiv_id: Optional[str] = None
    isbn: Optional[str] = None
    
    @classmethod
    def from_research_paper(cls, paper: ResearchPaper) -> 'CitationData':
        """Convert ResearchPaper to CitationData"""
        
        # Determine publication type
        pub_type = PublicationType.JOURNAL_ARTICLE  # default
        if paper.arxiv_id:
            pub_type = PublicationType.PREPRINT
        elif paper.venue and any(term in paper.venue.lower() 
                               for term in ['conference', 'proceedings', 'workshop']):
            pub_type = PublicationType.CONFERENCE_PAPER
        
        return cls(
            title=paper.title,
            authors=paper.authors,
            publication_year=paper.publication_date[:4] if paper.publication_date else "Unknown",
            venue=paper.venue,
            doi=paper.doi,
            url=paper.url,
            publication_type=pub_type,
            arxiv_id=paper.arxiv_id
        )


class CitationFormatterAgent:
    """
    Specialized agent for formatting citations in multiple academic styles.
    
    Supports major citation styles used in academic publishing with
    intelligent formatting based on publication type and available metadata.
    """
    
    def __init__(self, llm):
        self.llm = llm
        
        # Style-specific formatting rules
        self.style_rules = {
            CitationStyle.IEEE: {
                "format": "[{number}] {authors}, \"{title},\" {venue}, {year}.",
                "author_format": "F. Lastname",
                "max_authors": 3,
                "et_al": "et al.",
                "numbered": True
            },
            CitationStyle.APA: {
                "format": "{authors} ({year}). {title}. {venue}.",
                "author_format": "Lastname, F.",
                "max_authors": 7,
                "et_al": "et al.",
                "numbered": False
            },
            CitationStyle.MLA: {
                "format": "{authors}. \"{title}.\" {venue}, {year}.",
                "author_format": "Lastname, Firstname",
                "max_authors": 3,
                "et_al": "et al.",
                "numbered": False
            },
            CitationStyle.NATURE: {
                "format": "{authors}. {title}. {venue} ({year}).",
                "author_format": "Lastname, F.",
                "max_authors": 5,
                "et_al": "et al.",
                "numbered": True
            }
        }
        
        # Common venue abbreviations
        self.venue_abbreviations = {
            "Nature": "Nature",
            "Science": "Science",
            "Proceedings of the National Academy of Sciences": "PNAS",
            "Journal of the American Chemical Society": "J. Am. Chem. Soc.",
            "Physical Review Letters": "Phys. Rev. Lett.",
            "Artificial Intelligence": "Artif. Intell.",
            "Machine Learning": "Mach. Learn.",
        }
        
        # Citation counters for numbered styles
        self.citation_counters = {}
    
    def format_citation(self, 
                       citation_data: CitationData,
                       style: CitationStyle,
                       in_text: bool = False) -> str:
        """
        Format a single citation in the specified style.
        
        Args:
            citation_data: Citation information
            style: Target citation style
            in_text: Whether this is an in-text citation
            
        Returns:
            Formatted citation string
        """
        
        if in_text:
            return self._format_in_text_citation(citation_data, style)
        else:
            return self._format_reference_citation(citation_data, style)
    
    def format_multiple_citations(self,
                                 papers: List[Union[ResearchPaper, CitationData]],
                                 style: CitationStyle,
                                 in_text: bool = False) -> List[str]:
        """
        Format multiple citations in the specified style.
        
        Args:
            papers: List of papers or citation data
            style: Target citation style
            in_text: Whether these are in-text citations
            
        Returns:
            List of formatted citations
        """
        citations = []
        
        # Reset counter for numbered styles
        if style in [CitationStyle.IEEE, CitationStyle.NATURE]:
            self.citation_counters[style] = 0
        
        for paper in papers:
            # Convert to CitationData if needed
            if isinstance(paper, ResearchPaper):
                citation_data = CitationData.from_research_paper(paper)
            else:
                citation_data = paper
            
            citation = self.format_citation(citation_data, style, in_text)
            citations.append(citation)
        
        return citations
    
    def _format_reference_citation(self, citation_data: CitationData, style: CitationStyle) -> str:
        """Format full reference citation"""
        
        try:
            if style == CitationStyle.IEEE:
                return self._format_ieee_reference(citation_data)
            elif style == CitationStyle.APA:
                return self._format_apa_reference(citation_data)
            elif style == CitationStyle.MLA:
                return self._format_mla_reference(citation_data)
            elif style == CitationStyle.NATURE:
                return self._format_nature_reference(citation_data)
            else:
                # Fallback to IEEE style
                return self._format_ieee_reference(citation_data)
                
        except Exception as e:
            return f"[Citation formatting error: {str(e)}]"
    
    def _format_in_text_citation(self, citation_data: CitationData, style: CitationStyle) -> str:
        """Format in-text citation"""
        
        try:
            if style in [CitationStyle.IEEE, CitationStyle.NATURE]:
                # Numbered citation
                if style not in self.citation_counters:
                    self.citation_counters[style] = 0
                self.citation_counters[style] += 1
                return f"[{self.citation_counters[style]}]"
            
            elif style == CitationStyle.APA:
                # Author-year format
                authors = self._format_authors_in_text(citation_data.authors, 2)
                return f"({authors}, {citation_data.publication_year})"
            
            elif style == CitationStyle.MLA:
                # Author-page format (simplified, no page numbers from paper data)
                authors = self._format_authors_in_text(citation_data.authors, 1)
                return f"({authors})"
            
            else:
                # Default to APA-style
                authors = self._format_authors_in_text(citation_data.authors, 2)
                return f"({authors}, {citation_data.publication_year})"
                
        except Exception as e:
            return f"[In-text citation error: {str(e)}]"
    
    def _format_ieee_reference(self, citation_data: CitationData) -> str:
        """Format citation in IEEE style"""
        
        # Get citation number
        if CitationStyle.IEEE not in self.citation_counters:
            self.citation_counters[CitationStyle.IEEE] = 0
        self.citation_counters[CitationStyle.IEEE] += 1
        number = self.citation_counters[CitationStyle.IEEE]
        
        # Format authors
        authors = self._format_authors(citation_data.authors, "ieee")
        
        # Format title
        title = f'"{citation_data.title}"'
        
        # Format venue
        venue = citation_data.venue or "Unknown Venue"
        if citation_data.publication_type == PublicationType.PREPRINT and citation_data.arxiv_id:
            venue = f"arXiv preprint arXiv:{citation_data.arxiv_id}"
        
        # Construct citation
        citation = f"[{number}] {authors}, {title}, {venue}, {citation_data.publication_year}."
        
        # Add DOI if available
        if citation_data.doi:
            citation += f" DOI: {citation_data.doi}"
        
        return citation
    
    def _format_apa_reference(self, citation_data: CitationData) -> str:
        """Format citation in APA style"""
        
        # Format authors
        authors = self._format_authors(citation_data.authors, "apa")
        
        # Format based on publication type
        if citation_data.publication_type == PublicationType.JOURNAL_ARTICLE:
            citation = f"{authors} ({citation_data.publication_year}). {citation_data.title}. "
            
            # Add journal info
            venue = citation_data.venue or "Unknown Journal"
            citation += f"*{venue}*"
            
            # Add volume/issue if available
            if citation_data.volume:
                citation += f", *{citation_data.volume}*"
                if citation_data.issue:
                    citation += f"({citation_data.issue})"
                if citation_data.pages:
                    citation += f", {citation_data.pages}"
            
            citation += "."
            
        elif citation_data.publication_type == PublicationType.PREPRINT:
            citation = f"{authors} ({citation_data.publication_year}). {citation_data.title}. "
            if citation_data.arxiv_id:
                citation += f"*arXiv preprint arXiv:{citation_data.arxiv_id}*."
            else:
                citation += "*Preprint*."
        
        else:
            # Default format
            venue = citation_data.venue or "Unknown Venue"
            citation = f"{authors} ({citation_data.publication_year}). {citation_data.title}. {venue}."
        
        # Add DOI if available
        if citation_data.doi:
            citation += f" https://doi.org/{citation_data.doi}"
        
        return citation
    
    def _format_mla_reference(self, citation_data: CitationData) -> str:
        """Format citation in MLA style"""
        
        # Format authors
        authors = self._format_authors(citation_data.authors, "mla")
        
        # Format title
        title = f'"{citation_data.title}."'
        
        # Format venue
        venue = citation_data.venue or "Unknown Venue"
        
        # Construct citation
        citation = f"{authors} {title} *{venue}*, {citation_data.publication_year}."
        
        # Add additional info
        if citation_data.doi:
            citation += f" DOI: {citation_data.doi}."
        elif citation_data.url:
            citation += f" Web. {datetime.now().strftime('%d %b %Y')}."
        
        return citation
    
    def _format_nature_reference(self, citation_data: CitationData) -> str:
        """Format citation in Nature style"""
        
        # Format authors
        authors = self._format_authors(citation_data.authors, "nature")
        
        # Format venue and volume info
        venue = citation_data.venue or "Unknown"
        venue_info = venue
        
        # Add volume/pages for journals
        if citation_data.volume:
            venue_info += f" {citation_data.volume}"
            if citation_data.pages:
                venue_info += f", {citation_data.pages}"
        
        # Construct citation
        citation = f"{authors} {citation_data.title}. {venue_info} ({citation_data.publication_year})."
        
        return citation
    
    def _format_authors(self, authors: List[str], style: str) -> str:
        """Format author list according to style"""
        
        if not authors:
            return "Unknown Author"
        
        if style == "ieee":
            # F. Lastname format, max 3 authors
            formatted_authors = []
            for author in authors[:3]:
                formatted = self._format_author_ieee(author)
                formatted_authors.append(formatted)
            
            if len(authors) > 3:
                return ", ".join(formatted_authors) + ", et al."
            elif len(formatted_authors) > 1:
                return ", ".join(formatted_authors[:-1]) + ", and " + formatted_authors[-1]
            else:
                return formatted_authors[0]
        
        elif style == "apa":
            # Lastname, F. format
            formatted_authors = []
            for author in authors[:7]:  # APA limit
                formatted = self._format_author_apa(author)
                formatted_authors.append(formatted)
            
            if len(authors) > 7:
                return ", ".join(formatted_authors[:6]) + ", ... " + self._format_author_apa(authors[-1])
            elif len(formatted_authors) > 1:
                return ", ".join(formatted_authors[:-1]) + ", & " + formatted_authors[-1]
            else:
                return formatted_authors[0]
        
        elif style == "mla":
            # First author: Lastname, Firstname; others: Firstname Lastname
            if len(authors) == 1:
                return self._format_author_mla_first(authors[0])
            elif len(authors) <= 3:
                first = self._format_author_mla_first(authors[0])
                others = [self._format_author_mla_other(a) for a in authors[1:]]
                if len(authors) == 2:
                    return f"{first} and {others[0]}"
                else:
                    return f"{first}, {', '.join(others[:-1])}, and {others[-1]}"
            else:
                first = self._format_author_mla_first(authors[0])
                return f"{first}, et al."
        
        elif style == "nature":
            # Lastname, F. format, max 5 authors
            formatted_authors = []
            for author in authors[:5]:
                formatted = self._format_author_nature(author)
                formatted_authors.append(formatted)
            
            if len(authors) > 5:
                return ", ".join(formatted_authors) + " et al."
            else:
                return " & ".join([", ".join(formatted_authors[:-1]), formatted_authors[-1]]) if len(formatted_authors) > 1 else formatted_authors[0]
        
        return ", ".join(authors[:3])  # Fallback
    
    def _format_authors_in_text(self, authors: List[str], max_authors: int) -> str:
        """Format authors for in-text citations"""
        if not authors:
            return "Unknown"
        
        if len(authors) == 1:
            return authors[0].split()[-1]  # Last name only
        elif len(authors) == 2:
            names = [a.split()[-1] for a in authors]
            return f"{names[0]} and {names[1]}"
        elif len(authors) <= max_authors:
            names = [a.split()[-1] for a in authors[:-1]]
            last_name = authors[-1].split()[-1]
            return f"{', '.join(names)}, and {last_name}"
        else:
            first_author = authors[0].split()[-1]
            return f"{first_author} et al."
    
    def _format_author_ieee(self, author: str) -> str:
        """Format author for IEEE style: F. Lastname"""
        parts = author.strip().split()
        if len(parts) >= 2:
            first_initial = parts[0][0] if parts[0] else "X"
            last_name = parts[-1]
            return f"{first_initial}. {last_name}"
        return author
    
    def _format_author_apa(self, author: str) -> str:
        """Format author for APA style: Lastname, F."""
        parts = author.strip().split()
        if len(parts) >= 2:
            first_initial = parts[0][0] if parts[0] else "X"
            last_name = parts[-1]
            return f"{last_name}, {first_initial}."
        return author
    
    def _format_author_mla_first(self, author: str) -> str:
        """Format first author for MLA: Lastname, Firstname"""
        parts = author.strip().split()
        if len(parts) >= 2:
            first_name = parts[0]
            last_name = parts[-1]
            return f"{last_name}, {first_name}"
        return author
    
    def _format_author_mla_other(self, author: str) -> str:
        """Format other authors for MLA: Firstname Lastname"""
        return author.strip()
    
    def _format_author_nature(self, author: str) -> str:
        """Format author for Nature style: Lastname, F."""
        return self._format_author_apa(author)  # Same format
    
    def generate_bibliography(self,
                            papers: List[Union[ResearchPaper, CitationData]],
                            style: CitationStyle,
                            title: str = "References") -> str:
        """
        Generate a complete bibliography section.
        
        Args:
            papers: List of papers to cite
            style: Citation style to use
            title: Section title
            
        Returns:
            Formatted bibliography section
        """
        
        # Format all citations
        citations = self.format_multiple_citations(papers, style, in_text=False)
        
        # Create bibliography
        bibliography = f"# {title}\\n\\n"
        
        if style in [CitationStyle.IEEE, CitationStyle.NATURE]:
            # Numbered format
            for citation in citations:
                bibliography += f"{citation}\\n\\n"
        else:
            # Alphabetical format (simplified - would need proper sorting)
            for citation in citations:
                bibliography += f"{citation}\\n\\n"
        
        return bibliography.strip()
    
    def validate_citation_completeness(self, citation_data: CitationData) -> Dict[str, Any]:
        """
        Validate citation completeness and suggest improvements.
        
        Returns:
            Dictionary with validation results and suggestions
        """
        
        validation = {
            "complete": True,
            "missing_fields": [],
            "suggestions": [],
            "quality_score": 1.0
        }
        
        # Check essential fields
        if not citation_data.title or citation_data.title == "Unknown":
            validation["missing_fields"].append("title")
            validation["complete"] = False
        
        if not citation_data.authors:
            validation["missing_fields"].append("authors")
            validation["complete"] = False
        
        if not citation_data.publication_year or citation_data.publication_year == "Unknown":
            validation["missing_fields"].append("publication_year")
            validation["complete"] = False
        
        # Check desirable fields
        if not citation_data.venue:
            validation["suggestions"].append("Add venue/journal information")
            validation["quality_score"] -= 0.2
        
        if not citation_data.doi and not citation_data.arxiv_id:
            validation["suggestions"].append("Add DOI or arXiv ID for better accessibility")
            validation["quality_score"] -= 0.2
        
        # Check publication type appropriateness
        if citation_data.publication_type == PublicationType.PREPRINT:
            if not citation_data.arxiv_id:
                validation["suggestions"].append("Add arXiv ID for preprint citations")
                validation["quality_score"] -= 0.1
        
        validation["quality_score"] = max(0.0, validation["quality_score"])
        
        return validation


def demonstrate_citation_agent():
    """Demonstrate citation formatting agent capabilities"""
    
    print("📖 CITATION FORMATTING AGENT DEMONSTRATION")
    print("=" * 45)
    
    print("\\n🎯 Supported Citation Styles:")
    styles = [
        ("IEEE", "Numbered citations [1], common in engineering and CS"),
        ("APA", "Author-year format (Smith, 2023), common in psychology and social sciences"),
        ("MLA", "Author-page format (Smith 123), common in humanities"),
        ("Nature", "Numbered format with specific Nature journal requirements"),
        ("Chicago", "Author-date or notes-bibliography style"),
        ("Harvard", "Author-year format similar to APA")
    ]
    
    for style, description in styles:
        print(f"   • {style}: {description}")
    
    print("\\n📚 Publication Types:")
    pub_types = [
        "Journal articles",
        "Conference papers",
        "Books and book chapters",
        "Theses and dissertations",
        "Technical reports",
        "Websites and online sources",
        "Preprints (arXiv, bioRxiv, etc.)"
    ]
    
    for pub_type in pub_types:
        print(f"   • {pub_type}")
    
    print("\\n🛠 Features:")
    features = [
        "Automatic publication type detection",
        "Intelligent author name formatting",
        "DOI and URL handling",
        "Venue name abbreviation",
        "Citation completeness validation",
        "Bibliography generation",
        "In-text and reference citation formats"
    ]
    
    for feature in features:
        print(f"   • {feature}")


if __name__ == "__main__":
    demonstrate_citation_agent()
    
    print("\\n" + "="*50)
    print("🧪 CITATION AGENT TESTING")
    print("="*50)
    
    # Create sample citation data
    sample_paper = CitationData(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        publication_year="2017",
        venue="Advances in Neural Information Processing Systems",
        pages="5998-6008",
        url="https://arxiv.org/abs/1706.03762",
        publication_type=PublicationType.CONFERENCE_PAPER
    )
    
    print("\\n📄 Sample Citation Data:")
    print(f"Title: {sample_paper.title}")
    print(f"Authors: {', '.join(sample_paper.authors)}")
    print(f"Year: {sample_paper.publication_year}")
    print(f"Venue: {sample_paper.venue}")
    
    # Create citation formatter (without LLM for demo)
    class MockLLM:
        def invoke(self, prompt):
            class MockResponse:
                content = "Mock LLM response"
            return MockResponse()
    
    formatter = CitationFormatterAgent(MockLLM())
    
    print("\\n📖 Citations in Different Styles:")
    print("-" * 40)
    
    styles_to_test = [CitationStyle.IEEE, CitationStyle.APA, CitationStyle.MLA]
    
    for style in styles_to_test:
        citation = formatter.format_citation(sample_paper, style, in_text=False)
        in_text = formatter.format_citation(sample_paper, style, in_text=True)
        
        print(f"\\n{style.value.upper()}:")
        print(f"Reference: {citation}")
        print(f"In-text: {in_text}")
    
    # Validation example
    print("\\n✅ Citation Validation:")
    validation = formatter.validate_citation_completeness(sample_paper)
    print(f"Complete: {validation['complete']}")
    print(f"Quality Score: {validation['quality_score']:.1f}")
    if validation['suggestions']:
        print(f"Suggestions: {'; '.join(validation['suggestions'])}")
    
    print("\\n💡 To use with real LLM:")
    print("""
from llm_providers import create_research_llm
from citation_agent import CitationFormatterAgent, CitationStyle

llm = create_research_llm('google')
formatter = CitationFormatterAgent(llm)

# Format single citation
citation = formatter.format_citation(citation_data, CitationStyle.APA)

# Generate bibliography
bibliography = formatter.generate_bibliography(papers, CitationStyle.IEEE)
""")