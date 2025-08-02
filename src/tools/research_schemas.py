from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


if __name__ == "__main__":
    print("--- Testing research_schemas.py ---")

    sq = SearchQueryList(query=["bearing fault"], rationale="demo")
    assert sq.query == ["bearing fault"]

    reflection = Reflection(
        is_sufficient=False,
        knowledge_gap="need more data",
        follow_up_queries=["collect vibration data"],
    )
    assert not reflection.is_sufficient
    assert reflection.follow_up_queries[0] == "collect vibration data"

    print("\n--- research_schemas.py tests passed! ---")
