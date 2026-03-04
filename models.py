"""
Data models and type definitions for the RAG Multi-Agent system.
"""
from typing import TypedDict
from pydantic import BaseModel, Field


class GraderOutput(BaseModel):
    """Output model for PDF relevance grading"""
    is_relevant: bool = Field(
        description="If the document contains technical information directly related to the question, mark True; otherwise, mark False."
    )
    is_topic_related: bool = Field(
        description="Is the question related to the general topic of the knowledge base? (True for technical questions, False for off-topic like weather, jokes, etc.)",
        default=True
    )


class CriticOutput(BaseModel):
    """Output model for report evaluation"""
    is_satisfactory: bool = Field(
        description="If the report is sufficient, well-structured, and complete, mark True; otherwise, mark False."
    )
    feedback: str = Field(
        description="If the report is incomplete or rejected, explain why. If it's acceptable, leave empty."
    )
    quality_score: float = Field(
        description="Report quality score from 0.0 to 1.0, where 1.0 is perfect.",
        default=0.7
    )


class AgentState(TypedDict):
    """State passed between workflow nodes"""
    question: str
    rag_context: str
    web_context: str
    report: str
    feedback: str
    iterations: int
    is_satisfactory: bool
    is_relevant: bool
    is_topic_related: bool
    found_in_db: bool
