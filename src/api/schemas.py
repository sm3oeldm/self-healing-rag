"""
Pydantic schemas for the self-healing RAG API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request body for a RAG query."""
    question: str = Field(..., min_length=1, description="The user's question")


class QueryResponse(BaseModel):
    """Response body after a RAG pipeline run completes."""
    answer: str = Field(..., description="The final answer returned to the user")
    verdict: str = Field(..., description="PASS or FAIL from the critic")
    retry_count: int = Field(..., ge=0, description="Number of retries used")
    reason: Optional[str] = Field(None, description="Critic's explanation")
    original_question: str = Field(..., description="The original user question")


class StepEvent(BaseModel):
    """A pipeline step event for SSE streaming."""
    type: str = Field(..., description="Event type: step / token / result / retry / error")
    data: dict = Field(default_factory=dict, description="Event payload")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health: ok or error")
    vectorstore_ready: bool = Field(..., description="Whether ChromaDB is accessible")
    llm_configured: bool = Field(..., description="Whether an LLM API key is configured")
    version: str = Field("1.0.0", description="API version")
