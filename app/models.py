from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union

class InputData(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    conversation: Dict[str, Any] = Field(..., description="Conversation JSON object")
    context_vectors: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(..., description="Context vectors - can be array or wrapped response object")
    model_type: Optional[str] = Field(default="openai", description="LLM provider: 'openai' or 'gemini'")
    model_name: Optional[str] = Field(default=None, description="Specific model name (e.g., 'gpt-4o-mini', 'gemini-1.5-flash')")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")

class ContextWithScore(BaseModel):
    """Context chunk with similarity score."""
    text: str = Field(..., description="Context text")
    source_url: Optional[str] = Field(None, description="Source URL")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity to user query (0-1)")

class EvaluationReport(BaseModel):
    generated_response: str
    relevance_score: int = Field(..., ge=1, le=10)
    completeness_score: int = Field(..., ge=1, le=10)
    accuracy_score: int = Field(..., ge=1, le=10)
    hallucinations: List[str] = Field(default_factory=list)
    latency_ms: float
    cost_usd: float
    retrieved_context: List[ContextWithScore] = Field(default_factory=list, description="Context chunks with similarity scores")
    prompt_used: str = Field(..., description="Final prompt used to generate the response")
    explanations: Dict[str, str] = Field(default_factory=dict)