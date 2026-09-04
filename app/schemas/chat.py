from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What does the literature review recommend for resume parsing?",
                    "provider": "openai",
                },
                {
                    "query": "What are the main TA-MAS research themes?",
                    "provider": "vllm",
                },
                {
                    "query": "What documents have been indexed?"
                },
            ]
        }
    )

    query: str = Field(min_length=1, max_length=4_000, description="The user's question.")
    provider: Literal["openai", "vllm"] | None = Field(
        default=None,
        description="Model route: 'openai' for the hosted OpenAI model, 'vllm' for your local/Colab vLLM server. Omit to use DEFAULT_MODEL_PROVIDER.",
        examples=["openai", "vllm"],
    )
    allow_fallback: bool = Field(
        default=False,
        description="Only for vLLM: retry vLLM first, then use OpenAI if vLLM remains unavailable.",
    )


class Source(BaseModel):
    """A retrieved document citation. Populated in the RAG milestone."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    chunk_id: str
    excerpt: str


class SearchDocumentsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    limit: int = Field(ge=1, le=5)


class ToolUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: SearchDocumentsArguments
    result: str


class GeneratedAnswer(BaseModel):
    """The model-generated portion of a response, before delivery metadata."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[Source]
    tool_uses: list[ToolUse]


class ResponseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_provider: Literal["openai", "vllm"] = "openai"
    actual_provider: Literal["openai", "vllm"] = "openai"
    cache_hit: bool = False
    fallback_used: bool = False
    latency_ms: float = Field(default=0.0, ge=0)


class AssistantAnswer(GeneratedAnswer):
    """The stable public JSON contract returned by every chat endpoint."""

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class BatchChatRequest(BaseModel):
    requests: list[ChatRequest] = Field(min_length=1, max_length=10)


class BatchChatItem(BaseModel):
    index: int
    answer: AssistantAnswer | None = None
    error: str | None = None


class BatchChatResponse(BaseModel):
    results: list[BatchChatItem]
