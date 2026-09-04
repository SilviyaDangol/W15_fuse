"""LiteLLM-backed gateway for hosted OpenAI and OpenAI-compatible vLLM."""

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from litellm import acompletion
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.chat import GeneratedAnswer, ToolUse
from app.services.vector_store import RetrievedChunk, VectorStore

SYSTEM_PROMPT = """You are a precise document research assistant.
Answer only from supplied context when context is available; otherwise say that
the document collection has not yet provided supporting material. Use the
search_documents tool before answering any document-research question. Your final
response must be a JSON object with exactly these keys: answer, sources, and
tool_uses. sources and tool_uses must always be JSON lists. Do not add Markdown
or any text outside the JSON object. Do not invent citations; the application
adds verified sources after retrieval."""

SEARCH_DOCUMENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the indexed document collection for passages relevant to the user's question.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused document-search query."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Maximum passages to return."},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
    },
}


class LLMGateway(Protocol):
    async def answer(self, query: str, provider: str | None = None) -> GeneratedAnswer: ...


class UnifiedLLMGateway:
    """Routes identical requests through LiteLLM's provider-neutral interface."""

    def __init__(
        self,
        settings: Settings | None = None,
        completion_fn: Callable[..., Awaitable[Any]] | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.completion_fn = completion_fn or acompletion
        self.vector_store = vector_store

    async def answer(self, query: str, provider: str | None = None) -> GeneratedAnswer:
        selected_provider = provider or self.settings.default_model_provider
        messages: list[dict] = [
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        tool_uses: list[ToolUse] = []
        retrieved: list[RetrievedChunk] = []

        # vLLM is served with a compact model: retrieve deterministically and
        # provide context rather than relying on tool-choice behavior.
        if selected_provider == "vllm":
            retrieved = self._search(query, self.settings.retrieval_top_k)
            messages[0]["content"] += _context_message(retrieved)

        # At most one tool round prevents accidental infinite tool loops.
        request = {
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "top_p": self.settings.llm_top_p,
            "response_format": _response_format(selected_provider),
        }
        # The hosted route demonstrates strict schemas and function calling.
        # The compact Colab model is served for local-inference testing and is
        # more dependable when asked for JSON mode without tools in one turn.
        if selected_provider == "openai":
            request.update(
                {
                    "tools": [SEARCH_DOCUMENTS_TOOL],
                    # Every OpenAI research request performs retrieval. This
                    # makes grounding deterministic instead of hoping the model
                    # elects to call the tool.
                    "tool_choice": {"type": "function", "function": {"name": "search_documents"}},
                }
            )
        first = await self._complete(selected_provider, **request)
        message = first.choices[0].message
        final = first
        if message.tool_calls:
            messages.append(_assistant_tool_message(message))
            for call in message.tool_calls:
                if call.function.name != "search_documents":
                    raise ValueError("Model requested an unavailable tool.")
                try:
                    arguments = json.loads(call.function.arguments)
                    search_query = arguments["query"]
                    limit = arguments["limit"]
                    if not isinstance(search_query, str) or not isinstance(limit, int):
                        raise ValueError("Search query and limit are required.")
                    retrieved = self._search(search_query, limit)
                    result = _tool_summary(retrieved)
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
                    result = f"Tool error: {error}"
                    arguments = {"query": "invalid", "limit": 1}
                tool_uses.append(ToolUse(name="search_documents", arguments=arguments, result=result))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _tool_result(retrieved)})

            final = await self._complete(
                selected_provider,
                messages=messages,
                temperature=self.settings.llm_temperature,
                top_p=self.settings.llm_top_p,
                response_format=_response_format(selected_provider),
            )
        content = final.choices[0].message.content
        if not content:
            raise ValueError("Model returned no structured response.")
        answer = _parse_model_answer(content, selected_provider)
        # Tool history is application-owned, so a model cannot forge it in JSON.
        return answer.model_copy(update={"sources": [item.as_source() for item in retrieved], "tool_uses": tool_uses})

    def _search(self, query: str, limit: int) -> list[RetrievedChunk]:
        return (self.vector_store or VectorStore(self.settings)).search(query, limit)

    async def _complete(self, provider: str, **request: Any) -> Any:
        return await self.completion_fn(**self._provider_options(provider), **request)

    def _provider_options(self, provider: str) -> dict[str, str]:
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for the OpenAI route.")
            return {"model": f"openai/{self.settings.openai_model}", "api_key": self.settings.openai_api_key}
        if provider == "vllm":
            return {
                "model": f"openai/{self.settings.vllm_served_model_name}",
                "api_base": self.settings.vllm_base_url,
                "api_key": self.settings.vllm_api_key or "local-vllm",
            }
        raise ValueError("Provider must be either 'openai' or 'vllm'.")


def _response_format(provider: str) -> dict:
    if provider == "vllm":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "assistant_answer",
            "strict": True,
            "schema": GeneratedAnswer.model_json_schema(),
        },
    }


def _parse_model_answer(content: str, provider: str) -> GeneratedAnswer:
    """Validate strict OpenAI output; normalize unreliable compact vLLM metadata."""
    try:
        if provider == "vllm":
            raw = json.loads(content)
            if not isinstance(raw, dict) or not isinstance(raw.get("answer"), str):
                raise ValueError("vLLM JSON did not contain a string answer.")
            # The application owns citations and tool history. Small local models
            # often emit wrong shapes here even in JSON mode, so never trust them.
            return GeneratedAnswer(answer=raw["answer"], sources=[], tool_uses=[])
        return GeneratedAnswer.model_validate_json(content)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as error:
        raise ValueError("Model returned an invalid structured response.") from error


def _assistant_tool_message(message: Any) -> dict[str, Any]:
    """Convert LiteLLM's OpenAI-shaped tool call message into a next-turn input."""
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls or []
        ],
    }


def _tool_result(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No indexed document passages matched the search."
    return "\n\n".join(
        f"[source: {item.filename}; chunk: {item.chunk_id}]\n{item.text}" for item in chunks
    )


def _tool_summary(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No indexed document passages matched the search."
    identifiers = ", ".join(item.chunk_id for item in chunks)
    return f"Retrieved {len(chunks)} passages: {identifiers}"


def _context_message(chunks: list[RetrievedChunk]) -> str:
    return "\n\nRetrieved document context:\n" + _tool_result(chunks)
