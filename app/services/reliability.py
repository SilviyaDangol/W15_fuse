"""Redis-backed delivery controls and resilient chat orchestration."""

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass

import redis.asyncio as redis_async
from redis import Redis

from app.core.config import Settings, get_settings
from app.schemas.chat import AssistantAnswer, ChatRequest, GeneratedAnswer, ResponseMetadata
from app.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


class ProviderUnavailableError(Exception):
    """A provider could not produce a usable response after its retry budget."""


@dataclass(frozen=True)
class LimitResult:
    allowed: bool
    retry_after: int = 60


class RedisStore:
    """Best-effort Redis cache and fixed-window limiter; failures degrade safely."""

    collection_version_key = "rag:collection_version"

    def __init__(self, settings: Settings | None = None, client=None) -> None:
        self.settings = settings or get_settings()
        self.client = client or redis_async.from_url(self.settings.redis_url, decode_responses=True)

    async def limit(self, bucket: str, client_ip: str, maximum: int, cost: int = 1) -> LimitResult:
        # A minute bucket is simple, shared across API replicas, and needs no cleanup job.
        window = int(time.time() // 60)
        key = f"limit:{bucket}:{client_ip}:{window}"
        try:
            async with self.client.pipeline(transaction=True) as pipeline:
                pipeline.incrby(key, cost)
                pipeline.expire(key, 61, nx=True)
                result, _ = await pipeline.execute()
            return LimitResult(allowed=int(result) <= maximum, retry_after=60 - int(time.time() % 60))
        except Exception as error:  # Availability is reported by /health/ready; do not block all chat during a Redis blip.
            logger.warning("redis_rate_limit_unavailable error=%s", type(error).__name__)
            return LimitResult(allowed=True)

    async def collection_version(self) -> int:
        try:
            return int(await self.client.get(self.collection_version_key) or 0)
        except Exception as error:
            logger.warning("redis_collection_version_unavailable error=%s", type(error).__name__)
            return 0

    async def cached_answer(self, key: str) -> AssistantAnswer | None:
        try:
            raw = await self.client.get(key)
            if not raw:
                return None
            answer = AssistantAnswer.model_validate_json(raw)
            # A cache hit avoids provider work, so report zero model latency explicitly.
            return answer.model_copy(update={"metadata": answer.metadata.model_copy(update={"cache_hit": True, "latency_ms": 0.0})})
        except Exception as error:
            logger.warning("redis_cache_read_unavailable error=%s", type(error).__name__)
            return None

    async def cache_answer(self, key: str, answer: AssistantAnswer) -> None:
        try:
            await self.client.set(key, answer.model_dump_json(), ex=self.settings.chat_cache_ttl_seconds)
        except Exception as error:
            logger.warning("redis_cache_write_unavailable error=%s", type(error).__name__)

    async def ready(self) -> bool:
        try:
            return bool(await self.client.ping())
        except Exception:
            return False


def bump_collection_version_sync(settings: Settings | None = None) -> int:
    """Called only after a Celery worker has successfully indexed a document."""
    configured = settings or get_settings()
    try:
        client = Redis.from_url(configured.redis_url, decode_responses=True)
        return int(client.incr(RedisStore.collection_version_key))
    except Exception as error:
        logger.warning("redis_collection_version_bump_failed error=%s", type(error).__name__)
        return 0


class ChatService:
    def __init__(self, gateway: LLMGateway, store: RedisStore, settings: Settings | None = None) -> None:
        self.gateway = gateway
        self.store = store
        self.settings = settings or get_settings()

    async def answer(self, payload: ChatRequest) -> AssistantAnswer:
        requested = payload.provider or self.settings.default_model_provider
        if requested not in {"openai", "vllm"}:
            raise ProviderUnavailableError("Configured default provider is invalid.")
        version = await self.store.collection_version()
        # Versioning avoids expensive cache scans after a new document changes RAG context.
        key = self._cache_key(payload, requested, version)
        cached = await self.store.cached_answer(key)
        if cached:
            return cached
        started = time.perf_counter()
        fallback_used = False
        try:
            generated = await self._attempt_provider(payload.query, requested)
            actual = requested
        except ProviderUnavailableError:
            # Fallback is deliberately one-way; local vLLM may be temporary, OpenAI is never silently replaced.
            if requested != "vllm" or not payload.allow_fallback:
                raise
            fallback_used = True
            generated = await self._attempt_provider(payload.query, "openai")
            actual = "openai"
        answer = AssistantAnswer(
            **generated.model_dump(exclude={"metadata"}),
            metadata=ResponseMetadata(
                requested_provider=requested,
                actual_provider=actual,
                fallback_used=fallback_used,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            ),
        )
        await self.store.cache_answer(key, answer)
        return answer

    async def _attempt_provider(self, query: str, provider: str) -> GeneratedAnswer:
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_retry_attempts):
            try:
                return await self.gateway.answer(query, provider)
            except ValueError as error:
                # Invalid model JSON is commonly transient with small served models.
                last_error = error
            except Exception as error:
                last_error = error
            if attempt + 1 < self.settings.llm_retry_attempts:
                # Small jitter prevents simultaneous callers from retrying in lockstep.
                delay = self.settings.llm_retry_base_delay_seconds * (2**attempt) + random.uniform(0, 0.1)
                logger.warning("provider_retry provider=%s attempt=%s error=%s", provider, attempt + 1, type(last_error).__name__)
                await asyncio.sleep(delay)
        logger.warning("provider_unavailable provider=%s attempts=%s error=%s", provider, self.settings.llm_retry_attempts, type(last_error).__name__ if last_error else "unknown")
        raise ProviderUnavailableError(f"The {provider} provider is currently unavailable.") from last_error

    @staticmethod
    def _cache_key(payload: ChatRequest, requested: str, version: int) -> str:
        # Provider and fallback policy affect output, so they must be part of the cache identity.
        normalized = " ".join(payload.query.casefold().split())
        raw = json.dumps({"query": normalized, "provider": requested, "allow_fallback": payload.allow_fallback, "collection_version": version}, sort_keys=True)
        return "chat:" + hashlib.sha256(raw.encode()).hexdigest()
