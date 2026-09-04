from fastapi import Depends, HTTPException, status

from app.services.llm_gateway import LLMGateway, UnifiedLLMGateway
from app.services.reliability import ChatService, RedisStore


def get_llm_gateway() -> LLMGateway:
    try:
        return UnifiedLLMGateway()
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error


def get_redis_store() -> RedisStore:
    # Kept as a dependency so tests can replace Redis with an in-memory store.
    return RedisStore()


def get_chat_service(
    gateway: LLMGateway = Depends(get_llm_gateway),
    store: RedisStore = Depends(get_redis_store),
) -> ChatService:
    # Routes remain thin: this service owns cache, retry, and fallback policy.
    return ChatService(gateway, store)
