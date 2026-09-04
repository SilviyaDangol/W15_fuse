from fastapi import HTTPException, status

from app.services.llm_gateway import LLMGateway, UnifiedLLMGateway


def get_llm_gateway() -> LLMGateway:
    try:
        return UnifiedLLMGateway()
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
