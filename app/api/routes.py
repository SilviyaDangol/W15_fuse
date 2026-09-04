from pathlib import Path
from typing import Annotated
from uuid import uuid4
import logging
import asyncio

from celery.result import AsyncResult
from fastapi import APIRouter, Body, Depends, File, HTTPException, Request, UploadFile, status

from app.api.dependencies import get_chat_service, get_redis_store
from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.schemas.chat import AssistantAnswer, BatchChatItem, BatchChatRequest, BatchChatResponse, ChatRequest
from app.schemas.documents import DocumentIngestionAccepted
from app.schemas.jobs import DemoBatchRequest, JobCreated, JobStatus
from app.services.reliability import ChatService, ProviderUnavailableError, RedisStore
from app.services.vector_store import VectorStore
from app.tasks.ingestion import ingest_text_document, process_demo_batch

router = APIRouter()
logger = logging.getLogger(__name__)
SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="Check Redis and ChromaDB readiness")
async def readiness_check(store: RedisStore = Depends(get_redis_store)) -> dict[str, str]:
    # Do not declare the API ready until both delivery state and retrieval state are usable.
    redis_ready = await store.ready()
    try:
        VectorStore().collection.count()
        chroma_ready = True
    except Exception:
        chroma_ready = False
    if not redis_ready or not chroma_ready:
        raise HTTPException(status_code=503, detail="Required backing services are not ready.")
    return {"status": "ready", "redis": "ok", "chromadb": "ok"}


@router.get("/providers", summary="List available chat-provider values")
async def list_providers() -> dict:
    """Shows exactly which value to send in the chat request's provider field."""
    return {
        "default_provider": get_settings().default_model_provider,
        "providers": [
            {
                "value": "openai",
                "label": "OpenAI",
                "description": "Hosted model with structured JSON and document-search function calling.",
            },
            {
                "value": "vllm",
                "label": "vLLM",
                "description": "Open-source model served locally or through your configured Colab URL.",
            },
        ],
    }


@router.post(
    "/chat",
    response_model=AssistantAnswer,
    summary="Ask a grounded question",
    description="Use `openai` or `vllm` for `provider`, or omit it to use the configured default.",
)
async def chat(
    payload: Annotated[
        ChatRequest,
        Body(
            openapi_examples={
                "openai_recommended": {
                    "summary": "OpenAI — recommended for the final demo",
                    "description": "Uses strict JSON plus the search_documents function tool.",
                    "value": {
                        "query": "What does the literature review recommend for resume parsing?",
                        "provider": "openai",
                    },
                },
                "vllm_colab": {
                    "summary": "vLLM — local or Colab model",
                    "description": "Requires your vLLM server/ngrok tunnel to be running.",
                    "value": {
                        "query": "What are the main TA-MAS research themes?",
                        "provider": "vllm",
                    },
                },
                "default_provider": {
                    "summary": "Use the configured default",
                    "description": "No provider field: uses DEFAULT_MODEL_PROVIDER from .env.",
                    "value": {"query": "What document is currently indexed?"},
                },
            }
        ),
    ],
    request: Request,
    service: ChatService = Depends(get_chat_service),
    store: RedisStore = Depends(get_redis_store),
) -> AssistantAnswer:
    """Generate a schema-validated answer through the configured model provider."""
    await _enforce_limit(store, "chat", request, get_settings().chat_rate_limit_per_minute)
    try:
        return await service.answer(payload)
    except ProviderUnavailableError as error:
        logger.warning("chat_unavailable provider=%s error=%s", payload.provider, type(error).__name__)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Selected provider is temporarily unavailable.") from error
    except Exception as error:
        logger.exception("chat_gateway_failure provider=%s error=%s", payload.provider, type(error).__name__)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="The model gateway returned an invalid response.") from error


@router.post("/chat/batch", response_model=BatchChatResponse, summary="Process up to 10 chat requests concurrently")
async def chat_batch(
    payload: BatchChatRequest,
    request: Request,
    service: ChatService = Depends(get_chat_service),
    store: RedisStore = Depends(get_redis_store),
) -> BatchChatResponse:
    await _enforce_limit(store, "chat", request, get_settings().chat_rate_limit_per_minute, cost=len(payload.requests))
    # asyncio keeps the endpoint responsive while this semaphore bounds upstream pressure.
    semaphore = asyncio.Semaphore(get_settings().chat_batch_concurrency)

    async def run_one(index: int, item: ChatRequest) -> BatchChatItem:
        async with semaphore:
            try:
                return BatchChatItem(index=index, answer=await service.answer(item))
            except ProviderUnavailableError:
                return BatchChatItem(index=index, error="Selected provider is temporarily unavailable.")
            except Exception:
                logger.exception("batch_chat_failure index=%s", index)
                return BatchChatItem(index=index, error="The model gateway returned an invalid response.")

    return BatchChatResponse(results=await asyncio.gather(*(run_one(index, item) for index, item in enumerate(payload.requests))))


@router.post("/documents/ingest", response_model=DocumentIngestionAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="Upload a UTF-8 .txt/.md, .docx, or text-based .pdf document (maximum 5 MB)."),
    ],
    request: Request,
    store: RedisStore = Depends(get_redis_store),
) -> DocumentIngestionAccepted:
    """Store a document and queue its extraction, embedding, and indexing work."""
    await _enforce_limit(store, "upload", request, get_settings().upload_rate_limit_per_minute)
    extension = Path(file.filename).suffix.lower() if file.filename else ""
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        accepted = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{extension or 'none'}'. Upload one of: {accepted}.")
    content = await file.read()
    if not content or len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Document must be between 1 byte and 5 MB.")
    if extension in {".txt", ".md"}:
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HTTPException(status_code=400, detail="Text documents must be UTF-8 encoded.") from error
    document_id = str(uuid4())
    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{document_id}{extension}"
    # Persist before queuing so the separate Celery process has a stable input file.
    stored_path.write_bytes(content)
    task = ingest_text_document.delay(document_id, file.filename, str(stored_path.resolve()))
    logger.info("document_queued document_id=%s filename=%s bytes=%s job_id=%s", document_id, file.filename, len(content), task.id)
    return DocumentIngestionAccepted(document_id=document_id, filename=file.filename, job_id=task.id)


async def _enforce_limit(store: RedisStore, bucket: str, request: Request, maximum: int, cost: int = 1) -> None:
    # TestClient and Docker both supply client.host; a reverse proxy can be added later if deployed publicly.
    client_ip = request.client.host if request.client else "unknown"
    result = await store.limit(bucket, client_ip, maximum, cost)
    if not result.allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.", headers={"Retry-After": str(result.retry_after)})


@router.post("/jobs/demo-batch", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
async def submit_demo_batch(payload: DemoBatchRequest) -> JobCreated:
    """Queue work and return immediately instead of holding the client connection."""
    task = process_demo_batch.delay(payload.documents)
    return JobCreated(job_id=task.id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    task = AsyncResult(job_id, app=celery_app)
    if task.state == "PENDING" and task.result is None:
        # Celery cannot distinguish an unknown ID from a message awaiting a worker.
        return JobStatus(job_id=job_id, status="queued")
    if task.failed():
        raise HTTPException(status_code=500, detail="The background job failed.")
    return JobStatus(job_id=job_id, status=task.state.lower(), result=task.result if task.successful() else None)
