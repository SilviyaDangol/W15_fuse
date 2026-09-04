from pathlib import Path
from typing import Annotated
from uuid import uuid4
import logging

from celery.result import AsyncResult
from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile, status
from openai import APIError

from app.api.dependencies import get_llm_gateway
from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.schemas.chat import AssistantAnswer, ChatRequest
from app.schemas.documents import DocumentIngestionAccepted
from app.schemas.jobs import DemoBatchRequest, JobCreated, JobStatus
from app.services.llm_gateway import LLMGateway
from app.tasks.ingestion import ingest_text_document, process_demo_batch

router = APIRouter()
logger = logging.getLogger(__name__)
SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


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
    gateway: LLMGateway = Depends(get_llm_gateway),
) -> AssistantAnswer:
    """Generate a schema-validated answer through the configured model provider."""
    try:
        return await gateway.answer(payload.query, payload.provider)
    except (ValueError, APIError) as error:
        logger.warning("chat_unavailable provider=%s error=%s", payload.provider, type(error).__name__)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error


@router.post("/documents/ingest", response_model=DocumentIngestionAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="Upload a UTF-8 .txt/.md, .docx, or text-based .pdf document (maximum 5 MB)."),
    ]
) -> DocumentIngestionAccepted:
    """Store a document and queue its extraction, embedding, and indexing work."""
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
    stored_path.write_bytes(content)
    task = ingest_text_document.delay(document_id, file.filename, str(stored_path.resolve()))
    logger.info("document_queued document_id=%s filename=%s bytes=%s job_id=%s", document_id, file.filename, len(content), task.id)
    return DocumentIngestionAccepted(document_id=document_id, filename=file.filename, job_id=task.id)


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
