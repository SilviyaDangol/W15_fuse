import logging
from pathlib import Path

from celery import shared_task
from docx import Document

from app.core.config import get_settings
from app.services.chunking import chunk_text
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_demo_batch(self, documents: list[str]) -> dict:
    """Process a batch outside the API process.

    This will become the document chunking and embedding task in the RAG milestone.
    Keeping it as a task now means slow ingestion cannot block an HTTP request.
    """
    normalized = [document.strip() for document in documents if document.strip()]
    return {
        "documents_received": len(documents),
        "documents_processed": len(normalized),
        "total_characters": sum(len(document) for document in normalized),
    }


@shared_task(bind=True, autoretry_for=(OSError,), retry_backoff=True, max_retries=3)
def ingest_text_document(self, document_id: str, filename: str, file_path: str) -> dict:
    """Extract, chunk, embed, and persist a supported document in the worker."""
    settings = get_settings()
    path = Path(file_path)
    if path.suffix.lower() == ".docx":
        text = "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    elif path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF support is not installed. Rebuild the Docker image with docker compose up --build.") from error
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        with path.open(encoding="utf-8") as source:
            text = source.read()
    if not text.strip():
        raise ValueError("The document contains no extractable text. Scanned PDFs need OCR support.")
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    vectors_created = VectorStore(settings).index_chunks(document_id, filename, chunks)
    logger.info(
        "document_indexed document_id=%s filename=%s chunks=%s vectors=%s",
        document_id,
        filename,
        len(chunks),
        vectors_created,
    )
    return {
        "document_id": document_id,
        "filename": filename,
        "chunks_created": len(chunks),
        "vectors_created": vectors_created,
        "status": "indexed",
    }
