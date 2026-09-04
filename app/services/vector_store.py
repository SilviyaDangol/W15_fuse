"""OpenAI embeddings plus a persistent, local ChromaDB vector store."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
from openai import OpenAI

from app.core.config import Settings, get_settings
from app.schemas.chat import Source


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_id: str
    text: str
    distance: float

    def as_source(self) -> Source:
        return Source(
            document_id=self.document_id,
            filename=self.filename,
            chunk_id=self.chunk_id,
            excerpt=self.text[:400],
        )


class VectorStore:
    """Stores application-created embeddings; Chroma never sends data elsewhere."""

    collection_name = "document_chunks"

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client
        directory = Path(self.settings.vector_db_dir)
        directory.mkdir(parents=True, exist_ok=True)
        chroma = chromadb.PersistentClient(path=str(directory))
        self.collection = chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, document_id: str, filename: str, chunks: Sequence[str]) -> int:
        if not chunks:
            return 0
        # Embed the whole document batch in one request to reduce indexing round trips.
        embeddings = self._embed(list(chunks))
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        metadata = [
            {"document_id": document_id, "filename": filename, "chunk_index": index}
            for index in range(len(chunks))
        ]
        self.collection.upsert(ids=ids, documents=list(chunks), metadatas=metadata, embeddings=embeddings)
        return len(ids)

    def search(self, query: str, limit: int | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query or self.collection.count() == 0:
            return []
        # Retrieval happens before citations are created; models never invent source metadata.
        result = self.collection.query(
            query_embeddings=[self._embed([query])[0]],
            n_results=min(limit or self.settings.retrieval_top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadata = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            RetrievedChunk(
                document_id=str(item["document_id"]),
                filename=str(item["filename"]),
                chunk_id=str(chunk_id),
                text=str(text),
                distance=float(distance),
            )
            for chunk_id, text, item, distance in zip(result["ids"][0], documents, metadata, distances, strict=True)
        ]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to create and search document embeddings.")
        response = (self.client or OpenAI(api_key=self.settings.openai_api_key)).embeddings.create(
            model=self.settings.embedding_model,
            input=inputs,
            encoding_format="float",
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
