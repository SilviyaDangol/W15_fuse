"""Deterministic, dependency-free text chunking used before vectorization."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring a whitespace boundary."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end + 1)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = end - overlap
        # Never begin the next chunk partway through a word.
        if start > 0 and normalized[start - 1] != " ":
            start = normalized.rfind(" ", 0, start) + 1
    return chunks
