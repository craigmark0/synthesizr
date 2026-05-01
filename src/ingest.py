from src.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    size = chunk_size if chunk_size is not None else settings.chunk_size
    ovlp = overlap if overlap is not None else settings.chunk_overlap
    if ovlp >= size:
        raise ValueError(
            f"overlap ({ovlp}) must be less than chunk_size ({size})"
        )
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - ovlp
    return chunks
