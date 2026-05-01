import uuid
from typing import Optional

from google import genai
from sqlalchemy.orm import Session

from src.config import settings
from src.db import Chunk, ChunkType, TextChunk


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


def embed_text(text: str, client: genai.Client) -> list[float]:
    try:
        result = client.models.embed_content(
            model=settings.embedding_model,
            contents=text,
        )
    except Exception as exc:
        raise RuntimeError(f"embed_text failed for input of length {len(text)}") from exc
    if not result.embeddings:
        raise ValueError(f"Gemini returned no embeddings for input: {text!r:.50}")
    return result.embeddings[0].values


def store_document(
    text: str,
    source: str,
    user_metadata: Optional[dict],
    session: Session,
    client: genai.Client,
) -> tuple[str, int]:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("text produced no chunks after processing")
    document_id = uuid.uuid4()

    for index, chunk_content in enumerate(chunks):
        embedding = embed_text(chunk_content, client=client)
        chunk_id = uuid.uuid4()
        chunk = Chunk(
            id=chunk_id,
            chunk_type=ChunkType.text,
            document_id=document_id,
            chunk_index=index,
            source=source,
            embedding=embedding,
            user_metadata=user_metadata or {},
        )
        chunk.text_chunk = TextChunk(id=chunk_id, content=chunk_content)
        session.add(chunk)

    session.flush()
    return str(document_id), len(chunks)
