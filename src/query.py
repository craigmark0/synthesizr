from typing import Optional

from google import genai
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings
from src.ingest import embed_text


def search_chunks(
    question: str,
    session: Session,
    client: genai.Client,
    threshold: Optional[float] = None,
    fallback_top_k: Optional[int] = None,
) -> list[dict]:
    sim_threshold = threshold if threshold is not None else settings.similarity_threshold
    top_k = fallback_top_k if fallback_top_k is not None else settings.fallback_top_k

    query_vector = embed_text(question, client=client)
    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

    rows = session.execute(
        text("""
            SELECT tc.content,
                   c.source,
                   CAST(c.document_id AS TEXT) AS document_id,
                   1 - (c.embedding <=> CAST(:vector AS vector)) AS similarity
            FROM chunks c
            JOIN text_chunks tc ON tc.id = c.id
            ORDER BY c.embedding <=> CAST(:vector AS vector)
            LIMIT :limit
        """),
        {"vector": vector_str, "limit": top_k},
    ).fetchall()

    above = [
        {
            "content": r.content,
            "source": r.source,
            "document_id": r.document_id,
            "similarity": r.similarity,
        }
        for r in rows
        if r.similarity >= sim_threshold
    ]

    if above:
        return above

    return [
        {
            "content": r.content,
            "source": r.source,
            "document_id": r.document_id,
            "similarity": r.similarity,
        }
        for r in rows[:top_k]
    ]
