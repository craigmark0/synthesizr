import json
import logging
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from google import genai
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings
from src.db import get_db
from src.ingest import store_document
from src.query import search_chunks, synthesize

logger = logging.getLogger(__name__)

app = FastAPI(title="Synthesizr")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
SNIPPET_CHARS = 200


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


class IngestRequest(BaseModel):
    text: str
    source: str
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_stored: int


class SourceChunk(BaseModel):
    source: str
    document_id: str
    content: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    threshold: Optional[float] = None
    fallback_top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.post("/ingest", response_model=IngestResponse)
def ingest_json(
    request: IngestRequest,
    db: Session = Depends(get_db),
    client: genai.Client = Depends(get_gemini_client),
):
    try:
        document_id, chunks_stored = store_document(
            text=request.text,
            source=request.source,
            user_metadata=request.metadata or {},
            session=db,
            client=client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="Embedding service error")
    return IngestResponse(document_id=document_id, chunks_stored=chunks_stored)


@app.post("/ingest/upload", response_model=IngestResponse)
def ingest_upload(
    file: UploadFile = File(...),
    source: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    client: genai.Client = Depends(get_gemini_client),
):
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")
    try:
        text_content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded text")

    src = source or file.filename or "upload"

    if metadata:
        try:
            user_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="metadata must be valid JSON")
        if not isinstance(user_metadata, dict):
            raise HTTPException(status_code=422, detail="metadata must be a JSON object")
    else:
        user_metadata = {}

    try:
        document_id, chunks_stored = store_document(
            text=text_content,
            source=src,
            user_metadata=user_metadata,
            session=db,
            client=client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError:
        raise HTTPException(status_code=502, detail="Embedding service error")
    return IngestResponse(document_id=document_id, chunks_stored=chunks_stored)


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(
    request: QueryRequest,
    include_content_snippets: bool = False,
    db: Session = Depends(get_db),
    client: genai.Client = Depends(get_gemini_client),
):
    try:
        chunks = search_chunks(
            question=request.question,
            session=db,
            client=client,
            threshold=request.threshold,
            fallback_top_k=request.fallback_top_k,
        )
        answer = synthesize(question=request.question, chunks=chunks, client=client)
    except RuntimeError:
        raise HTTPException(status_code=502, detail="Embedding service error")

    seen = set()
    sources = []
    for c in chunks:
        if c["document_id"] not in seen:
            seen.add(c["document_id"])
            sources.append(SourceChunk(
                content=c["content"][:SNIPPET_CHARS] if include_content_snippets else None,
                source=c["source"],
                document_id=c["document_id"],
            ))
    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        logger.error("Health check DB query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
