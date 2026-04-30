# Synthesizr MVP Design

**Date:** 2026-05-01
**Scope:** Text-only MVP — API-only, local infrastructure, Docker Compose

---

## Overview

Synthesizr is a general-purpose multimodal RAG (Retrieval-Augmented Generation) tool. Users ingest content and query it with natural language questions. The system retrieves semantically relevant content and synthesises an answer using an LLM.

The MVP is text-only, API-only, and runs entirely in Docker. It establishes the full RAG pipeline end-to-end and a database schema designed to extend to video, audio, and images without restructuring.

---

## Stack

| Concern | Choice |
|---|---|
| Language | Python |
| API framework | FastAPI |
| Embedding model | Gemini Embedding 2 (via Google Gemini Python SDK) |
| LLM | Gemini 2.5 Flash (via Google Gemini Python SDK) |
| Vector store | PostgreSQL + pgvector extension |
| ORM / query builder | SQLAlchemy |
| Migrations | Alembic |
| Infrastructure | Docker Compose |

---

## Project Structure

```
synthesizr/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── alembic/
│   ├── env.py
│   └── versions/
└── src/
    ├── main.py       # FastAPI app and route definitions
    ├── config.py     # Settings loaded from environment variables
    ├── db.py         # SQLAlchemy engine, session, and model definitions
    ├── ingest.py     # Chunking, embedding, and storage logic
    └── query.py      # Embed query, retrieve chunks, synthesise answer
```

---

## Database Schema

Class Table Inheritance — a parent `chunks` table holds all common fields including the embedding vector. Each content type has a child table with type-specific fields. The child table's primary key is a foreign key to `chunks.id`.

### `chunks` (parent)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `chunk_type` | enum | `'text'`, `'video'`, `'audio'`, `'image'` |
| `document_id` | UUID | Groups all chunks from one ingested document |
| `chunk_index` | integer | Position of this chunk within the document |
| `source` | text | Filename or label provided at ingest time |
| `embedding` | vector(3072) | Gemini Embedding 2 output |
| `ingested_at` | timestamptz | Set automatically on insert |
| `metadata` | JSONB | Open user-defined key-value fields |

### `text_chunks` (child — MVP)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key, FK → `chunks.id` |
| `content` | text | The actual chunk text |

### Future child tables (not in MVP)

- `video_chunks` — `start_time`, `end_time`, `transcript`
- `audio_chunks` — `start_time`, `end_time`, `transcript`
- `image_chunks` — `caption`, `ocr_text`

---

## API Endpoints

### `POST /ingest`

Accepts text content via either:
- **JSON body:** `{ "text": "...", "source": "my-doc.txt", "metadata": { ... } }`
- **File upload (multipart):** `file=@my-doc.txt`, `source=my-doc.txt`, `metadata=<JSON string, e.g. '{"author":"John"}'>`

**Processing steps:**
1. Extract text from file or JSON body
2. Chunk text into ~1000 character segments with 200 character overlap
3. Embed each chunk via Gemini Embedding 2
4. Store each chunk in `chunks` + `text_chunks`, merging system metadata with user metadata
5. Return document ID and chunk count

**Response:**
```json
{ "document_id": "abc123", "chunks_stored": 12 }
```

**System metadata** (always set automatically):
- `document_id`, `chunk_index`, `ingested_at`, `source`

**User metadata** (optional, open-ended):
- Any key-value pairs passed in the `metadata` field
- Stored in the JSONB `metadata` column — no schema enforced
- Examples: `published_at`, `author`, `url`, `program_datetime`

### `POST /query`

Accepts a natural language question.

**Processing steps:**
1. Embed the question via Gemini Embedding 2
2. Run cosine similarity search against `chunks.embedding` in Postgres
3. Return all chunks above a similarity threshold of 0.7; fall back to top 5 if nothing clears the threshold
4. Pass retrieved chunk texts + original question to Gemini 2.5 Flash
5. System prompt instructs the LLM to answer using only the provided context and to say so if the context is insufficient
6. Return the answer and the source chunks used

**Request:**
```json
{ "question": "What does the document say about X?" }
```

**Response:**
```json
{
  "answer": "...",
  "sources": [
    { "text": "...", "source": "my-doc.txt", "document_id": "abc123" }
  ]
}
```

### `GET /health`

Returns 200 if the API is running and Postgres is reachable. Used for Docker health checks.

---

## Docker Setup

Two services in `docker-compose.yml`:

**`postgres`**
- Image: `pgvector/pgvector:pg16` (Postgres 16 with pgvector pre-installed)
- Data persisted to a named Docker volume
- Exposes port 5432 (internal only)

**`api`**
- Built from `Dockerfile` (Python 3.12 slim base)
- Depends on `postgres` passing a health check before starting
- Reads `GOOGLE_API_KEY` from `.env` (never committed)
- Exposes port 8000
- Mounts `src/` for development hot-reload

**.env.example**
```
GOOGLE_API_KEY=
DATABASE_URL=postgresql://synthesizr:synthesizr@postgres:5432/synthesizr
```

---

## Post-MVP Backlog

- **Agentic retrieval loop** — if the LLM determines retrieved context is insufficient, it generates a follow-up search query and retrieves again (max 2–3 iterations)
- **Query reformulation** — a pre-step that rewrites the raw user question into a search-optimised query before hitting the vector store
- **Video chunk support** — direct video embedding for short clips; transcribe-then-embed for long-form podcast content
- **Audio chunk support**
- **Image chunk support**
- **Auth** — API key or session-based auth before multi-user or production use
- **Document management** — `GET /documents`, `DELETE /documents/{id}`
