# Synthesizr MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a text-only RAG API that ingests text documents, stores them as searchable vector embeddings in Postgres, and answers natural language questions by retrieving relevant chunks and synthesising an answer with Gemini.

**Architecture:** FastAPI handles HTTP. `ingest.py` chunks, embeds (Gemini Embedding 2), and stores text into Postgres + pgvector via SQLAlchemy. `query.py` embeds the question, retrieves semantically similar chunks via cosine similarity, and synthesises an answer using Gemini 2.5 Flash. Both the Gemini client and DB session are injected as dependencies, keeping all logic unit-testable without a real DB or API key. Alembic manages schema migrations.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL 16 + pgvector, google-genai SDK, pytest, pytest-mock

---

## File Map

| File | Responsibility |
|---|---|
| `docker-compose.yml` | Defines `api` and `postgres` services |
| `Dockerfile` | Python 3.12 image for the API |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for required environment variables |
| `alembic.ini` | Alembic configuration |
| `alembic/env.py` | Alembic runtime — reads DATABASE_URL from env, points at SQLAlchemy models |
| `alembic/versions/001_initial_schema.py` | Creates `chunks` and `text_chunks` tables with pgvector index |
| `src/__init__.py` | Empty — marks src as a package |
| `src/config.py` | Loads and validates env vars using pydantic-settings; exposes `settings` singleton |
| `src/db.py` | SQLAlchemy engine, `get_db()` dependency, ORM models: `Chunk`, `TextChunk`, `ChunkType` |
| `src/ingest.py` | `chunk_text()`, `embed_text()`, `store_document()` |
| `src/query.py` | `search_chunks()`, `synthesize()` |
| `src/main.py` | FastAPI app; routes: `POST /ingest`, `POST /ingest/upload`, `POST /query`, `GET /health` |
| `tests/__init__.py` | Empty — marks tests as a package |
| `tests/conftest.py` | Sets env vars before imports; shared fixtures: `mock_db`, `mock_client`, `test_client` |
| `tests/test_config.py` | Tests settings load correctly from env |
| `tests/test_db.py` | Tests ORM model instantiation |
| `tests/test_ingest.py` | Unit tests for `chunk_text`, `embed_text`, `store_document` |
| `tests/test_query.py` | Unit tests for `search_chunks`, `synthesize` |
| `tests/test_main.py` | Endpoint tests via FastAPI TestClient with mocked dependencies |

---

### Task 1: Project scaffolding

**Files:**
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
sqlalchemy==2.0.36
alembic==1.14.0
psycopg2-binary==2.9.10
pgvector==0.3.6
google-genai==0.8.0
pydantic-settings==2.6.1
python-multipart==0.0.12
pytest==8.3.4
pytest-mock==3.14.0
httpx==0.28.1
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: synthesizr
      POSTGRES_USER: synthesizr
      POSTGRES_PASSWORD: synthesizr
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U synthesizr"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./src:/app/src
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  pgdata:
```

- [ ] **Step 4: Create `.env.example`**

```
GOOGLE_API_KEY=
DATABASE_URL=postgresql://synthesizr:synthesizr@postgres:5432/synthesizr
```

- [ ] **Step 5: Create `.env` from `.env.example` and fill in your `GOOGLE_API_KEY`**

Copy `.env.example` to `.env` and paste your Gemini API key. Never commit `.env`.

Add `.env` to `.gitignore`:

```
.env
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 6: Create package init files**

```bash
mkdir -p src tests
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 7: Verify Docker builds without errors**

```bash
docker compose build
```

Expected: build completes with no errors. The final line will be something like `=> exporting to image`.

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml Dockerfile requirements.txt .env.example .gitignore src/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding"
```

---

### Task 2: Config module

**Files:**
- Create: `src/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create `tests/conftest.py` with environment variable setup**

This must be at the very top — before any src imports — so pydantic-settings can load env vars when modules are first imported during tests.

```python
import os

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://synthesizr:synthesizr@localhost:5432/synthesizr")
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_config.py`:

```python
from src.config import settings


def test_settings_loads_google_api_key():
    assert settings.google_api_key == "test-key"


def test_settings_loads_database_url():
    assert "postgresql" in settings.database_url


def test_settings_has_default_similarity_threshold():
    assert settings.similarity_threshold == 0.7


def test_settings_has_default_fallback_top_k():
    assert settings.fallback_top_k == 5


def test_settings_has_default_chunk_size():
    assert settings.chunk_size == 1000


def test_settings_has_default_chunk_overlap():
    assert settings.chunk_overlap == 200


def test_settings_has_embedding_model():
    assert settings.embedding_model == "gemini-embedding-2-preview"


def test_settings_has_llm_model():
    assert settings.llm_model == "gemini-2.5-flash"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 4: Implement `src/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str
    database_url: str
    similarity_threshold: float = 0.7
    fallback_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "gemini-embedding-2-preview"
    llm_model: str = "gemini-2.5-flash"


settings = Settings()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: config module with pydantic-settings"
```

---

### Task 3: Database models

**Files:**
- Create: `src/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db.py`:

```python
import uuid
from src.db import Chunk, ChunkType, TextChunk


def test_chunk_type_enum_values():
    assert ChunkType.text.value == "text"
    assert ChunkType.video.value == "video"
    assert ChunkType.audio.value == "audio"
    assert ChunkType.image.value == "image"


def test_chunk_model_instantiation():
    chunk = Chunk(
        chunk_type=ChunkType.text,
        document_id=uuid.uuid4(),
        chunk_index=0,
        source="test.txt",
        embedding=[0.1] * 3072,
        user_metadata={},
    )
    assert chunk.chunk_type == ChunkType.text
    assert chunk.chunk_index == 0
    assert chunk.source == "test.txt"


def test_text_chunk_model_instantiation():
    chunk_id = uuid.uuid4()
    text_chunk = TextChunk(id=chunk_id, content="hello world")
    assert text_chunk.content == "hello world"
    assert text_chunk.id == chunk_id


def test_chunk_has_generated_id_by_default():
    chunk = Chunk(
        chunk_type=ChunkType.text,
        document_id=uuid.uuid4(),
        chunk_index=0,
        source="test.txt",
        embedding=[0.0] * 3072,
        user_metadata={},
    )
    assert chunk.id is not None
    assert isinstance(chunk.id, uuid.UUID)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.db'`

- [ ] **Step 3: Implement `src/db.py`**

```python
import enum
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from src.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class ChunkType(enum.Enum):
    text = "text"
    video = "video"
    audio = "audio"
    image = "image"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_type = Column(Enum(ChunkType, name="chunk_type"), nullable=False)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    source = Column(Text, nullable=False)
    embedding = Column(Vector(3072), nullable=False)
    ingested_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    user_metadata = Column("metadata", JSONB, nullable=False, default=dict)

    text_chunk = relationship(
        "TextChunk",
        back_populates="chunk",
        uselist=False,
        cascade="all, delete-orphan",
    )


class TextChunk(Base):
    __tablename__ = "text_chunks"

    id = Column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    content = Column(Text, nullable=False)

    chunk = relationship("Chunk", back_populates="text_chunk")


def get_db():
    with Session(engine) as session:
        yield session
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/db.py tests/test_db.py
git commit -m "feat: SQLAlchemy models for chunks and text_chunks"
```

---

### Task 4: Alembic setup and initial migration

**Files:**
- Create: `alembic.ini` (via `alembic init`)
- Modify: `alembic/env.py`
- Create: `alembic/versions/001_initial_schema.py`

- [ ] **Step 1: Initialise Alembic**

```bash
alembic init alembic
```

Expected: creates `alembic.ini` and `alembic/` directory with `env.py`, `script.py.mako`, and `versions/`.

- [ ] **Step 2: Clear the `sqlalchemy.url` in `alembic.ini`**

Open `alembic.ini`. Find the line:
```
sqlalchemy.url = driver://user:pass@localhost/dbname
```
Replace it with:
```
sqlalchemy.url =
```

We read the URL from the environment in `env.py` instead.

- [ ] **Step 3: Replace `alembic/env.py` entirely**

```python
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.db import Base

config = context.config
config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Create `alembic/versions/001_initial_schema.py`**

```python
"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-01
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE TYPE chunk_type AS ENUM ('text', 'video', 'audio', 'image')")

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "chunk_type",
            sa.Enum("text", "video", "audio", "image", name="chunk_type"),
            nullable=False,
        ),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("embedding", Vector(3072), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )

    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)"
    )

    op.create_table(
        "text_chunks",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("content", sa.Text, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("text_chunks")
    op.drop_table("chunks")
    op.execute("DROP TYPE chunk_type")
    op.execute("DROP EXTENSION IF EXISTS vector")
```

- [ ] **Step 5: Start Postgres and run the migration**

```bash
docker compose up postgres -d
DATABASE_URL=postgresql://synthesizr:synthesizr@localhost:5432/synthesizr alembic upgrade head
```

Expected output ends with: `Running upgrade  -> 001, initial schema`

- [ ] **Step 6: Verify the schema was created**

```bash
docker compose exec postgres psql -U synthesizr -d synthesizr -c "\dt"
```

Expected: two rows — `chunks` and `text_chunks`.

- [ ] **Step 7: Commit**

```bash
git add alembic.ini alembic/
git commit -m "feat: Alembic setup and initial schema migration"
```

---

### Task 5: Text chunking

**Files:**
- Create: `src/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest.py`:

```python
from src.ingest import chunk_text


def test_chunk_text_short_input_returns_single_chunk():
    result = chunk_text("hello world", chunk_size=1000, overlap=200)
    assert result == ["hello world"]


def test_chunk_text_empty_input_returns_empty_list():
    result = chunk_text("", chunk_size=1000, overlap=200)
    assert result == []


def test_chunk_text_splits_at_chunk_size():
    text = "a" * 1200
    result = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(result) == 2
    assert len(result[0]) == 1000


def test_chunk_text_overlap_preserved_in_second_chunk():
    # "a" * 1000 + "b" * 200 = 1200 chars
    # chunk 1: chars 0–999 (all a's)
    # next start: 1000 - 200 = 800
    # chunk 2: chars 800–1199 (200 a's + 200 b's)
    text = "a" * 1000 + "b" * 200
    result = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(result) == 2
    assert result[1].startswith("a" * 200)
    assert result[1].endswith("b" * 200)


def test_chunk_text_exact_chunk_size_returns_single_chunk():
    text = "a" * 1000
    result = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(result) == 1
    assert result[0] == text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingest'`

- [ ] **Step 3: Create `src/ingest.py` with `chunk_text`**

```python
from src.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    size = chunk_size if chunk_size is not None else settings.chunk_size
    ovlp = overlap if overlap is not None else settings.chunk_overlap
    if not text:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: chunk_text with configurable size and overlap"
```

---

### Task 6: Text embedding

**Files:**
- Modify: `src/ingest.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Add the failing tests**

Add to the bottom of `tests/test_ingest.py`:

```python
from unittest.mock import MagicMock
from src.config import settings
from src.ingest import embed_text


def test_embed_text_returns_vector_of_correct_length():
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 3072
    mock_client.models.embed_content.return_value.embeddings = [mock_embedding]

    result = embed_text("hello world", client=mock_client)

    assert len(result) == 3072
    assert result[0] == 0.1


def test_embed_text_calls_gemini_with_correct_model():
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.0] * 3072
    mock_client.models.embed_content.return_value.embeddings = [mock_embedding]

    embed_text("some text", client=mock_client)

    mock_client.models.embed_content.assert_called_once_with(
        model=settings.embedding_model,
        contents="some text",
    )
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pytest tests/test_ingest.py::test_embed_text_returns_vector_of_correct_length tests/test_ingest.py::test_embed_text_calls_gemini_with_correct_model -v
```

Expected: FAIL — `ImportError: cannot import name 'embed_text'`

- [ ] **Step 3: Add `embed_text` to `src/ingest.py`**

Add this import at the top of `src/ingest.py`:
```python
from google import genai
```

Add this function after `chunk_text`:

```python
def embed_text(text: str, client: genai.Client) -> list[float]:
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
    )
    return result.embeddings[0].values
```

- [ ] **Step 4: Run all ingest tests to verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: embed_text using Gemini Embedding 2"
```

---

### Task 7: Document storage

**Files:**
- Modify: `src/ingest.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Add the failing tests**

Add to the bottom of `tests/test_ingest.py`:

```python
import uuid
from src.ingest import store_document
from src.db import ChunkType


def test_store_document_returns_document_id_and_chunk_count():
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 3072
    mock_client.models.embed_content.return_value.embeddings = [mock_embedding]

    # "a" * 1000 + "b" * 500 = 1500 chars → 2 chunks at size=1000, overlap=200
    doc_id, chunk_count = store_document(
        text="a" * 1000 + "b" * 500,
        source="test.txt",
        user_metadata={"author": "alice"},
        session=mock_session,
        client=mock_client,
    )

    assert isinstance(uuid.UUID(doc_id), uuid.UUID)
    assert chunk_count == 2
    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()


def test_store_document_sets_correct_metadata_on_chunk():
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 3072
    mock_client.models.embed_content.return_value.embeddings = [mock_embedding]

    store_document(
        text="short text",
        source="my-doc.txt",
        user_metadata={"published_at": "2024-01-01"},
        session=mock_session,
        client=mock_client,
    )

    stored_chunk = mock_session.add.call_args[0][0]
    assert stored_chunk.user_metadata == {"published_at": "2024-01-01"}
    assert stored_chunk.source == "my-doc.txt"
    assert stored_chunk.chunk_type == ChunkType.text
    assert stored_chunk.chunk_index == 0
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pytest tests/test_ingest.py::test_store_document_returns_document_id_and_chunk_count tests/test_ingest.py::test_store_document_sets_correct_metadata_on_chunk -v
```

Expected: FAIL — `ImportError: cannot import name 'store_document'`

- [ ] **Step 3: Add `store_document` to `src/ingest.py`**

Add these imports at the top of `src/ingest.py` (after the existing imports):
```python
import uuid
from typing import Optional
from sqlalchemy.orm import Session
from src.db import Chunk, ChunkType, TextChunk
```

Add this function after `embed_text`:

```python
def store_document(
    text: str,
    source: str,
    user_metadata: Optional[dict],
    session: Session,
    client: genai.Client,
) -> tuple[str, int]:
    chunks = chunk_text(text)
    document_id = uuid.uuid4()

    for index, chunk_content in enumerate(chunks):
        embedding = embed_text(chunk_content, client=client)
        chunk = Chunk(
            chunk_type=ChunkType.text,
            document_id=document_id,
            chunk_index=index,
            source=source,
            embedding=embedding,
            user_metadata=user_metadata or {},
        )
        chunk.text_chunk = TextChunk(id=chunk.id, content=chunk_content)
        session.add(chunk)

    session.commit()
    return str(document_id), len(chunks)
```

- [ ] **Step 4: Run all ingest tests to verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: store_document chunks, embeds, and persists text to Postgres"
```

---

### Task 8: Ingest endpoints + health check

**Files:**
- Create: `src/main.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Add FastAPI fixtures to `tests/conftest.py`**

Add these imports and fixtures below the existing env var lines:

```python
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.main import app, get_gemini_client
from src.db import get_db


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_client():
    client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 3072
    client.models.embed_content.return_value.embeddings = [mock_embedding]
    return client


@pytest.fixture
def test_client(mock_db, mock_client):
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_gemini_client] = lambda: mock_client
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Write the failing endpoint tests**

Create `tests/test_main.py`:

```python
import json


def test_health_returns_ok(test_client, mock_db):
    from unittest.mock import MagicMock
    mock_db.execute.return_value = MagicMock()

    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_json_body_returns_document_id_and_chunk_count(test_client):
    response = test_client.post("/ingest", json={
        "text": "This is a test document with enough content.",
        "source": "test.txt",
        "metadata": {"author": "alice"},
    })
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["chunks_stored"] >= 1


def test_ingest_json_body_without_metadata_succeeds(test_client):
    response = test_client.post("/ingest", json={
        "text": "Plain text with no metadata.",
        "source": "plain.txt",
    })
    assert response.status_code == 200
    assert "document_id" in response.json()


def test_ingest_file_upload_returns_document_id_and_chunk_count(test_client, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document uploaded as a file.")

    with open(test_file, "rb") as f:
        response = test_client.post(
            "/ingest/upload",
            files={"file": ("test.txt", f, "text/plain")},
            data={
                "source": "test.txt",
                "metadata": json.dumps({"author": "bob"}),
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["chunks_stored"] >= 1


def test_ingest_file_upload_without_metadata_succeeds(test_client, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("File content with no metadata.")

    with open(test_file, "rb") as f:
        response = test_client.post(
            "/ingest/upload",
            files={"file": ("test.txt", f, "text/plain")},
            data={"source": "test.txt"},
        )
    assert response.status_code == 200


def test_ingest_json_body_requires_text_field(test_client):
    response = test_client.post("/ingest", json={"source": "test.txt"})
    assert response.status_code == 422


def test_ingest_json_body_requires_source_field(test_client):
    response = test_client.post("/ingest", json={"text": "some text"})
    assert response.status_code == 422
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_main.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.main'`

- [ ] **Step 4: Create `src/main.py`**

```python
import json
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from google import genai
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings
from src.db import get_db
from src.ingest import store_document

app = FastAPI(title="Synthesizr")


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


class IngestRequest(BaseModel):
    text: str
    source: str
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_stored: int


@app.post("/ingest", response_model=IngestResponse)
def ingest_json(
    request: IngestRequest,
    db: Session = Depends(get_db),
    client: genai.Client = Depends(get_gemini_client),
):
    document_id, chunks_stored = store_document(
        text=request.text,
        source=request.source,
        user_metadata=request.metadata or {},
        session=db,
        client=client,
    )
    return IngestResponse(document_id=document_id, chunks_stored=chunks_stored)


@app.post("/ingest/upload", response_model=IngestResponse)
def ingest_upload(
    file: UploadFile = File(...),
    source: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    client: genai.Client = Depends(get_gemini_client),
):
    text_content = file.file.read().decode("utf-8")
    src = source or file.filename or "upload"
    user_metadata = json.loads(metadata) if metadata else {}

    document_id, chunks_stored = store_document(
        text=text_content,
        source=src,
        user_metadata=user_metadata,
        session=db,
        client=client,
    )
    return IngestResponse(document_id=document_id, chunks_stored=chunks_stored)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
```

- [ ] **Step 5: Run all tests to verify they pass**

```bash
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/main.py tests/conftest.py tests/test_main.py
git commit -m "feat: POST /ingest, POST /ingest/upload, GET /health endpoints"
```

---

### Task 9: Similarity search

**Files:**
- Create: `src/query.py`
- Create: `tests/test_query.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_query.py`:

```python
from unittest.mock import MagicMock
from src.query import search_chunks


def _make_row(content: str, source: str, doc_id: str, similarity: float) -> MagicMock:
    row = MagicMock()
    row.content = content
    row.source = source
    row.document_id = doc_id
    row.similarity = similarity
    return row


def test_search_chunks_returns_chunks_above_threshold():
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 3072
    mock_client.models.embed_content.return_value.embeddings = [mock_embedding]

    rows = [
        _make_row("very relevant", "doc.txt", "abc-123", 0.9),
        _make_row("somewhat relevant", "doc2.txt", "def-456", 0.75),
        _make_row("not relevant", "doc3.txt", "ghi-789", 0.3),
    ]
    mock_session.execute.return_value.fetchall.return_value = rows

    results = search_chunks(
        question="what is this?",
        session=mock_session,
        client=mock_client,
        threshold=0.7,
        fallback_top_k=5,
    )

    assert len(results) == 2
    assert results[0]["content"] == "very relevant"
    assert results[1]["content"] == "somewhat relevant"


def test_search_chunks_falls_back_to_top_k_when_nothing_above_threshold():
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 3072
    mock_client.models.embed_content.return_value.embeddings = [mock_embedding]

    rows = [_make_row(f"text {i}", "doc.txt", "abc", 0.2) for i in range(10)]
    mock_session.execute.return_value.fetchall.return_value = rows

    results = search_chunks(
        question="what is this?",
        session=mock_session,
        client=mock_client,
        threshold=0.7,
        fallback_top_k=5,
    )

    assert len(results) == 5


def test_search_chunks_result_contains_expected_keys():
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 3072
    mock_client.models.embed_content.return_value.embeddings = [mock_embedding]

    rows = [_make_row("some content", "source.txt", "doc-id", 0.8)]
    mock_session.execute.return_value.fetchall.return_value = rows

    results = search_chunks(
        question="question",
        session=mock_session,
        client=mock_client,
        threshold=0.7,
        fallback_top_k=5,
    )

    assert results[0].keys() == {"content", "source", "document_id", "similarity"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_query.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.query'`

- [ ] **Step 3: Create `src/query.py` with `search_chunks`**

```python
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
            LIMIT 50
        """),
        {"vector": vector_str},
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
```

- [ ] **Step 4: Run all query tests to verify they pass**

```bash
pytest tests/test_query.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/query.py tests/test_query.py
git commit -m "feat: search_chunks with cosine similarity threshold and top-k fallback"
```

---

### Task 10: LLM synthesis

**Files:**
- Modify: `src/query.py`
- Modify: `tests/test_query.py`

- [ ] **Step 1: Add the failing tests**

Add to the bottom of `tests/test_query.py`:

```python
from unittest.mock import MagicMock
from src.query import synthesize


def test_synthesize_returns_llm_answer():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = "The answer is 42."

    chunks = [{"content": "42 is the answer.", "source": "doc.txt", "document_id": "abc", "similarity": 0.9}]
    result = synthesize(question="What is the answer?", chunks=chunks, client=mock_client)

    assert result == "The answer is 42."


def test_synthesize_includes_chunk_content_in_prompt():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = "Some answer."

    chunks = [{"content": "Important fact.", "source": "doc.txt", "document_id": "abc", "similarity": 0.9}]
    synthesize(question="What happened?", chunks=chunks, client=mock_client)

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    prompt = call_kwargs["contents"]
    assert "Important fact." in prompt
    assert "What happened?" in prompt


def test_synthesize_includes_source_in_prompt():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = "Answer."

    chunks = [{"content": "Fact here.", "source": "my-source.txt", "document_id": "abc", "similarity": 0.9}]
    synthesize(question="Q?", chunks=chunks, client=mock_client)

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    prompt = call_kwargs["contents"]
    assert "my-source.txt" in prompt
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pytest tests/test_query.py::test_synthesize_returns_llm_answer tests/test_query.py::test_synthesize_includes_chunk_content_in_prompt tests/test_query.py::test_synthesize_includes_source_in_prompt -v
```

Expected: FAIL — `ImportError: cannot import name 'synthesize'`

- [ ] **Step 3: Add `synthesize` to `src/query.py`**

Add this import at the top of `src/query.py`:
```python
from google.genai import types
```

Add these below `search_chunks`:

```python
_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the context provided. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Do not use knowledge outside of the provided context."
)


def synthesize(question: str, chunks: list[dict], client: genai.Client) -> str:
    context_sections = "\n\n---\n\n".join(
        f"Source: {c['source']}\n{c['content']}" for c in chunks
    )
    prompt = f"Context:\n{context_sections}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=settings.llm_model,
        config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
        contents=prompt,
    )
    return response.text
```

- [ ] **Step 4: Run all query tests to verify they pass**

```bash
pytest tests/test_query.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/query.py tests/test_query.py
git commit -m "feat: synthesize answer from retrieved chunks using Gemini 2.5 Flash"
```

---

### Task 11: Query endpoint

**Files:**
- Modify: `src/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Add the failing tests**

Add to the bottom of `tests/test_main.py`:

```python
def test_query_returns_answer_and_sources(test_client, mock_db, mock_client):
    mock_client.models.generate_content.return_value.text = "The sky is blue."

    from unittest.mock import MagicMock
    mock_row = MagicMock()
    mock_row.content = "Blue sky due to Rayleigh scattering."
    mock_row.source = "science.txt"
    mock_row.document_id = "abc-123"
    mock_row.similarity = 0.85
    mock_db.execute.return_value.fetchall.return_value = [mock_row]

    response = test_client.post("/query", json={"question": "Why is the sky blue?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The sky is blue."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "science.txt"
    assert data["sources"][0]["document_id"] == "abc-123"


def test_query_requires_question_field(test_client):
    response = test_client.post("/query", json={})
    assert response.status_code == 422
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pytest tests/test_main.py::test_query_returns_answer_and_sources tests/test_main.py::test_query_requires_question_field -v
```

Expected: FAIL — `404 Not Found` (route not yet defined)

- [ ] **Step 3: Add `POST /query` to `src/main.py`**

Add this import at the top of `src/main.py`:
```python
from src.query import search_chunks, synthesize
```

Add these classes and the route after the `/ingest/upload` route:

```python
class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    content: str
    source: str
    document_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    client: genai.Client = Depends(get_gemini_client),
):
    chunks = search_chunks(question=request.question, session=db, client=client)
    answer = synthesize(question=request.question, chunks=chunks, client=client)
    sources = [
        SourceChunk(content=c["content"], source=c["source"], document_id=c["document_id"])
        for c in chunks
    ]
    return QueryResponse(answer=answer, sources=sources)
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: POST /query endpoint — retrieve chunks and synthesise answer"
```

---

### Task 12: End-to-end smoke test

No new code — this verifies the full running system.

- [ ] **Step 1: Start the full stack**

```bash
docker compose up --build -d
```

Expected: both containers start. Check with:
```bash
docker compose ps
```
Both `api` and `postgres` should show `running`.

- [ ] **Step 2: Confirm the API is healthy**

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 3: Ingest a document via JSON body**

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Python programming language was created by Guido van Rossum and first released in 1991. Python emphasises code readability and uses significant whitespace.",
    "source": "python-facts.txt",
    "metadata": {"topic": "programming"}
  }' | python3 -m json.tool
```

Expected:
```json
{
    "document_id": "<some-uuid>",
    "chunks_stored": 1
}
```

- [ ] **Step 4: Ingest a document via file upload**

```bash
echo "FastAPI is a modern, fast web framework for building APIs with Python. It is based on standard Python type hints and provides automatic interactive API documentation at /docs." \
  > /tmp/fastapi-facts.txt

curl -s -X POST http://localhost:8000/ingest/upload \
  -F "file=@/tmp/fastapi-facts.txt" \
  -F "source=fastapi-facts.txt" \
  -F 'metadata={"topic":"web frameworks"}' | python3 -m json.tool
```

Expected:
```json
{
    "document_id": "<some-uuid>",
    "chunks_stored": 1
}
```

- [ ] **Step 5: Query the ingested content**

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who created Python and when was it released?"}' | python3 -m json.tool
```

Expected: a response with `answer` mentioning Guido van Rossum and 1991, and `sources` listing `python-facts.txt`.

- [ ] **Step 6: Browse the auto-generated API docs**

Open `http://localhost:8000/docs` in a browser. You should see the Swagger UI with all three endpoints documented and interactive.
