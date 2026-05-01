"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-01
"""
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE TYPE chunk_type AS ENUM ('text', 'video', 'audio', 'image')")

    op.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY,
            chunk_type chunk_type NOT NULL,
            document_id UUID NOT NULL,
            chunk_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            embedding vector(3072) NOT NULL,
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks (document_id)")
    # vector(3072) exceeds ivfflat/hnsw's 2000-dim limit; cast to halfvec for the ANN index.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding ON chunks "
        "USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (
            id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            content TEXT NOT NULL
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS text_chunks")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TYPE IF EXISTS chunk_type")
    op.execute("DROP EXTENSION IF EXISTS vector")
