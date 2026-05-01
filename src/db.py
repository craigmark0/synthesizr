import enum
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, Text, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, relationship

from src.config import settings

engine = create_engine(settings.database_url)


class Base(DeclarativeBase):
    pass


class ChunkType(enum.Enum):
    text = "text"
    video = "video"
    audio = "audio"
    image = "image"


class Chunk(Base):
    __tablename__ = "chunks"
    __mapper_args__ = {
        "polymorphic_on": "chunk_type",
        "polymorphic_identity": None,
    }

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


# Column(default=...) fires at INSERT time, not at Python object construction.
# This listener assigns the UUID immediately so chunk.id is never None after Chunk().
@event.listens_for(Chunk, "init")
def _chunk_set_default_id(target, args, kwargs):
    if not kwargs.get("id"):
        target.id = uuid.uuid4()


class TextChunk(Base):
    __tablename__ = "text_chunks"
    __mapper_args__ = {"polymorphic_identity": ChunkType.text}

    id = Column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    content = Column(Text, nullable=False)

    chunk = relationship("Chunk", back_populates="text_chunk", passive_deletes=True)


def get_db():
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
