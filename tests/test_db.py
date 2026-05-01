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
