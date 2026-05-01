import pytest
from unittest.mock import MagicMock

from src.config import settings
from src.ingest import chunk_text, embed_text


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


def test_chunk_text_raises_when_overlap_not_less_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_text_whitespace_only_returns_empty_list():
    result = chunk_text("   \n\t  ", chunk_size=1000, overlap=200)
    assert result == []


def test_chunk_text_three_chunks():
    text = "a" * 2600
    result = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(result) == 3
    assert len(result[0]) == 1000
    assert len(result[1]) == 1000
    assert len(result[2]) == 1000


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


def test_embed_text_raises_when_gemini_returns_no_embeddings():
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value.embeddings = []

    with pytest.raises(ValueError, match="no embeddings"):
        embed_text("hello", client=mock_client)


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
