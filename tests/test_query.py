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
