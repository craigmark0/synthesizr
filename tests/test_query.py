from unittest.mock import MagicMock
from src.config import settings
from src.query import search_chunks, synthesize


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


def test_synthesize_empty_chunks_returns_no_context_message():
    mock_client = MagicMock()
    result = synthesize(question="What is X?", chunks=[], client=mock_client)
    assert "No relevant context" in result
    mock_client.models.generate_content.assert_not_called()


def test_synthesize_calls_correct_model_with_system_instruction():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = "Answer."

    chunks = [{"content": "Fact.", "source": "doc.txt", "document_id": "abc", "similarity": 0.9}]
    synthesize(question="Q?", chunks=chunks, client=mock_client)

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == settings.llm_model
    assert call_kwargs["config"].system_instruction == (
        "You are a helpful assistant. Answer the user's question using only the context provided. "
        "If the context does not contain enough information to answer, say so clearly. "
        "Do not use knowledge outside of the provided context."
    )
