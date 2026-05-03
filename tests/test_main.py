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


def test_ingest_json_body_empty_text_returns_422(test_client):
    response = test_client.post("/ingest", json={
        "text": "",
        "source": "test.txt",
    })
    assert response.status_code == 422


def test_ingest_upload_non_utf8_file_returns_422(test_client, tmp_path):
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")

    with open(binary_file, "rb") as f:
        response = test_client.post(
            "/ingest/upload",
            files={"file": ("binary.bin", f, "application/octet-stream")},
            data={"source": "binary.bin"},
        )
    assert response.status_code == 422


def test_ingest_upload_metadata_not_json_object_returns_422(test_client, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("some content")

    with open(test_file, "rb") as f:
        response = test_client.post(
            "/ingest/upload",
            files={"file": ("test.txt", f, "text/plain")},
            data={"source": "test.txt", "metadata": '"just a string"'},
        )
    assert response.status_code == 422


def test_ingest_json_body_embed_failure_returns_502(test_client, mock_client):
    mock_client.models.embed_content.side_effect = RuntimeError("API down")

    response = test_client.post("/ingest", json={
        "text": "some text to embed",
        "source": "test.txt",
    })
    assert response.status_code == 502


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

def _make_row(content, source, doc_id, similarity):
    from unittest.mock import MagicMock
    row = MagicMock()
    row.content = content
    row.source = source
    row.document_id = doc_id
    row.similarity = similarity
    return row


def test_query_returns_answer_and_sources(test_client, mock_db, mock_client):
    mock_db.execute.return_value.fetchall.return_value = [
        _make_row("42 is the answer.", "doc.txt", "abc-123", 0.9),
    ]
    mock_client.models.generate_content.return_value.text = "The answer is 42."

    response = test_client.post("/query", json={"question": "What is the answer?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The answer is 42."
    assert data["sources"] == [{"source": "doc.txt", "document_id": "abc-123"}]


def test_query_empty_db_returns_no_context_answer(test_client, mock_db):
    mock_db.execute.return_value.fetchall.return_value = []

    response = test_client.post("/query", json={"question": "What is X?"})

    assert response.status_code == 200
    data = response.json()
    assert "No relevant context" in data["answer"]
    assert data["sources"] == []


def test_query_requires_question_field(test_client):
    response = test_client.post("/query", json={})
    assert response.status_code == 422


def test_query_embed_failure_returns_502(test_client, mock_client):
    mock_client.models.embed_content.side_effect = RuntimeError("API down")

    response = test_client.post("/query", json={"question": "What is X?"})
    assert response.status_code == 502


def test_query_synthesize_failure_returns_502(test_client, mock_db, mock_client):
    mock_db.execute.return_value.fetchall.return_value = [
        _make_row("content", "doc.txt", "abc-123", 0.9),
    ]
    mock_client.models.generate_content.side_effect = RuntimeError("LLM down")

    response = test_client.post("/query", json={"question": "What?"})
    assert response.status_code == 502


def test_query_sources_are_deduplicated(test_client, mock_db, mock_client):
    mock_db.execute.return_value.fetchall.return_value = [
        _make_row("content a", "doc.txt", "doc-id-1", 0.9),
        _make_row("content b", "doc.txt", "doc-id-1", 0.85),   # same document_id → deduplicated
        _make_row("content c", "other.txt", "doc-id-2", 0.8),
    ]
    mock_client.models.generate_content.return_value.text = "Answer."

    response = test_client.post("/query", json={"question": "Q?"})

    assert response.status_code == 200
    assert response.json()["sources"] == [
        {"source": "doc.txt", "document_id": "doc-id-1"},
        {"source": "other.txt", "document_id": "doc-id-2"},
    ]


def test_query_without_snippet_param_omits_content(test_client, mock_db, mock_client):
    mock_db.execute.return_value.fetchall.return_value = [
        _make_row("42 is the answer.", "doc.txt", "abc-123", 0.9),
    ]
    mock_client.models.generate_content.return_value.text = "Answer."

    response = test_client.post("/query", json={"question": "What is the answer?"})

    assert response.status_code == 200
    assert "content" not in response.json()["sources"][0]


def test_query_with_snippet_param_returns_content(test_client, mock_db, mock_client):
    mock_db.execute.return_value.fetchall.return_value = [
        _make_row("42 is the answer.", "doc.txt", "abc-123", 0.9),
    ]
    mock_client.models.generate_content.return_value.text = "Answer."

    response = test_client.post(
        "/query?include_content_snippets=true",
        json={"question": "What is the answer?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "content" in data["sources"][0]
    assert data["sources"][0]["content"] == "42 is the answer."


def test_query_snippet_truncates_to_200_chars(test_client, mock_db, mock_client):
    long_content = "x" * 300
    mock_db.execute.return_value.fetchall.return_value = [
        _make_row(long_content, "doc.txt", "abc-123", 0.9),
    ]
    mock_client.models.generate_content.return_value.text = "Answer."

    response = test_client.post(
        "/query?include_content_snippets=true",
        json={"question": "What?"},
    )

    assert response.status_code == 200
    assert len(response.json()["sources"][0]["content"]) == 200


def test_query_snippet_does_not_truncate_short_content(test_client, mock_db, mock_client):
    short_content = "Short answer."
    mock_db.execute.return_value.fetchall.return_value = [
        _make_row(short_content, "doc.txt", "abc-123", 0.9),
    ]
    mock_client.models.generate_content.return_value.text = "Answer."

    response = test_client.post(
        "/query?include_content_snippets=true",
        json={"question": "What?"},
    )

    assert response.status_code == 200
    assert response.json()["sources"][0]["content"] == short_content
