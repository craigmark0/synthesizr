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
