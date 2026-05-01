import os

# Must run before any src imports — pydantic-settings reads env at class instantiation time.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://synthesizr:synthesizr@localhost:5432/synthesizr")

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
