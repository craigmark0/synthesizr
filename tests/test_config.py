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
