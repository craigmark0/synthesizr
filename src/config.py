from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: Annotated[str, Field(min_length=1)]
    database_url: Annotated[str, Field(min_length=1)]
    similarity_threshold: float = 0.7
    fallback_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "gemini-embedding-2-preview"
    llm_model: str = "gemini-2.5-flash"


settings = Settings()
