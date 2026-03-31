from functools import lru_cache
from typing import Literal, Union
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "chatbot-api"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    secret_key: str = "changeme"

    # LLM
    llm_provider: Literal["openai", "anthropic", "gemini"] = "gemini"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    anthropic_api_key: str = ""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_embedding_model: str = "models/text-embedding-004"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Vector Store
    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    chroma_persist_dir: str = "./data/chroma"

    # Meta MongoDB (config de la propia API)
    meta_mongodb_uri: str = "mongodb://localhost:27017"
    meta_mongodb_db: str = "chatbot_meta"

    # RAG
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 100
    rag_top_k: int = 5
    rag_score_threshold: float = 0.35

    # Schema introspection
    schema_sample_docs: int = 5
    schema_max_depth: int = 3
    schema_allowed_collections: list[str] = [
        "events", "attendees", "speakers", "agendas",
        "users", "members", "highlights", "news", "posters",
    ]
    # Colecciones específicas por plataforma (sobreescriben la lista general)
    gencampus_allowed_collections: list[str] = [
        "events", "activities", "courseattendees", "transcript_segments", "modules", "organizations",
        "organizationusers", "quizzes", "UserAttemptsQuiz", "users", "host",
    ]

    # Platform DB connections (para seed)
    acho_mongo_uri: str = ""
    acho_mongo_db: str = "achoDev"

    gencampus_mongo_uri: str = ""
    gencampus_mongo_db: str = "gencampus"

    genlive_mongo_uri: str = ""
    genlive_mongo_db: str = "live-events-v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
