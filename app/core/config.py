from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Document Research Assistant"
    app_env: str = "development"
    redis_url: str = "redis://localhost:6379/0"
    celery_task_always_eager: bool = False
    default_model_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    vllm_model: str = "Qwen/Qwen3-0.6B"
    vllm_served_model_name: str = "local-qwen"
    vllm_gpu_memory_utilization: float = 0.8
    vllm_max_model_len: int = 4096
    hf_token: str | None = None
    upload_dir: str = "data/uploads"
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "text-embedding-3-small"
    vector_db_dir: str = "data/chroma"
    retrieval_top_k: int = 4
    log_file: str = "data/logs/assistant.log"


@lru_cache
def get_settings() -> Settings:
    return Settings()
