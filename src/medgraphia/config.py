"""
Central configuration module.  All settings are loaded from the environment
(or a .env file) via Pydantic Settings.  Import `settings` everywhere instead
of reading env vars directly.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Storage Backend
    # ------------------------------------------------------------------
    storage_backend: Literal["local", "s3"] = "local"

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("neo4j")
    neo4j_database: str = "neo4j"
    neo4j_page_cache: str = "1G"
    neo4j_heap_initial: str = "512M"
    neo4j_heap_max: str = "1G"

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    vector_store: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_chunks: str = "medgraphia_chunks"
    qdrant_collection_entities: str = "medgraphia_entities"

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    embedding_provider: Literal["huggingface", "ollama", "openai"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434"
    embedding_batch_size: int = 32

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_provider: Literal["deepseek", "openai", "anthropic", "ollama", "local"] = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_base_url: str = ""
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1

    deepseek_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: SecretStr = SecretStr("")

    # Model used for community summary generation; defaults to llm_model if empty
    community_summary_llm: str = ""

    # ------------------------------------------------------------------
    # Safety guardrails
    # ------------------------------------------------------------------
    guardrails_enabled: bool = False
    llama_guard_model: str = "meta-llama/Llama-Guard-4-8B"
    ragas_faithfulness_threshold: float = 0.75

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    auth_strategy: Literal["none", "apikey", "oidc"] = "apikey"
    admin_bootstrap_key: SecretStr = SecretStr("change-me")

    keycloak_server_url: str = ""
    keycloak_realm: str = "medgraphia"
    keycloak_client_id: str = "medgraphia-api"
    keycloak_client_secret: SecretStr = SecretStr("")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    tracing_enabled: bool = False
    metrics_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")

    # ------------------------------------------------------------------
    # S3 Storage (MinIO/AWS)
    # ------------------------------------------------------------------
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: SecretStr = SecretStr("minioadmin")
    minio_bucket_raw: str = "medgraphia-raw"
    minio_bucket_parsed: str = "medgraphia-parsed"
    minio_secure: bool = False

    # ------------------------------------------------------------------
    # Pipeline / data ingestion
    # ------------------------------------------------------------------
    default_domain: str = "t2dm"
    pubmed_max_results: int = 200
    pubmed_email: str = "user@example.com"
    pubmed_api_key: str = ""
    drug_label_limit: int = 30

    # ------------------------------------------------------------------
    # Privacy / compliance
    # ------------------------------------------------------------------
    pii_deidentify: bool = False

    # ------------------------------------------------------------------
    # GraphRAG framework
    # ------------------------------------------------------------------
    graphrag_framework: Literal["lightrag"] = "lightrag"

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8058
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Derived / validated fields
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _set_community_summary_llm(self) -> "Settings":
        if not self.community_summary_llm:
            self.community_summary_llm = self.llm_model
        return self

    @property
    def use_s3_storage(self) -> bool:
        return self.storage_backend == "s3"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


# Module-level convenience alias — importable as `from medgraphia.config import settings`
settings: Settings = Field(default_factory=get_settings)


# Override the module-level alias after import so it's actually the instance
import sys as _sys
_sys.modules[__name__].settings = get_settings()  # type: ignore[attr-defined]
