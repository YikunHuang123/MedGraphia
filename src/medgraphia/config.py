"""
Central configuration module.  All settings are loaded from the environment
(or a .env file) via Pydantic Settings.  Import `settings` everywhere instead
of reading env vars directly.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

# Force load .env and override any existing system environment variables
load_dotenv(override=True)

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
    # llm_provider: Literal["deepseek", "openai", "anthropic", "gemini", "groq", "ollama", "local"] = "ollama"
    llm_provider: Literal["deepseek", "openai", "anthropic", "gemini", "groq", "ollama", "local"] = "groq"
    llm_model: str = "llama-3.1-8b-instant"
    # llm_model: str = "qwen2.5:3b"
    llm_base_url: str = ""
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1

    # Task-specific LLM overrides
    rewriter_llm_provider: str = "openai"
    rewriter_llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    rewriter_llm_api_key: SecretStr = SecretStr("")
    rewriter_llm_base_url: str = ""
    
    extractor_llm_provider: str = "openai"
    extractor_llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    extractor_llm_api_key: SecretStr = SecretStr("")
    extractor_llm_base_url: str = ""
    
    summarizer_llm_provider: str = "groq"
    summarizer_llm_model: str = "llama-3.1-8b-instant"
    summarizer_llm_api_key: SecretStr = SecretStr("")
    summarizer_llm_base_url: str = ""

    deepseek_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")

    # ------------------------------------------------------------------
    # LLM Router — per-tier model configuration (Phase 7)
    # ------------------------------------------------------------------
    # Each tier maps to a (provider, model) pair.  Falls back to llm_provider /
    # llm_model when not explicitly set.
    # llm_small_provider: str = "ollama"        # e.g. "ollama"
    # llm_small_model: str = "qwen2.5:3b"           # e.g. "qwen2.5:7b"
    # llm_medium_provider: str = "ollama"       # e.g. "deepseek"
    # llm_medium_model: str = "qwen2.5:3b"          # e.g. "deepseek-chat"
    # llm_large_provider: str = "ollama"        # e.g. "openai"
    # llm_large_model: str = "qwen2.5:3b"           # e.g. "gpt-4o"
    llm_small_provider: str = "deepseek"  # e.g. "ollama"
    llm_small_model: str = "deepseek-chat"           # e.g. "qwen2.5:7b"
    llm_medium_provider: str = "deepseek"       # e.g. "deepseek"
    llm_medium_model: str = "deepseek-chat"          # e.g. "deepseek-chat"
    llm_large_provider: str = "deepseek"        # e.g. "openai"
    llm_large_model: str = "deepseek-chat"           # e.g. "gpt-4o"

    # Model used for community summary generation; defaults to llm_model if empty
    community_summary_llm: str = ""
    community_min_size: int = 3
    community_resolution: float = 1.0

    # ------------------------------------------------------------------
    # Safety guardrails
    # ------------------------------------------------------------------
    guardrails_enabled: bool = True
    llama_guard_provider: str = "ollama"
    llama_guard_model: str = "llama-guard3:1b"
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
    pubmed_max_results: int = 20000
    pubmed_email: str = "user@example.com"
    pubmed_api_key: str = ""
    drug_label_limit: int = 30

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------
    reranker_threshold: float = 0.0  # Minimum score for a passage to be considered relevant

    ner_gliner_model: str = "urchade/gliner_mediumv2.1"
    ner_gliner_threshold: float = 0.30  # lowered further
    ner_bert_en_model: str = "d4data/biomedical-ner-all"
    ner_bert_zh_model: str = "iioSnail/bert-base-chinese-medical-ner" # High-quality Chinese medical NER
    ner_bert_de_model: str = ""
    ner_confidence_threshold: float = 0.25

    # ------------------------------------------------------------------
    # Entity Linking (EL)
    # ------------------------------------------------------------------
    el_sapbert_model: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    el_bm25_top_k: int = 50             # BM25 candidate pool size
    el_link_threshold: float = 0.70     # minimum score to accept a CUI match
    el_sapbert_threshold: float = 0.75  # minimum SapBERT cosine to consider confident
    mesh_dir: str = "data/mesh"         # path to MeSH d2024.bin

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
