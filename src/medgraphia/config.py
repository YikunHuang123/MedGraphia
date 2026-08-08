"""
Central configuration module.  All settings are loaded from the environment
(or a .env file) via Pydantic Settings.  Import `settings` everywhere instead
of reading env vars directly.
"""

from __future__ import annotations

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
    # Graph retrieval (Bipartite Entity-Chunk + Personalized PageRank)
    # ------------------------------------------------------------------
    ppr_damping_factor: float = 0.85
    ppr_max_iterations: int = 20
    ppr_top_k_chunks: int = 15

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
    deepseek_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr = SecretStr("")
    fireworks_api_key: SecretStr = SecretStr("")
    together_api_key: SecretStr = SecretStr("")
    siliconflow_api_key: SecretStr = SecretStr("")
    cerebras_api_key: SecretStr = SecretStr("")

    # Global default LLM.
    # Directly used (no per-task override available):
    #   - llm/client.py (LLMClient.from_settings) : provider is always this value
    # Fallback when no task-specific override is set:
    #   - llm/client.py (make_llm_client)  : provider_override or default
    #   - llm/gateway.py (from_settings)   : provider_override or default
    #   - llm/dspy_setup.py                : provider_override or default
    #   - generation/llm_router.py         : each tier's provider/model attr or default
    default_llm_provider: Literal["deepseek", "openai", "anthropic", "gemini", "groq", "ollama", "vllm", "local", "cerebras", "siliconflow", "fireworks_ai"] = "ollama"
    default_llm_model: str = "qwen3:14b"
    llm_base_url: str = "http://localhost:11434"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1

    # vLLM has its own base_url field — llm_base_url is shared with DeepSeek's
    # override fallback, so repurposing it here would silently redirect DeepSeek
    # calls into vLLM's OpenAI-compatible endpoint too.
    vllm_base_url: str = "http://localhost:8010/v1"

    # Per-model vLLM endpoints, keyed by model_name (not tier — the gateway
    # layer doesn't know about tiers). Each engine is its own vLLM process
    # with Sleep Mode enabled; falls back to vllm_base_url when a model_name
    # isn't listed here.
    vllm_small_base_url: str = "http://localhost:8010/v1"
    vllm_medium_base_url: str = "http://localhost:8011/v1"
    # Seconds of inactivity before an idle vLLM engine is put back to sleep.
    vllm_sleep_idle_seconds: int = 120


    # Model used for community summary generation; defaults to default_llm_model if empty
    community_summary_llm: str = ""
    community_min_size: int = 3
    community_resolution: float = 1.0

    # Task-specific LLM overrides
    # rewriter_llm_provider: str = "ollama"
    # rewriter_llm_model: str = "qwen3.5:9b"
    rewriter_llm_provider: str = "cerebras"
    rewriter_llm_model: str = "gpt-oss-120b"
    rewriter_llm_api_key: SecretStr = SecretStr("")
    rewriter_llm_base_url: str = ""

    summarizer_llm_provider: str = "cerebras"
    summarizer_llm_model: str = "gpt-oss-120b"
    summarizer_llm_api_key: SecretStr = SecretStr("")
    summarizer_llm_base_url: str = ""

    translator_llm_provider: str = "cerebras"
    translator_llm_model: str = "gpt-oss-120b"
    translator_llm_api_key: SecretStr = SecretStr("")
    translator_llm_base_url: str = ""

    # Synthetic data generation (Teacher model)
    synthetic_data_llm_provider: str = "deepseek"
    synthetic_data_llm_model: str = "deepseek-v4-pro"
    synthetic_data_llm_api_key: SecretStr = SecretStr("")
    synthetic_data_llm_base_url: str = ""

    # Judge LLM (used for evaluations, metrics, and quality control)
    judge_llm_provider: str = "deepseek"
    judge_llm_model: str = "deepseek-v4-pro"
    judge_llm_api_key: SecretStr = SecretStr("")
    judge_llm_base_url: str = ""


    # ------------------------------------------------------------------
    # LLM Router — per-tier model configuration
    # ------------------------------------------------------------------
    # Each tier maps to a (provider, model) pair.  Falls back to default_llm_provider /
    # default_llm_model when not explicitly set.
    # llm_small_provider: str = "ollama"        # e.g. "ollama"
    # llm_small_model: str = "Qwen3.5:9b"           # e.g. "qwen2.5:7b"
    # llm_medium_provider: str = "ollama"       # e.g. "deepseek"
    # llm_medium_model: str = "Qwen3.5:9b"          # e.g. "deepseek-chat"
    # llm_large_provider: str = "ollama"        # e.g. "openai"
    # llm_large_model: str = "qwen3:14b"           # e.g. "gpt-4o"
    #
    # Optional: run SMALL/MEDIUM locally on vLLM with Sleep Mode instead
    # (see llm/vllm_sleep_manager.py) — requires the vLLM engines to already
    # be running with --enable-sleep-mode on the ports below before startup:
    #   llm_small_provider: str = "vllm"
    #   llm_small_model: str = "Qwen/Qwen2.5-0.5B-Instruct"    # vllm_small_base_url, default port 8010
    #   llm_medium_provider: str = "vllm"
    #   llm_medium_model: str = "Qwen/Qwen2.5-3B-Instruct"     # vllm_medium_base_url, default port 8011
    llm_small_provider: str = "fireworks"
    llm_small_model: str = "accounts/fireworks/models/gpt-oss-20b"
    llm_medium_provider: str = "fireworks"
    llm_medium_model: str = "accounts/fireworks/models/deepseek-v4-flash"
    llm_large_provider: str = "fireworks"
    llm_large_model: str = "accounts/fireworks/models/qwen3p7-plus"

    # ------------------------------------------------------------------
    # Safety guardrails
    # ------------------------------------------------------------------
    guardrails_enabled: bool = True
    # llama_guard_provider: str = "ollama"
    # llama_guard_model: str = "llama-guard3:1b"
    llama_guard_provider: str = "openrouter"
    llama_guard_model: str = "meta-llama/llama-guard-4-12b"
    ragas_faithfulness_threshold: float = 0.75

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    auth_strategy: Literal["none", "apikey", "oidc"] = "apikey"
    admin_bootstrap_key: SecretStr = SecretStr("change-me")

    # Daily request caps on /chat and /chat/stream (Redis-backed; no-op if Redis
    # is unreachable). Admin-role keys are exempt. Protects LLM spend on public demos.
    # rate_limit_ip_daily keys on the UI's X-Client-ID guest cookie when present
    # (falls back to raw IP for direct API callers) — see api/rate_limit.py.
    rate_limit_enabled: bool = True
    rate_limit_ip_daily: int = 4
    rate_limit_global_daily: int = 80

    # Comma-separated allowed CORS origins, e.g. "https://your-domain.com".
    # "*" (default) allows any origin — fine for local dev, tighten for public deployments.
    cors_allowed_origins: str = "*"

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
    pubmed_max_results: int = 20000
    pubmed_email: str = "user@example.com"
    pubmed_api_key: str = ""
    drug_label_limit: int = 30

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------
    # reranker_provider: str = "siliconflow"
    # reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_provider: str = "fireworks"
    reranker_model: str = "accounts/fireworks/models/qwen3-reranker-8b"
    reranker_api_url: str = ""
    reranker_api_key: SecretStr = SecretStr("")

    reranker_threshold: float = 0.25  # Minimum score for a passage to be considered relevant
    reranker_fallback_top_n: int = 2  # Kept even below threshold so context is never empty
    reranker_noise_floor: float = 0.05  # Below this, fallback content is noise — skip generation entirely

    # ------------------------------------------------------------------
    # Multilingual retrieval
    # ------------------------------------------------------------------
    multilingual_retrieval_enabled: bool = True
    multilingual_per_lang_quota: int = 12

    # ------------------------------------------------------------------
    # Per-user QA memory (long-term conversational memory)
    # ------------------------------------------------------------------
    qa_memory_limit: int = 5  # max QA memories injected into generation context per turn
    qa_memory_half_life_days: float = 30.0
    qa_memory_max_per_entity: int = 20  # capacity eviction bound per (user, entity) pair
    qa_memory_recent_turns: int = 3  # raw messages always included alongside retrieved memories

    # ------------------------------------------------------------------
    # Query-time knowledge graph completion
    # ------------------------------------------------------------------
    gap_completion_enabled: bool = True
    gap_completion_max_tool_calls: int = 2  # per generation request
    gap_completion_pubmed_limit: int = 4  # abstracts fetched per tool call
    single_entity_gap_completion_enabled: bool = True  # live fetch when retrieval finds genuinely nothing (no_evidence)


    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------
    ner_gliner_model: str = "Ihor/gliner-biomed-large-v1.0"
    ner_gliner_threshold: float = 0.30  # lowered further
    ner_bert_en_model: str = "d4data/biomedical-ner-all"
    ner_bert_zh_model: str = "Adapting/bert-base-chinese-finetuned-NER-biomedical"
    ner_bert_de_model: str = "BachelorThesis/GerMedBERT_NER_V01_BRONCO_CARDIO"
    ner_confidence_threshold: float = 0.25

    # ------------------------------------------------------------------
    # Entity Linking 
    # ------------------------------------------------------------------
    el_sapbert_model: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    el_bm25_top_k: int = 50  # BM25 candidate pool size
    el_link_threshold: float = 0.70  # minimum score to accept a CUI match
    el_sapbert_threshold: float = 0.75  # minimum SapBERT cosine to consider confident
    mesh_dir: str = "data/mesh"  # path to MeSH d2024.bin
    el_translation_file: str = ""  # TSV: <CUI>\t<lang>\t<label>, e.g. data/mesh/mesh_translations.tsv

    # ------------------------------------------------------------------
    # Redis — optional cache + task-queue broker
    # ------------------------------------------------------------------
    # Leave unset (or empty) to run without Redis.
    # When set, enables:
    #   • Arq task queue for durable build-pipeline execution
    # redis_url: str | None = None
    redis_url: str | None = "redis://localhost:6379/0"

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
    def _set_community_summary_llm(self) -> Settings:
        if not self.community_summary_llm:
            self.community_summary_llm = self.default_llm_model
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
