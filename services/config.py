from __future__ import annotations

# Configuration loading and normalization.
# Settings are built from YAML defaults plus environment overrides so the rest
# of the runtime can depend on one typed object instead of scattered env reads.

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "defaults.yaml"


class LLMSettings(BaseModel):
    provider: Literal["openai", "deepseek", "openai_compatible"] = "openai_compatible"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2200, ge=64, le=8192)
    timeout: float = Field(default=60.0, gt=0)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class EmbeddingSettings(BaseModel):
    provider: Literal["openai", "deepseek", "openai_compatible"] = "openai_compatible"
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = Field(default=1024, ge=64, le=4096)
    batch_size: int = Field(default=32, ge=1, le=256)
    timeout: float = Field(default=60.0, gt=0)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class VectorDBSettings(BaseModel):
    provider: Literal["qdrant", "milvus", "weaviate", "chroma", "pgvector", "pinecone", "memory"] = "qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "agentops_knowledge"
    qdrant_local_path: str = ".qdrant"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_db_name: str = "default"
    milvus_collection: str = "agentops_knowledge"
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str | None = None
    weaviate_collection: str = "AgentOpsKnowledge"
    weaviate_grpc_host: str | None = None
    weaviate_grpc_port: int | None = Field(default=None, ge=1, le=65535)
    weaviate_grpc_secure: bool | None = None
    chroma_host: str | None = None
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = False
    chroma_headers: dict[str, str] = Field(default_factory=dict)
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"
    chroma_path: str = ".chroma"
    chroma_collection: str = "agentops_knowledge"
    pgvector_dsn: str | None = None
    pgvector_schema: str = Field(default="public", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    pgvector_table: str = Field(default="agentops_knowledge", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    pinecone_api_key: str | None = None
    pinecone_host: str | None = None
    pinecone_index: str = "agentops-knowledge"
    pinecone_namespace: str = "agentops"
    pinecone_cloud: Literal["aws", "gcp", "azure"] = "aws"
    pinecone_region: str = "us-east-1"
    timeout: float = Field(default=30.0, gt=0)


class RAGSettings(BaseModel):
    enabled: bool = True
    source_dir: str = "knowledge_base"
    top_k: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1200)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class WebGroundingSettings(BaseModel):
    mode: Literal["auto", "tavily_jina", "exa"] = "auto"
    search_provider: Literal["tavily", "exa"] | None = None
    reader_provider: Literal["jina", "exa"] | None = None
    enable_web_search: bool = True
    enable_vector_rag: bool = True
    web_search_top_k: int = Field(default=5, ge=1, le=20)
    web_reader_top_k: int = Field(default=3, ge=1, le=20)
    evidence_merge_strategy: Literal["score", "source_priority"] = "score"


class TavilySettings(BaseModel):
    api_key: str | None = None
    search_depth: Literal["basic", "advanced"] = "basic"
    topic: str = "general"
    max_results: int = Field(default=5, ge=1, le=20)
    include_raw_content: bool = False


class JinaReaderSettings(BaseModel):
    api_key: str | None = None
    base_url: str = "https://r.jina.ai/"
    timeout: float = Field(default=30.0, gt=0)
    use_json: bool = False
    bypass_cache: bool = False


class ExaSettings(BaseModel):
    api_key: str | None = None
    base_url: str = "https://api.exa.ai"
    search_type: str = "neural"
    max_results: int = Field(default=5, ge=1, le=20)
    use_contents: bool = True


class GovernanceSettings(BaseModel):
    overall_risk_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    recommendation_confidence_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    evidence_completeness_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    contradiction_severity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    unresolved_questions_threshold: int = Field(default=2, ge=0, le=20)
    high_stakes_task_categories: list[str] = Field(default_factory=lambda: ["architecture", "security", "finance"])
    manual_approval_policy_by_task_type: dict[str, str] = Field(default_factory=dict)


class ContextCompactionSettings(BaseModel):
    enabled: bool = True
    max_estimated_tokens: int = Field(default=24000, ge=1000)
    light_pressure_ratio: float = Field(default=0.60, ge=0.0, le=1.0)
    high_pressure_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    critical_pressure_ratio: float = Field(default=0.85, ge=0.0, le=1.0)
    recent_user_messages: int = Field(default=2, ge=1, le=10)
    recent_messages: int = Field(default=8, ge=1, le=50)
    trace_tail_size: int = Field(default=25, ge=1, le=500)
    semantic_dedupe_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    tool_offload_enabled: bool = True


class ReportingSettings(BaseModel):
    # Controls deterministic conclusion depth independently from provider token
    # limits. Detailed remains the default, while deployments can choose a
    # smaller surface for cost- or latency-sensitive interfaces.
    detail_level: Literal["compact", "standard", "detailed"] = "detailed"
    max_findings_in_conclusion: int = Field(default=5, ge=1, le=8)
    max_evidence_in_conclusion: int = Field(default=6, ge=1, le=12)
    max_risks_in_conclusion: int = Field(default=5, ge=1, le=10)
    include_next_actions: bool = True


class Settings(BaseModel):
    app_name: str = "LangGraph AgentOps Studio"
    log_level: str = "INFO"
    output_root: str = "runs"
    checkpoint_mode: str = "memory"
    max_search_results: int = Field(default=5, ge=1, le=20)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_revisions: int = Field(default=2, ge=0, le=5)
    risk_threshold_for_human_review: float = Field(default=0.68, ge=0.0, le=1.0)
    enable_langsmith: bool = False
    langsmith_project: str = "langgraph-agentops-studio"
    default_reviewer: str = "governance-board"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    web_grounding: WebGroundingSettings = Field(default_factory=WebGroundingSettings)
    tavily: TavilySettings = Field(default_factory=TavilySettings)
    jina: JinaReaderSettings = Field(default_factory=JinaReaderSettings)
    exa: ExaSettings = Field(default_factory=ExaSettings)
    governance: GovernanceSettings = Field(default_factory=GovernanceSettings)
    context_compaction: ContextCompactionSettings = Field(default_factory=ContextCompactionSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)


def _load_yaml_defaults() -> dict[str, Any]:
    if not DEFAULT_CONFIG_PATH.exists():
        return {}
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _json_env(name: str, default: dict[str, Any]) -> dict[str, Any]:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return default
    return parsed if isinstance(parsed, dict) else default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _optional_float_env(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _optional_int_env(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _optional_bool_env(name: str, default: bool | None) -> bool | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional_str_env(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _list_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _dict_env(name: str, default: dict[str, str]) -> dict[str, str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return default
    if not isinstance(parsed, dict):
        return default
    return {str(key): str(value) for key, value in parsed.items()}


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    # Cache the resolved settings because runtime construction is process-wide
    # and repeated reloads would only duplicate env/default parsing work.
    load_dotenv(ROOT_DIR / ".env")
    defaults = _load_yaml_defaults()

    llm_defaults = defaults.get("llm", {})
    embedding_defaults = defaults.get("embedding", {})
    vector_defaults = defaults.get("vector_db", {})
    rag_defaults = defaults.get("rag", {})
    web_defaults = defaults.get("web_grounding", {})
    tavily_defaults = defaults.get("tavily", {})
    jina_defaults = defaults.get("jina", {})
    exa_defaults = defaults.get("exa", {})
    governance_defaults = defaults.get("governance", {})
    context_defaults = defaults.get("context_compaction", {})
    reporting_defaults = defaults.get("reporting", {})

    return Settings.model_validate(
        {
            "app_name": os.getenv("APP_NAME", defaults.get("project_name", "LangGraph AgentOps Studio")),
            "log_level": os.getenv("LOG_LEVEL", defaults.get("log_level", "INFO")),
            "output_root": os.getenv("OUTPUT_ROOT", defaults.get("output_root", "runs")),
            "checkpoint_mode": os.getenv("CHECKPOINT_MODE", defaults.get("checkpoint_mode", "memory")),
            "max_search_results": _int_env("MAX_SEARCH_RESULTS", defaults.get("max_search_results", 5)),
            "max_retries": _int_env("MAX_RETRIES", defaults.get("max_retries", 2)),
            "max_revisions": _int_env("MAX_REVISIONS", defaults.get("max_revisions", 2)),
            "risk_threshold_for_human_review": _float_env(
                "RISK_THRESHOLD_FOR_HUMAN_REVIEW",
                defaults.get("risk_threshold_for_human_review", 0.68),
            ),
            "enable_langsmith": _to_bool(os.getenv("ENABLE_LANGSMITH"), defaults.get("enable_langsmith", False)),
            "langsmith_project": os.getenv("LANGSMITH_PROJECT", defaults.get("langsmith_project", "langgraph-agentops-studio")),
            "default_reviewer": os.getenv("DEFAULT_REVIEWER", defaults.get("default_reviewer", "governance-board")),
            "llm": {
                "provider": os.getenv("LLM_PROVIDER", llm_defaults.get("provider", "openai_compatible")),
                "model": os.getenv("LLM_MODEL", llm_defaults.get("model", "gpt-4o-mini")),
                "api_key": _optional_str_env("LLM_API_KEY", llm_defaults.get("api_key")),
                "base_url": _optional_str_env("LLM_BASE_URL", llm_defaults.get("base_url")),
                "temperature": _float_env("LLM_TEMPERATURE", llm_defaults.get("temperature", 0.2)),
                "max_tokens": _int_env("LLM_MAX_TOKENS", llm_defaults.get("max_tokens", 2200)),
                "timeout": _float_env("LLM_TIMEOUT", llm_defaults.get("timeout", 60.0)),
                "extra_headers": _json_env("LLM_EXTRA_HEADERS_JSON", llm_defaults.get("extra_headers", {})),
                "extra_kwargs": _json_env("LLM_EXTRA_KWARGS_JSON", llm_defaults.get("extra_kwargs", {})),
            },
            "embedding": {
                "provider": os.getenv("EMBEDDING_PROVIDER", embedding_defaults.get("provider", "openai_compatible")),
                "model": os.getenv("EMBEDDING_MODEL", embedding_defaults.get("model", "text-embedding-3-small")),
                "api_key": _optional_str_env("EMBEDDING_API_KEY", embedding_defaults.get("api_key")),
                "base_url": _optional_str_env("EMBEDDING_BASE_URL", embedding_defaults.get("base_url")),
                "dimensions": _int_env("EMBEDDING_DIMENSIONS", embedding_defaults.get("dimensions", 1024)),
                "batch_size": _int_env("EMBEDDING_BATCH_SIZE", embedding_defaults.get("batch_size", 32)),
                "timeout": _float_env("EMBEDDING_TIMEOUT", embedding_defaults.get("timeout", 60.0)),
                "extra_headers": _json_env(
                    "EMBEDDING_EXTRA_HEADERS_JSON",
                    embedding_defaults.get("extra_headers", {}),
                ),
                "extra_kwargs": _json_env("EMBEDDING_EXTRA_KWARGS_JSON", embedding_defaults.get("extra_kwargs", {})),
            },
            "vector_db": {
                "provider": os.getenv("VECTOR_DB_PROVIDER", vector_defaults.get("provider", "qdrant")),
                "qdrant_url": _optional_str_env("QDRANT_URL", vector_defaults.get("qdrant_url")),
                "qdrant_api_key": _optional_str_env("QDRANT_API_KEY", vector_defaults.get("qdrant_api_key")),
                "qdrant_collection": os.getenv(
                    "QDRANT_COLLECTION",
                    vector_defaults.get("qdrant_collection", "agentops_knowledge"),
                ),
                "qdrant_local_path": os.getenv(
                    "QDRANT_LOCAL_PATH",
                    vector_defaults.get("qdrant_local_path", ".qdrant"),
                ),
                "milvus_uri": os.getenv(
                    "MILVUS_URI",
                    vector_defaults.get("milvus_uri", "http://localhost:19530"),
                ),
                "milvus_token": _optional_str_env("MILVUS_TOKEN", vector_defaults.get("milvus_token")),
                "milvus_db_name": os.getenv(
                    "MILVUS_DB_NAME",
                    vector_defaults.get("milvus_db_name", "default"),
                ),
                "milvus_collection": os.getenv(
                    "MILVUS_COLLECTION",
                    vector_defaults.get("milvus_collection", "agentops_knowledge"),
                ),
                "weaviate_url": os.getenv(
                    "WEAVIATE_URL",
                    vector_defaults.get("weaviate_url", "http://localhost:8080"),
                ),
                "weaviate_api_key": _optional_str_env(
                    "WEAVIATE_API_KEY",
                    vector_defaults.get("weaviate_api_key"),
                ),
                "weaviate_collection": os.getenv(
                    "WEAVIATE_COLLECTION",
                    vector_defaults.get("weaviate_collection", "AgentOpsKnowledge"),
                ),
                "weaviate_grpc_host": _optional_str_env(
                    "WEAVIATE_GRPC_HOST",
                    vector_defaults.get("weaviate_grpc_host"),
                ),
                "weaviate_grpc_port": _optional_int_env(
                    "WEAVIATE_GRPC_PORT",
                    vector_defaults.get("weaviate_grpc_port"),
                ),
                "weaviate_grpc_secure": _optional_bool_env(
                    "WEAVIATE_GRPC_SECURE",
                    vector_defaults.get("weaviate_grpc_secure"),
                ),
                "chroma_host": _optional_str_env("CHROMA_HOST", vector_defaults.get("chroma_host")),
                "chroma_port": _int_env("CHROMA_PORT", vector_defaults.get("chroma_port", 8000)),
                "chroma_ssl": _to_bool(os.getenv("CHROMA_SSL"), vector_defaults.get("chroma_ssl", False)),
                "chroma_headers": _dict_env(
                    "CHROMA_HEADERS_JSON",
                    vector_defaults.get("chroma_headers", {}),
                ),
                "chroma_tenant": os.getenv(
                    "CHROMA_TENANT",
                    vector_defaults.get("chroma_tenant", "default_tenant"),
                ),
                "chroma_database": os.getenv(
                    "CHROMA_DATABASE",
                    vector_defaults.get("chroma_database", "default_database"),
                ),
                "chroma_path": os.getenv("CHROMA_PATH", vector_defaults.get("chroma_path", ".chroma")),
                "chroma_collection": os.getenv(
                    "CHROMA_COLLECTION",
                    vector_defaults.get("chroma_collection", "agentops_knowledge"),
                ),
                "pgvector_dsn": _optional_str_env("PGVECTOR_DSN", vector_defaults.get("pgvector_dsn")),
                "pgvector_schema": os.getenv(
                    "PGVECTOR_SCHEMA",
                    vector_defaults.get("pgvector_schema", "public"),
                ),
                "pgvector_table": os.getenv(
                    "PGVECTOR_TABLE",
                    vector_defaults.get("pgvector_table", "agentops_knowledge"),
                ),
                "pinecone_api_key": _optional_str_env(
                    "PINECONE_API_KEY",
                    vector_defaults.get("pinecone_api_key"),
                ),
                "pinecone_host": _optional_str_env("PINECONE_HOST", vector_defaults.get("pinecone_host")),
                "pinecone_index": os.getenv(
                    "PINECONE_INDEX",
                    vector_defaults.get("pinecone_index", "agentops-knowledge"),
                ),
                "pinecone_namespace": os.getenv(
                    "PINECONE_NAMESPACE",
                    vector_defaults.get("pinecone_namespace", "agentops"),
                ),
                "pinecone_cloud": os.getenv(
                    "PINECONE_CLOUD",
                    vector_defaults.get("pinecone_cloud", "aws"),
                ),
                "pinecone_region": os.getenv(
                    "PINECONE_REGION",
                    vector_defaults.get("pinecone_region", "us-east-1"),
                ),
                "timeout": _float_env("VECTOR_DB_TIMEOUT", vector_defaults.get("timeout", 30.0)),
            },
            "rag": {
                "enabled": _to_bool(os.getenv("RAG_ENABLED"), rag_defaults.get("enabled", True)),
                "source_dir": os.getenv("RAG_SOURCE_DIR", rag_defaults.get("source_dir", "knowledge_base")),
                "top_k": _int_env("RAG_TOP_K", rag_defaults.get("top_k", 4)),
                "chunk_size": _int_env("RAG_CHUNK_SIZE", rag_defaults.get("chunk_size", 900)),
                "chunk_overlap": _int_env("RAG_CHUNK_OVERLAP", rag_defaults.get("chunk_overlap", 120)),
                "score_threshold": _optional_float_env("RAG_SCORE_THRESHOLD", rag_defaults.get("score_threshold")),
            },
            "web_grounding": {
                "mode": os.getenv("WEB_SEARCH_MODE", web_defaults.get("mode", "auto")),
                "search_provider": _optional_str_env("WEB_SEARCH_PROVIDER", web_defaults.get("search_provider")),
                "reader_provider": _optional_str_env("WEB_READER_PROVIDER", web_defaults.get("reader_provider")),
                "enable_web_search": _to_bool(
                    os.getenv("ENABLE_WEB_SEARCH"),
                    web_defaults.get("enable_web_search", True),
                ),
                "enable_vector_rag": _to_bool(
                    os.getenv("ENABLE_VECTOR_RAG"),
                    web_defaults.get("enable_vector_rag", True),
                ),
                "web_search_top_k": _int_env("WEB_SEARCH_TOP_K", web_defaults.get("web_search_top_k", 5)),
                "web_reader_top_k": _int_env("WEB_READER_TOP_K", web_defaults.get("web_reader_top_k", 3)),
                "evidence_merge_strategy": os.getenv(
                    "EVIDENCE_MERGE_STRATEGY",
                    web_defaults.get("evidence_merge_strategy", "score"),
                ),
            },
            "tavily": {
                "api_key": _optional_str_env("TAVILY_API_KEY", tavily_defaults.get("api_key")),
                "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", tavily_defaults.get("search_depth", "basic")),
                "topic": os.getenv("TAVILY_TOPIC", tavily_defaults.get("topic", "general")),
                "max_results": _int_env("TAVILY_MAX_RESULTS", tavily_defaults.get("max_results", 5)),
                "include_raw_content": _to_bool(
                    os.getenv("TAVILY_INCLUDE_RAW_CONTENT"),
                    tavily_defaults.get("include_raw_content", False),
                ),
            },
            "jina": {
                "api_key": _optional_str_env("JINA_API_KEY", jina_defaults.get("api_key")),
                "base_url": os.getenv("JINA_READER_BASE_URL", jina_defaults.get("base_url", "https://r.jina.ai/")),
                "timeout": _float_env("JINA_READER_TIMEOUT", jina_defaults.get("timeout", 30.0)),
                "use_json": _to_bool(os.getenv("JINA_READER_JSON"), jina_defaults.get("use_json", False)),
                "bypass_cache": _to_bool(
                    os.getenv("JINA_READER_BYPASS_CACHE"),
                    jina_defaults.get("bypass_cache", False),
                ),
            },
            "exa": {
                "api_key": _optional_str_env("EXA_API_KEY", exa_defaults.get("api_key")),
                "base_url": os.getenv("EXA_BASE_URL", exa_defaults.get("base_url", "https://api.exa.ai")),
                "search_type": os.getenv("EXA_SEARCH_TYPE", exa_defaults.get("search_type", "neural")),
                "max_results": _int_env("EXA_MAX_RESULTS", exa_defaults.get("max_results", 5)),
                "use_contents": _to_bool(os.getenv("EXA_USE_CONTENTS"), exa_defaults.get("use_contents", True)),
            },
            "governance": {
                "overall_risk_threshold": _float_env(
                    "GOVERNANCE_OVERALL_RISK_THRESHOLD",
                    governance_defaults.get("overall_risk_threshold", 0.65),
                ),
                "recommendation_confidence_threshold": _float_env(
                    "GOVERNANCE_RECOMMENDATION_CONFIDENCE_THRESHOLD",
                    governance_defaults.get("recommendation_confidence_threshold", 0.62),
                ),
                "evidence_completeness_threshold": _float_env(
                    "GOVERNANCE_EVIDENCE_COMPLETENESS_THRESHOLD",
                    governance_defaults.get("evidence_completeness_threshold", 0.55),
                ),
                "contradiction_severity_threshold": _float_env(
                    "GOVERNANCE_CONTRADICTION_SEVERITY_THRESHOLD",
                    governance_defaults.get("contradiction_severity_threshold", 0.55),
                ),
                "unresolved_questions_threshold": _int_env(
                    "GOVERNANCE_UNRESOLVED_QUESTIONS_THRESHOLD",
                    governance_defaults.get("unresolved_questions_threshold", 2),
                ),
                "high_stakes_task_categories": _list_env(
                    "GOVERNANCE_HIGH_STAKES_TASK_CATEGORIES",
                    governance_defaults.get("high_stakes_task_categories", ["architecture", "security", "finance"]),
                ),
                "manual_approval_policy_by_task_type": _dict_env(
                    "GOVERNANCE_MANUAL_APPROVAL_POLICY_BY_TASK_TYPE_JSON",
                    governance_defaults.get("manual_approval_policy_by_task_type", {}),
                ),
            },
            "context_compaction": {
                "enabled": _to_bool(
                    os.getenv("CONTEXT_COMPACTION_ENABLED"),
                    context_defaults.get("enabled", True),
                ),
                "max_estimated_tokens": _int_env(
                    "CONTEXT_MAX_ESTIMATED_TOKENS",
                    context_defaults.get("max_estimated_tokens", 24000),
                ),
                "light_pressure_ratio": _float_env(
                    "CONTEXT_LIGHT_PRESSURE_RATIO",
                    context_defaults.get("light_pressure_ratio", 0.60),
                ),
                "high_pressure_ratio": _float_env(
                    "CONTEXT_HIGH_PRESSURE_RATIO",
                    context_defaults.get("high_pressure_ratio", 0.75),
                ),
                "critical_pressure_ratio": _float_env(
                    "CONTEXT_CRITICAL_PRESSURE_RATIO",
                    context_defaults.get("critical_pressure_ratio", 0.85),
                ),
                "recent_user_messages": _int_env(
                    "CONTEXT_RECENT_USER_MESSAGES",
                    context_defaults.get("recent_user_messages", 2),
                ),
                "recent_messages": _int_env(
                    "CONTEXT_RECENT_MESSAGES",
                    context_defaults.get("recent_messages", 8),
                ),
                "trace_tail_size": _int_env(
                    "CONTEXT_TRACE_TAIL_SIZE",
                    context_defaults.get("trace_tail_size", 25),
                ),
                "semantic_dedupe_threshold": _float_env(
                    "CONTEXT_SEMANTIC_DEDUPE_THRESHOLD",
                    context_defaults.get("semantic_dedupe_threshold", 0.90),
                ),
                "tool_offload_enabled": _to_bool(
                    os.getenv("CONTEXT_TOOL_OFFLOAD_ENABLED"),
                    context_defaults.get("tool_offload_enabled", True),
                ),
            },
            "reporting": {
                "detail_level": os.getenv(
                    "REPORT_DETAIL_LEVEL",
                    reporting_defaults.get("detail_level", "detailed"),
                ),
                "max_findings_in_conclusion": _int_env(
                    "REPORT_MAX_FINDINGS_IN_CONCLUSION",
                    reporting_defaults.get("max_findings_in_conclusion", 5),
                ),
                "max_evidence_in_conclusion": _int_env(
                    "REPORT_MAX_EVIDENCE_IN_CONCLUSION",
                    reporting_defaults.get("max_evidence_in_conclusion", 6),
                ),
                "max_risks_in_conclusion": _int_env(
                    "REPORT_MAX_RISKS_IN_CONCLUSION",
                    reporting_defaults.get("max_risks_in_conclusion", 5),
                ),
                "include_next_actions": _to_bool(
                    os.getenv("REPORT_INCLUDE_NEXT_ACTIONS"),
                    reporting_defaults.get("include_next_actions", True),
                ),
            },
        }
    )
