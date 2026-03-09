from __future__ import annotations

import os

from services.config import load_settings


def test_provider_configuration_from_environment() -> None:
    previous = dict(os.environ)
    try:
        os.environ["LLM_PROVIDER"] = "openai_compatible"
        os.environ["LLM_MODEL"] = "custom-llm"
        os.environ["LLM_BASE_URL"] = "https://gateway.example.com/v1"
        os.environ["LLM_API_KEY"] = "llm-key"

        os.environ["EMBEDDING_PROVIDER"] = "openai_compatible"
        os.environ["EMBEDDING_MODEL"] = "custom-embedding"
        os.environ["EMBEDDING_BASE_URL"] = "https://gateway.example.com/v1"
        os.environ["EMBEDDING_API_KEY"] = "emb-key"

        os.environ["WEB_SEARCH_MODE"] = "exa"
        os.environ["WEB_SEARCH_PROVIDER"] = "exa"
        os.environ["WEB_READER_PROVIDER"] = "exa"
        os.environ["ENABLE_WEB_SEARCH"] = "true"
        os.environ["ENABLE_VECTOR_RAG"] = "true"

        load_settings.cache_clear()
        settings = load_settings()

        assert settings.llm.provider == "openai_compatible"
        assert settings.llm.model == "custom-llm"
        assert settings.embedding.provider == "openai_compatible"
        assert settings.embedding.model == "custom-embedding"
        assert settings.web_grounding.mode == "exa"
        assert settings.web_grounding.search_provider == "exa"
        assert settings.web_grounding.reader_provider == "exa"
    finally:
        os.environ.clear()
        os.environ.update(previous)
        load_settings.cache_clear()
