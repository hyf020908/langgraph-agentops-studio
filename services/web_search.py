from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib import request as urllib_request

from schemas.models import SearchResult
from services.config import Settings


class WebSearchConfigError(RuntimeError):
    pass


class BaseWebSearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        ...


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url=url, data=data, headers=headers, method="POST")
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


@dataclass(slots=True)
class TavilySearchProvider:
    api_key: str
    search_depth: str = "basic"
    topic: str = "general"
    include_raw_content: bool = False
    endpoint: str = "https://api.tavily.com/search"
    requester: Any = _post_json
    name: str = "tavily"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "topic": self.topic,
            "search_depth": self.search_depth,
            "include_raw_content": self.include_raw_content,
        }
        response = self.requester(self.endpoint, {"Content-Type": "application/json"}, payload)
        now = datetime.now(UTC).isoformat()
        results: list[SearchResult] = []
        for index, item in enumerate(response.get("results", []), start=1):
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            content = str(item.get("raw_content") or item.get("content") or "")
            snippet = str(item.get("content") or content[:500] or item.get("title") or "")
            results.append(
                SearchResult(
                    source_id=f"TAV-{index:03d}",
                    provider="tavily",
                    source_type="web_search",
                    title=str(item.get("title", "Untitled")),
                    url=url,
                    source=url,
                    domain=_extract_domain(url),
                    snippet=snippet[:500],
                    content=content or None,
                    score=_safe_float(item.get("score"), default=0.6),
                    credibility=min(0.95, max(0.45, _safe_float(item.get("score"), default=0.62))),
                    published_at=str(item.get("published_date", "unknown")),
                    retrieved_at=now,
                    metadata={"search_depth": self.search_depth, "topic": self.topic},
                )
            )
        return results


@dataclass(slots=True)
class ExaSearchProvider:
    api_key: str
    base_url: str = "https://api.exa.ai"
    search_type: str = "neural"
    requester: Any = _post_json
    name: str = "exa"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        endpoint = f"{self.base_url.rstrip('/')}/search"
        payload = {
            "query": query,
            "type": self.search_type,
            "num_results": max_results,
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        response = self.requester(endpoint, headers, payload)
        now = datetime.now(UTC).isoformat()
        results: list[SearchResult] = []
        for index, item in enumerate(response.get("results", []), start=1):
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            text = str(item.get("text", ""))
            results.append(
                SearchResult(
                    source_id=str(item.get("id") or f"EXA-{index:03d}"),
                    provider="exa",
                    source_type="web_search",
                    title=str(item.get("title", "Untitled")),
                    url=url,
                    source=url,
                    domain=_extract_domain(url),
                    snippet=(text[:500] if text else str(item.get("highlights", ""))[:500]),
                    content=text or None,
                    score=_safe_float(item.get("score"), default=0.62),
                    credibility=min(0.95, max(0.45, _safe_float(item.get("score"), default=0.62))),
                    published_at=str(item.get("published_date", "unknown")),
                    retrieved_at=now,
                    metadata={"search_type": self.search_type},
                )
            )
        return results


def build_web_search_provider(settings: Settings) -> BaseWebSearchProvider | None:
    if not settings.web_grounding.enable_web_search:
        return None

    provider_name, _ = resolve_web_mode(settings)

    if provider_name == "tavily":
        if not settings.tavily.api_key:
            raise WebSearchConfigError("TAVILY_API_KEY is required for Tavily web search.")
        return TavilySearchProvider(
            api_key=settings.tavily.api_key,
            search_depth=settings.tavily.search_depth,
            topic=settings.tavily.topic,
            include_raw_content=settings.tavily.include_raw_content,
        )

    if provider_name == "exa":
        if not settings.exa.api_key:
            raise WebSearchConfigError("EXA_API_KEY is required for Exa web search.")
        return ExaSearchProvider(
            api_key=settings.exa.api_key,
            base_url=settings.exa.base_url,
            search_type=settings.exa.search_type,
        )

    raise WebSearchConfigError(f"Unsupported web search provider: {provider_name}")


def resolve_web_mode(settings: Settings) -> tuple[str, str]:
    mode = settings.web_grounding.mode
    override_search = settings.web_grounding.search_provider
    override_reader = settings.web_grounding.reader_provider

    if override_search and override_reader:
        return override_search, override_reader
    if override_search == "tavily" and not override_reader:
        return "tavily", "jina"
    if override_search == "exa" and not override_reader:
        return "exa", "exa"
    if override_reader == "jina" and not override_search:
        return "tavily", "jina"
    if override_reader == "exa" and not override_search:
        return "exa", "exa"

    if mode == "tavily_jina":
        return "tavily", "jina"

    if mode == "exa":
        return "exa", "exa"

    if mode == "auto":
        if settings.tavily.api_key:
            return "tavily", "jina"
        if settings.exa.api_key:
            return "exa", "exa"
        raise WebSearchConfigError(
            "WEB_SEARCH_MODE=auto requires either TAVILY_API_KEY (for tavily+jina) "
            "or EXA_API_KEY (for exa search/contents)."
        )

    raise WebSearchConfigError(f"Unsupported WEB_SEARCH_MODE: {mode}")


def _extract_domain(url: str) -> str:
    value = url.replace("https://", "").replace("http://", "")
    return value.split("/", 1)[0].lower() if value else "unknown"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
