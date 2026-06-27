from __future__ import annotations

# External webpage-reader providers.
# Reader providers fetch fuller page content for top search hits so the research
# pipeline can rank evidence on more than snippets alone.

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib import request as urllib_request
from urllib.parse import quote

from pydantic import BaseModel, Field

from services.config import Settings
from services.web_search import resolve_web_mode

logger = logging.getLogger("agentops.web_reader")


class WebReaderConfigError(RuntimeError):
    pass


class PageContent(BaseModel):
    url: str
    title: str | None = None
    content: str
    provider: str
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseWebReaderProvider(Protocol):
    name: str

    def read_urls(self, urls: list[str]) -> dict[str, PageContent]:
        ...


def _browser_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Upgrade-Insecure-Requests": "1",
    }


def _get_text(url: str, headers: dict[str, str], timeout: float = 30.0) -> str:
    req = urllib_request.Request(url=url, headers={**_browser_headers(), **headers}, method="GET")
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url=url, data=data, headers=headers, method="POST")
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


@dataclass(slots=True)
class JinaReaderProvider:
    base_url: str = "https://r.jina.ai/"
    api_key: str | None = None
    timeout: float = 30.0
    use_json: bool = False
    bypass_cache: bool = False
    requester: Any = _get_text
    name: str = "jina"

    def read_urls(self, urls: list[str]) -> dict[str, PageContent]:
        outputs: dict[str, PageContent] = {}
        headers = _browser_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.use_json:
            headers["Accept"] = "application/json"
        if self.bypass_cache:
            headers["x-no-cache"] = "true"

        for url in urls:
            target = self._build_reader_url(url)
            # Jina is called once per URL because the reader endpoint is URL-based.
            try:
                content = self.requester(target, headers, self.timeout)
                text = _coerce_jina_text(content)
            except Exception:
                logger.warning("Jina reader skipped URL after fetch failure: %s", url, exc_info=True)
                continue
            if not text.strip():
                continue
            outputs[url] = PageContent(url=url, content=text, provider="jina", metadata={"reader_url": target})
        return outputs

    def _build_reader_url(self, url: str) -> str:
        base = self.base_url.rstrip("/")
        if "{url}" in self.base_url:
            return self.base_url.format(url=quote(url, safe=""))
        return f"{base}/{url}"


@dataclass(slots=True)
class ExaContentsProvider:
    api_key: str
    base_url: str = "https://api.exa.ai"
    requester: Any = _post_json
    timeout: float = 30.0
    name: str = "exa"

    def read_urls(self, urls: list[str]) -> dict[str, PageContent]:
        if not urls:
            return {}

        # Exa supports batch content retrieval, so the provider returns a map
        # keyed by original URL for easy reconciliation with search results.
        endpoint = f"{self.base_url.rstrip('/')}/contents"
        payload = {"urls": urls, "text": True}
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        try:
            response = self.requester(endpoint, headers, payload, self.timeout)
        except Exception:
            logger.warning("Exa contents reader failed; continuing without page contents.", exc_info=True)
            return {}

        outputs: dict[str, PageContent] = {}
        for item in response.get("results", []):
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            outputs[url] = PageContent(
                url=url,
                title=item.get("title"),
                content=text,
                provider="exa",
                metadata={"source": "exa_contents"},
            )
        return outputs


def build_web_reader_provider(settings: Settings) -> BaseWebReaderProvider | None:
    if not settings.web_grounding.enable_web_search:
        return None

    _, reader_name = resolve_web_mode(settings)

    if reader_name == "jina":
        return JinaReaderProvider(
            base_url=settings.jina.base_url,
            api_key=settings.jina.api_key,
            timeout=settings.jina.timeout,
            use_json=settings.jina.use_json,
            bypass_cache=settings.jina.bypass_cache,
        )

    if reader_name == "exa":
        if not settings.exa.api_key:
            raise WebReaderConfigError("EXA_API_KEY is required when WEB_READER_PROVIDER=exa.")
        return ExaContentsProvider(
            api_key=settings.exa.api_key,
            base_url=settings.exa.base_url,
        )

    raise WebReaderConfigError(f"Unsupported web reader provider: {reader_name}")


def _coerce_jina_text(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(parsed, dict):
        for key in ("content", "markdown", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return text
