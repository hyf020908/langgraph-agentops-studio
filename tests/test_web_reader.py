from __future__ import annotations

from services.web_reader import ExaContentsProvider, JinaReaderProvider


def test_jina_reader_skips_failed_urls_and_keeps_successes() -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    def requester(url: str, headers: dict[str, str], timeout: float) -> str:
        calls.append((url, headers))
        if "blocked.example" in url:
            raise RuntimeError("403 forbidden")
        return "Readable page content"

    provider = JinaReaderProvider(api_key="token", requester=requester)
    pages = provider.read_urls(["https://blocked.example/a", "https://ok.example/b"])

    assert list(pages) == ["https://ok.example/b"]
    assert pages["https://ok.example/b"].content == "Readable page content"
    assert all("Mozilla/5.0" in headers["User-Agent"] for _, headers in calls)
    assert all(headers["Authorization"] == "Bearer token" for _, headers in calls)


def test_jina_reader_returns_empty_when_all_urls_fail() -> None:
    provider = JinaReaderProvider(requester=lambda url, headers, timeout: (_ for _ in ()).throw(RuntimeError("403")))

    assert provider.read_urls(["https://blocked.example/a"]) == {}


def test_exa_reader_returns_empty_when_batch_request_fails() -> None:
    provider = ExaContentsProvider(api_key="token", requester=lambda *args: (_ for _ in ()).throw(RuntimeError("403")))

    assert provider.read_urls(["https://blocked.example/a"]) == {}
