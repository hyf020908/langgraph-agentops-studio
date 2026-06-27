from __future__ import annotations

from services.web_search import TavilySearchProvider


def test_tavily_source_ids_are_stable_and_url_based() -> None:
    response = {
        "results": [
            {"title": "A", "url": "https://example.com/a", "content": "alpha", "score": 0.8},
            {"title": "B", "url": "https://example.com/b", "content": "beta", "score": 0.7},
        ]
    }
    provider = TavilySearchProvider(api_key="token", requester=lambda *args: response)

    first = provider.search("first query", max_results=2)
    second = provider.search("second query", max_results=2)

    assert first[0].source_id == second[0].source_id
    assert first[0].source_id != first[1].source_id
    assert first[0].source_id.startswith("TAV-")
