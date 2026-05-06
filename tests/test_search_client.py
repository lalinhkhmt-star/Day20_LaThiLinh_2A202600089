from typing import Any

from multi_agent_research_lab.services.search_client import SearchClient


class FakeTavilyClient:
    def search(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "results": [
                {
                    "title": "Useful result",
                    "url": "https://example.com/useful",
                    "content": "A useful search snippet.",
                    "score": 0.91,
                },
                {
                    "title": "Empty result",
                    "url": "https://example.com/empty",
                    "content": "",
                    "score": 0.2,
                },
            ]
        }


def test_search_client_parses_tavily_results() -> None:
    client = SearchClient(
        api_key="test-key",
        tavily_factory=lambda _api_key: FakeTavilyClient(),
        use_mock_fallback=False,
    )

    docs = client.search("multi-agent systems", max_results=5)

    assert len(docs) == 1
    assert docs[0].title == "Useful result"
    assert docs[0].metadata["provider"] == "tavily"
    assert docs[0].metadata["score"] == 0.91


def test_search_client_uses_mock_without_api_key() -> None:
    client = SearchClient(api_key="")

    docs = client.search("GraphRAG", max_results=2)

    assert len(docs) == 2
    assert docs[0].metadata["provider"] == "mock"
