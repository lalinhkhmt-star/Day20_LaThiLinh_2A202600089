"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

TavilyFactory = Callable[[str], Any]


class SearchClient:
    """Provider-agnostic search client backed by Tavily with mock fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        tavily_factory: TavilyFactory | None = None,
        use_mock_fallback: bool = True,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.tavily_api_key
        self._tavily_factory = tavily_factory
        self._use_mock_fallback = use_mock_fallback

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily if TAVILY_API_KEY is set; otherwise falls back to curated mock results.
        """
        max_results = max(1, max_results)
        if self._api_key:
            return self._tavily_search(query, max_results)

        logger.warning("TAVILY_API_KEY not set; using mock search results")
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            client = self._build_tavily_client()
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=False,
            )
            docs = self._parse_tavily_results(response, max_results)
            logger.info("Tavily search returned %d results for query: %.60s", len(docs), query)
            if docs:
                return docs
            logger.warning("Tavily returned no usable results for query: %.60s", query)
        except ImportError:
            logger.warning("tavily-python not installed; falling back to mock search")
        except Exception as exc:
            logger.error("Tavily search failed: %s; falling back to mock", exc)

        if self._use_mock_fallback:
            return self._mock_search(query, max_results)
        return []

    def _build_tavily_client(self) -> Any:
        if self._tavily_factory is not None:
            return self._tavily_factory(self._api_key or "")

        from tavily import TavilyClient  # type: ignore[import-untyped]

        return TavilyClient(api_key=self._api_key)

    def _parse_tavily_results(
        self, response: dict[str, Any], max_results: int
    ) -> list[SourceDocument]:
        docs: list[SourceDocument] = []
        for result in response.get("results", [])[:max_results]:
            title = str(result.get("title") or "Untitled").strip()
            snippet = str(result.get("content") or result.get("snippet") or "").strip()
            url = result.get("url")
            if not snippet:
                continue
            docs.append(
                SourceDocument(
                    title=title,
                    url=str(url) if url else None,
                    snippet=snippet,
                    metadata={
                        "provider": "tavily",
                        "score": result.get("score"),
                        "raw_content_available": bool(result.get("raw_content")),
                    },
                )
            )
        return docs

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return a small set of curated mock results about GraphRAG and agents."""
        mock_results = [
            SourceDocument(
                title="GraphRAG: Unlocking LLM discovery on narrative private data",
                url="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/",
                snippet=(
                    "GraphRAG is a structured, hierarchical approach to Retrieval-Augmented Generation "
                    "that uses knowledge graphs to answer complex queries over large text corpora."
                ),
                metadata={"provider": "mock", "query": query},
            ),
            SourceDocument(
                title="LangGraph: Building Stateful Multi-Actor Applications",
                url="https://langchain-ai.github.io/langgraph/concepts/",
                snippet=(
                    "LangGraph extends LangChain to build cyclic, stateful graphs of agents that can "
                    "collaborate and route tasks between specialised nodes."
                ),
                metadata={"provider": "mock", "query": query},
            ),
            SourceDocument(
                title="Anthropic: Building Effective Agents",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Effective agents combine clear role definitions, minimal coupling, explicit state "
                    "handoff, and robust failure handling rather than monolithic LLM calls."
                ),
                metadata={"provider": "mock", "query": query},
            ),
            SourceDocument(
                title="OpenAI Agents SDK - Orchestration and Handoffs",
                url="https://developers.openai.com/api/docs/guides/agents/orchestration",
                snippet=(
                    "The OpenAI Agents SDK supports multi-agent orchestration with handoffs, tracing, "
                    "and guardrails for building reliable agentic pipelines."
                ),
                metadata={"provider": "mock", "query": query},
            ),
            SourceDocument(
                title="From RAG to GraphRAG: State-of-the-Art Survey 2024",
                url="https://arxiv.org/abs/2404.16130",
                snippet=(
                    "This survey compares naive RAG, advanced RAG, and GraphRAG across precision, "
                    "recall, and hallucination metrics, highlighting graph-based indexing advantages."
                ),
                metadata={"provider": "mock", "query": query},
            ),
        ]
        return mock_results[:max_results]
