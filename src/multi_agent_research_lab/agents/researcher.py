"""Researcher agent — collects sources and summarises research notes."""

from __future__ import annotations

import logging
from textwrap import shorten

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research specialist. Given a topic, synthesise the provided search
results into concise, well-structured research notes.

Rules:
- Extract only the top 5 facts needed to answer the query.
- Cite each claim with [Source N] notation.
- Hard limit: 180 words.
- Do NOT invent information not present in the sources.
"""

MAX_SOURCE_SNIPPET_CHARS = 500


class ResearcherAgent(BaseAgent):
    """Collects sources via search and creates concise, cited research notes."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.sources and state.research_notes."""
        query = state.request.query
        max_sources = state.request.max_sources

        with trace_span("researcher.search", {"query": query}) as span:
            sources = self._search.search(query, max_results=max_sources)
            span["num_sources"] = len(sources)

        state.sources = sources
        logger.info("Researcher found %d sources", len(sources))

        # Format sources for LLM
        sources_text = "\n\n".join(
            f"[Source {i+1}] {s.title}\nURL: {s.url or 'N/A'}\n"
            f"{shorten(s.snippet, width=MAX_SOURCE_SNIPPET_CHARS, placeholder='...')}"
            for i, s in enumerate(sources)
        )

        user_prompt = (
            f"Research topic: {query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Sources:\n{sources_text}\n\n"
            "Write compact research notes with [Source N] citations. Stay under 180 words."
        )

        with trace_span("researcher.llm") as span:
            response = self._llm.complete(SYSTEM_PROMPT, user_prompt)
            span["input_tokens"] = response.input_tokens
            span["output_tokens"] = response.output_tokens
            span["cost_usd"] = response.cost_usd

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "num_sources": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher.done",
            {"num_sources": len(sources), "notes_length": len(response.content)},
        )
        logger.info("Researcher produced %d chars of research notes", len(response.content))
        return state
