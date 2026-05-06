"""Analyst agent - turns research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a critical analyst. Given research notes, produce a compact analysis.

Your output must include:
1. **Key Claims** - 3 findings with evidence strength (Strong/Moderate/Weak).
2. **Trade-offs** - 1-2 important caveats.
3. **Recommendation** - what the final answer should emphasise.

Hard limit: 140 words. Use terse bullet points.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured key claims, gaps, and recommendations."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.analysis_notes from state.research_notes."""
        if not state.research_notes:
            raise AgentExecutionError("AnalystAgent requires research_notes; run ResearcherAgent first.")

        user_prompt = (
            f"Original query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            "Produce a compact structured analysis as described. Stay under 140 words."
        )

        with trace_span("analyst.llm") as span:
            response = self._llm.complete(SYSTEM_PROMPT, user_prompt)
            span["input_tokens"] = response.input_tokens
            span["output_tokens"] = response.output_tokens
            span["cost_usd"] = response.cost_usd

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst.done",
            {"analysis_length": len(response.content)},
        )
        logger.info("Analyst produced %d chars of analysis notes", len(response.content))
        return state
