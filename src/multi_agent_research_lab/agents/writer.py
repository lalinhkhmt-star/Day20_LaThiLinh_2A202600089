"""Writer agent - produces the final answer from research and analysis notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional technical writer. Synthesize research and analysis notes
into a concise final answer for the specified audience.

Requirements:
- Hard limit: 250 words, even if the user asks for a different length.
- Use 3 short sections: Summary, Key Points, References.
- Include inline citations like [Source N] for major claims.
- Keep references short: title + URL only.
- Maintain a neutral, informative tone.
"""


class WriterAgent(BaseAgent):
    """Synthesises research_notes + analysis_notes into a final answer with citations."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.final_answer."""
        if not state.research_notes:
            raise AgentExecutionError("WriterAgent requires research_notes.")

        source_refs = "\n".join(
            f"[Source {i+1}] {s.title} - {s.url or 'N/A'}"
            for i, s in enumerate(state.sources)
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis notes:\n{state.analysis_notes or 'N/A'}\n\n"
            f"Available sources:\n{source_refs}\n\n"
            "Write the final answer under 250 words with citations and short references."
        )

        with trace_span("writer.llm") as span:
            response = self._llm.complete(SYSTEM_PROMPT, user_prompt)
            span["input_tokens"] = response.input_tokens
            span["output_tokens"] = response.output_tokens
            span["cost_usd"] = response.cost_usd

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer.done",
            {"answer_length": len(response.content)},
        )
        logger.info("Writer produced %d chars final answer", len(response.content))
        return state
