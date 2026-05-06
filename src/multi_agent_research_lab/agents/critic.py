"""Optional critic agent for lightweight answer validation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append basic citation coverage findings."""
        answer = state.final_answer or ""
        cited_sources = {
            index
            for index in range(1, len(state.sources) + 1)
            if f"[Source {index}]" in answer
        }
        coverage = len(cited_sources) / len(state.sources) if state.sources else 0.0
        findings = (
            f"Citation coverage: {len(cited_sources)}/{len(state.sources)} "
            f"({coverage:.0%})."
        )
        if not answer:
            findings += " Final answer is missing."
            state.errors.append("critic: final_answer missing")
        elif state.sources and coverage < 0.5:
            findings += " Citation coverage is below the recommended 50% threshold."
            state.errors.append("critic: low citation coverage")
        else:
            findings += " Basic validation passed."

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=findings,
                metadata={"citation_coverage": coverage},
            )
        )
        state.add_trace_event(
            "critic.done",
            {"citation_coverage": coverage, "num_sources": len(state.sources)},
        )
        return state
