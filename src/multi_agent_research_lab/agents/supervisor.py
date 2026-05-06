"""Supervisor / router agent — decides which worker runs next and when to stop."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# Route constants
ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (deterministic state-machine):
      1. If research_notes is missing → route to researcher.
      2. If analysis_notes is missing (research done) → route to analyst.
      3. If final_answer is missing (analysis done) → route to writer.
      4. If final_answer is present → done.
      5. If max_iterations exceeded → fallback to writer or done.
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update state.route_history with the next route and return state."""
        settings = get_settings()

        with trace_span("supervisor.route", {"iteration": state.iteration}) as span:
            next_route = self._decide(state, settings.max_iterations)
            span["next_route"] = next_route

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor.routed",
            {"route": next_route, "iteration": state.iteration},
        )
        logger.info(
            "Supervisor → %s (iteration %d/%d)",
            next_route,
            state.iteration,
            settings.max_iterations,
        )
        return state

    def _decide(self, state: ResearchState, max_iterations: int) -> str:
        """Deterministic routing based on current state fields."""
        # Guard: max iterations exceeded — force done to prevent infinite loop
        if state.iteration >= max_iterations:
            logger.warning(
                "Max iterations (%d) reached — forcing done. errors=%s",
                max_iterations,
                state.errors,
            )
            if state.final_answer is None:
                state.final_answer = (
                    f"[Fallback] Max iterations reached after {state.iteration} steps. "
                    f"Partial answer: {state.analysis_notes or state.research_notes or 'No output.'}"
                )
            return ROUTE_DONE

        # Happy path
        if state.final_answer is not None:
            return ROUTE_DONE
        if state.analysis_notes is not None:
            return ROUTE_WRITER
        if state.research_notes is not None:
            return ROUTE_ANALYST
        return ROUTE_RESEARCHER
