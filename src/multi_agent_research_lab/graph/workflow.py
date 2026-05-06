"""LangGraph workflow — builds and runs the multi-agent research graph."""

from __future__ import annotations

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_workflow

logger = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    """Convert ResearchState to plain dict for LangGraph."""
    return state.model_dump()


def _dict_to_state(data: dict[str, Any]) -> ResearchState:
    """Convert plain dict back to ResearchState."""
    return ResearchState.model_validate(data)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent research graph using LangGraph.

    Graph topology:
        START → supervisor → {researcher | analyst | writer | END}
        researcher → supervisor
        analyst → supervisor
        writer → supervisor
        supervisor (route=done) → END
    """

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()

    def build(self) -> Any:
        """Create and return a compiled LangGraph graph."""
        try:
            from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "langgraph not installed. Run: pip install 'langgraph>=0.2'"
            ) from exc

        graph = StateGraph(dict)

        # --- Node functions ---
        def supervisor_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(state_dict)
            state = self._supervisor.run(state)
            return _state_to_dict(state)

        def researcher_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(state_dict)
            try:
                state = self._researcher.run(state)
            except Exception as exc:
                logger.error("ResearcherAgent failed: %s", exc)
                state.errors.append(f"researcher: {exc}")
            return _state_to_dict(state)

        def analyst_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(state_dict)
            try:
                state = self._analyst.run(state)
            except Exception as exc:
                logger.error("AnalystAgent failed: %s", exc)
                state.errors.append(f"analyst: {exc}")
            return _state_to_dict(state)

        def writer_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(state_dict)
            try:
                state = self._writer.run(state)
            except Exception as exc:
                logger.error("WriterAgent failed: %s", exc)
                state.errors.append(f"writer: {exc}")
            return _state_to_dict(state)

        # --- Routing function ---
        def route_after_supervisor(state_dict: dict[str, Any]) -> str:
            state = _dict_to_state(state_dict)
            if not state.route_history:
                return ROUTE_RESEARCHER
            last_route = state.route_history[-1]
            if last_route == ROUTE_DONE:
                return END  # type: ignore[return-value]
            return last_route

        # --- Add nodes ---
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)

        # --- Add edges ---
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {
                ROUTE_RESEARCHER: "researcher",
                ROUTE_ANALYST: "analyst",
                ROUTE_WRITER: "writer",
                END: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        compiled = graph.compile()
        logger.info("MultiAgentWorkflow graph compiled successfully")
        return compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Compile the graph, invoke it, and return the final ResearchState."""
        with trace_workflow(
            "multi_agent.workflow",
            {
                "query": state.request.query,
                "max_sources": state.request.max_sources,
                "audience": state.request.audience,
            },
        ) as span:
            try:
                graph = self.build()
            except ImportError:
                logger.warning("LangGraph unavailable; running deterministic in-process workflow")
                result = self._run_without_langgraph(state)
            else:
                initial = _state_to_dict(state)
                result_dict = graph.invoke(initial)
                result = _dict_to_state(result_dict)

            span["route_history"] = result.route_history
            span["iterations"] = result.iteration
            span["errors"] = result.errors
            return result

    def _run_without_langgraph(self, state: ResearchState) -> ResearchState:
        """Run the same supervisor-worker loop without LangGraph for local smoke tests."""
        while True:
            state = self._supervisor.run(state)
            last_route = state.route_history[-1]
            if last_route == ROUTE_DONE:
                return state
            try:
                if last_route == ROUTE_RESEARCHER:
                    state = self._researcher.run(state)
                elif last_route == ROUTE_ANALYST:
                    state = self._analyst.run(state)
                elif last_route == ROUTE_WRITER:
                    state = self._writer.run(state)
                else:
                    state.errors.append(f"unknown route: {last_route}")
            except Exception as exc:
                logger.error("%s route failed: %s", last_route, exc)
                state.errors.append(f"{last_route}: {exc}")
