from multi_agent_research_lab.agents import CriticAgent, SupervisorAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_happy_path() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == ROUTE_RESEARCHER

    state.research_notes = "notes"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == ROUTE_ANALYST

    state.analysis_notes = "analysis"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == ROUTE_WRITER

    state.final_answer = "answer"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == ROUTE_DONE


def test_critic_records_citation_coverage() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[
            SourceDocument(title="A", url=None, snippet="a"),
            SourceDocument(title="B", url=None, snippet="b"),
        ],
        final_answer="A claim [Source 1].",
    )

    state = CriticAgent().run(state)

    assert state.agent_results[-1].metadata["citation_coverage"] == 0.5
    assert state.trace[-1]["name"] == "critic.done"
