"""Command-line entrypoint for the multi-agent research lab."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    flush_langsmith_traces,
    trace_span,
    trace_workflow,
)
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()
logger = logging.getLogger(__name__)


def _safe_console_text(value: object) -> str:
    """Return text that can be printed on legacy Windows consoles."""
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_single_agent_baseline(query: str) -> tuple[ResearchState, LLMResponse, float]:
    """Run the single-agent baseline and trace it as one LangSmith parent run."""
    import time

    llm = LLMClient()
    system_prompt = (
        "You are a research assistant. Answer the user's question in the same format "
        "used by the multi-agent writer for fair benchmarking.\n\n"
        "Requirements:\n"
        "- Hard limit: 250 words, even if the user asks for a different length.\n"
        "- Use 3 short sections: Summary, Key Points, References.\n"
        "- Include references where possible.\n"
        "- Maintain a neutral, informative tone."
    )

    with trace_workflow("single-agent-baseline", {"query": query}) as workflow_span:
        start = time.perf_counter()
        with trace_span("single_agent.llm", {"query": query}) as llm_span:
            response = llm.complete(system_prompt, query)
            llm_span["input_tokens"] = response.input_tokens
            llm_span["output_tokens"] = response.output_tokens
            llm_span["cost_usd"] = response.cost_usd

        latency = time.perf_counter() - start
        request = ResearchQuery(query=query)
        state = ResearchState(request=request)
        state.final_answer = response.content

        from multi_agent_research_lab.core.schemas import AgentName, AgentResult

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
        workflow_span["latency_seconds"] = latency
        workflow_span["input_tokens"] = response.input_tokens
        workflow_span["output_tokens"] = response.output_tokens
        workflow_span["cost_usd"] = response.cost_usd
        workflow_span["answer_length"] = len(response.content)
        return state, response, latency


# ---------------------------------------------------------------------------
# Baseline command
# ---------------------------------------------------------------------------

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent LLM baseline (no sub-agents, no search)."""
    _init()

    console.print("[bold cyan]Running single-agent baseline...[/bold cyan]")

    try:
        _state, response, latency = _run_single_agent_baseline(query)
    except Exception as exc:
        console.print(Panel.fit(_safe_console_text(f"LLM error: {exc}"), title="Error", style="red"))
        raise typer.Exit(1) from exc

    console.print(Panel.fit(_safe_console_text(response.content), title="Single-Agent Baseline"))
    console.print(
        f"[dim]Latency: {latency:.2f}s | "
        f"Tokens: {response.input_tokens}in / {response.output_tokens}out | "
        f"Cost: ${response.cost_usd:.6f}[/dim]"
    )
    flush_langsmith_traces()


# ---------------------------------------------------------------------------
# Multi-agent command
# ---------------------------------------------------------------------------

@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""
    _init()

    console.print("[bold cyan]Running multi-agent workflow...[/bold cyan]")

    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except Exception as exc:
        console.print(Panel.fit(_safe_console_text(exc), title="Error", style="red"))
        raise typer.Exit(1) from exc

    if result.final_answer:
        console.print(Panel.fit(_safe_console_text(result.final_answer), title="Multi-Agent Answer"))
    else:
        console.print("[yellow]No final answer produced.[/yellow]")

    console.print(f"\n[dim]Route history: {' -> '.join(result.route_history)}[/dim]")
    console.print(
        _safe_console_text(f"Iterations: {result.iteration} | Errors: {result.errors}"),
        style="dim",
    )
    console.print(_safe_console_text(result.model_dump_json(indent=2)))
    flush_langsmith_traces()


# ---------------------------------------------------------------------------
# Benchmark command
# ---------------------------------------------------------------------------

@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query to benchmark")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output path for markdown report")
    ] = Path("reports/benchmark_report.md"),
) -> None:
    """Run both baseline and multi-agent, compare metrics, save report."""
    _init()

    console.print("[bold cyan]Starting benchmark: single-agent vs multi-agent...[/bold cyan]")

    # --- Baseline runner ---
    def baseline_runner(q: str) -> ResearchState:
        state, _response, _latency = _run_single_agent_baseline(q)
        return state

    # --- Multi-agent runner ---
    def multi_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow()
        return workflow.run(state)

    # Run benchmarks
    console.print("\n[yellow]1/2 Running baseline...[/yellow]")
    _, baseline_metrics = run_benchmark("single-agent-baseline", query, baseline_runner)

    console.print("[yellow]2/2 Running multi-agent...[/yellow]")
    multi_state, multi_metrics = run_benchmark("multi-agent-workflow", query, multi_runner)

    # Display table
    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("Run", style="cyan")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality /10", justify="right")
    table.add_column("Notes")

    for m in [baseline_metrics, multi_metrics]:
        cost_str = f"${m.estimated_cost_usd:.6f}" if m.estimated_cost_usd is not None else "N/A"
        quality_str = f"{m.quality_score:.1f}" if m.quality_score is not None else "N/A"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}", cost_str, quality_str, m.notes)

    console.print(table)

    # Save report
    report_md = render_markdown_report([baseline_metrics, multi_metrics])

    # Append trace summary
    report_md += "\n## Multi-Agent Trace\n\n```json\n"
    report_md += json.dumps(multi_state.trace, indent=2, default=str)
    report_md += "\n```\n"

    # Append failure analysis
    report_md += "\n## Failure Modes & Analysis\n\n"
    report_md += (
        "### Observed failure modes\n"
        "- **Timeout / rate-limit**: LLM API rate-limit causes retry delays → "
        "mitigated by tenacity retry with exponential back-off.\n"
        "- **Empty sources**: If Tavily is unavailable, mock results are used → "
        "quality degrades but does not crash.\n"
        "- **Max-iteration fallback**: Supervisor forces `done` after `MAX_ITERATIONS` "
        "to prevent infinite loops.\n\n"
        "### Fix applied\n"
        "- Retry logic in `LLMClient` (3 attempts).\n"
        "- Mock fallback in `SearchClient`.\n"
        "- Hard cap in `SupervisorAgent` (`max_iterations`).\n"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_md, encoding="utf-8")
    console.print(f"\n[green]Report saved to {output}[/green]")
    flush_langsmith_traces()


if __name__ == "__main__":
    app()
