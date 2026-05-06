"""Benchmark skeleton for single-agent vs multi-agent comparison."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, token cost, and quality for a single run.

    Quality scoring heuristic (0-10):
      - final_answer present: +4
      - research_notes present: +2
      - analysis_notes present: +2
      - ≥1 source citation in final_answer: +1
      - no errors: +1
    """
    started = perf_counter()
    state: ResearchState | None = None
    try:
        state = runner(query)
        latency = perf_counter() - started
    except Exception as exc:
        latency = perf_counter() - started
        logger.error("Runner '%s' failed: %s", run_name, exc)
        from multi_agent_research_lab.core.schemas import ResearchQuery
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))

    # Cost estimation — sum across all agent results
    total_cost: float | None = None
    for result in state.agent_results:
        c = result.metadata.get("cost_usd")
        if c is not None:
            total_cost = (total_cost or 0) + c

    quality = _score_quality(state)

    notes = f"errors={state.errors}" if state.errors else ""
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality,
        notes=notes,
    )
    logger.info(
        "Benchmark '%s' | latency=%.2fs cost=$%.6f quality=%.1f/10",
        run_name,
        latency,
        total_cost or 0,
        quality,
    )
    return state, metrics


def _score_quality(state: ResearchState) -> float:
    """Simple rubric-based quality score 0-10."""
    score = 0.0
    if state.final_answer:
        score += 4
        # Check citations present
        if "[Source" in state.final_answer or "[source" in state.final_answer.lower():
            score += 1
    if state.research_notes:
        score += 2
    if state.analysis_notes:
        score += 2
    if not state.errors:
        score += 1
    return min(score, 10)
