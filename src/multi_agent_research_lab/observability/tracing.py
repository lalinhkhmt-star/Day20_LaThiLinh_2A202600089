"""Tracing hooks for local state traces and optional LangSmith upload."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)
_current_parent_run_id: ContextVar[Any | None] = ContextVar(
    "current_langsmith_parent_run_id",
    default=None,
)


@lru_cache(maxsize=1)
def _get_langsmith_client() -> Any | None:
    settings = get_settings()
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return None

    try:
        from langsmith import Client  # type: ignore[import-untyped]

        return Client(
            api_url=settings.langsmith_endpoint,
            api_key=settings.langsmith_api_key,
        )
    except ImportError:
        logger.warning("langsmith package not installed; LangSmith traces disabled")
    except Exception as exc:
        logger.warning("Could not initialize LangSmith client: %s", exc)
    return None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal provider-neutral span context used by agents and workflows.

    The yielded dictionary is still used by local code. If `LANGSMITH_TRACING=true`
    and `LANGSMITH_API_KEY` is configured, the same span is also uploaded to LangSmith.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    settings = get_settings()
    client = _get_langsmith_client()
    langsmith_run_id = uuid4() if client is not None else None

    if client is not None and langsmith_run_id is not None:
        try:
            client.create_run(
                id=langsmith_run_id,
                name=name,
                run_type=_infer_run_type(name),
                inputs=attributes or {},
                project_name=settings.langsmith_project,
                parent_run_id=_current_parent_run_id.get(),
                start_time=datetime.now(UTC),
                tags=["multi-agent-research-lab"],
                extra={"metadata": {"app_env": settings.app_env}},
            )
        except Exception as exc:
            langsmith_run_id = None
            logger.warning("Could not start LangSmith span '%s': %s", name, exc)

    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if client is not None and langsmith_run_id is not None:
            try:
                client.update_run(
                    langsmith_run_id,
                    outputs=span,
                    end_time=datetime.now(UTC),
                    error=span.get("error"),
                )
            except Exception as exc:
                logger.warning("Could not finish LangSmith span '%s': %s", name, exc)


@contextmanager
def trace_workflow(
    name: str,
    inputs: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Create one parent LangSmith run and nest all `trace_span` calls under it."""
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": inputs or {}, "duration_seconds": None}
    settings = get_settings()
    client = _get_langsmith_client()
    run_id = uuid4() if client is not None else None
    token = None

    if client is not None and run_id is not None:
        try:
            client.create_run(
                id=run_id,
                name=name,
                run_type="chain",
                inputs=inputs or {},
                project_name=settings.langsmith_project,
                start_time=datetime.now(UTC),
                tags=["multi-agent-research-lab", "workflow"],
                extra={"metadata": {"app_env": settings.app_env}},
            )
            token = _current_parent_run_id.set(run_id)
        except Exception as exc:
            run_id = None
            logger.warning("Could not start LangSmith workflow '%s': %s", name, exc)

    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if token is not None:
            _current_parent_run_id.reset(token)
        if client is not None and run_id is not None:
            try:
                client.update_run(
                    run_id,
                    outputs=span,
                    end_time=datetime.now(UTC),
                    error=span.get("error"),
                )
            except Exception as exc:
                logger.warning("Could not finish LangSmith workflow '%s': %s", name, exc)


def flush_langsmith_traces() -> None:
    """Flush buffered LangSmith trace operations before the CLI process exits."""
    client = _get_langsmith_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("Could not flush LangSmith traces: %s", exc)


def _infer_run_type(name: str) -> str:
    if ".llm" in name:
        return "llm"
    if ".search" in name:
        return "tool"
    return "chain"
