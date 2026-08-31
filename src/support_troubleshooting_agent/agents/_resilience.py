"""Shared resilience helpers for LLM and state-error handling."""

from __future__ import annotations

import time
from typing import Any


_TRANSIENT_ERROR_MARKERS = (
    "429",
    "rate limit",
    "too many requests",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "temporarily_unavailable",
    "connection reset",
    "connection error",
    "econnreset",
    "service unavailable",
    "503",
    "502",
    "500",
    "overloaded",
    "try again",
)


def is_transient_error(exc: Exception) -> bool:
    """Return True when an exception is likely transient and retryable."""

    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def append_error(
    state: dict[str, Any],
    step: str,
    message: str,
    *,
    details: str | None = None,
    user_message: str | None = None,
) -> list[dict[str, Any]]:
    """Append a structured error entry to the shared state while preserving prior errors."""

    current_errors = list(state.get("errors", []) or [])
    error_entry: dict[str, Any] = {"step": step, "message": message}
    if details:
        error_entry["details"] = details
    if user_message:
        error_entry["user_message"] = user_message
    current_errors.append(error_entry)
    return current_errors


def invoke_with_retry(model: Any, prompt: Any, payload: dict[str, Any], step_name: str) -> Any:
    """Retry transient LLM failures up to three attempts without changing the business logic."""

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return (prompt | model).invoke(payload)
        except Exception as exc:  # pragma: no cover - executed in real runtime only
            last_error = exc
            if not is_transient_error(exc) or attempt == 2:
                raise

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"{step_name}: LLM invocation failed without an exception payload.")


def add_execution_trace(
    state: dict[str, Any],
    agent_name: str,
    summary: str,
    *,
    decision: str = "completed",
    started_at: float | None = None,
) -> dict[str, Any]:
    """Record observable workflow actions and elapsed runtime without exposing raw reasoning."""

    completed_steps = list(state.get("completed_steps", []) or [])
    if agent_name not in completed_steps:
        completed_steps.append(agent_name)

    reasoning_summary = list(state.get("reasoning_summary", []) or [])
    entry: dict[str, Any] = {
        "agent": agent_name,
        "decision": decision,
        "summary": summary,
    }

    elapsed = 0.0
    if started_at is not None:
        elapsed = max(0.0, time.perf_counter() - started_at)
        entry["execution_time"] = round(elapsed, 3)

    reasoning_summary.append(entry)

    total_time = float(state.get("execution_time", 0.0) or 0.0)
    if started_at is not None:
        total_time += elapsed

    return {
        "current_agent": agent_name,
        "completed_steps": completed_steps,
        "reasoning_summary": reasoning_summary,
        "execution_time": round(total_time, 3),
    }
