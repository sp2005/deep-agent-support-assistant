"""LangGraph node for ChromaDB-backed retrieval from ticket and log context."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from support_troubleshooting_agent.agents._resilience import add_execution_trace, append_error
from support_troubleshooting_agent.rag.retriever import retrieve_documents


def _normalize_log_analysis(log_analysis: Any) -> str:
    """Convert structured log analysis into a compact retrieval query string."""

    if not log_analysis:
        return ""

    if isinstance(log_analysis, str):
        return log_analysis.strip()

    if isinstance(log_analysis, dict):
        summary = log_analysis.get("summary", "")
        anomalies = log_analysis.get("anomalies", [])
        evidence = log_analysis.get("evidence", [])
        parts = [summary]
        if isinstance(anomalies, list):
            parts.extend(str(item) for item in anomalies)
        if isinstance(evidence, list):
            parts.extend(str(item) for item in evidence)
        return " ".join(part for part in parts if part).strip()

    try:
        return json.dumps(log_analysis, ensure_ascii=False, default=str)
    except Exception:
        return str(log_analysis)


def _build_retrieval_query(state: dict[str, Any]) -> str:
    """Build a retrieval query from the ticket summary and log signals."""

    ticket_summary = str(state.get("ticket_summary", "")).strip()
    log_analysis = _normalize_log_analysis(state.get("log_analysis"))

    parts: list[str] = []
    if ticket_summary:
        parts.append(ticket_summary)
    if log_analysis:
        parts.append(log_analysis)

    return " ".join(parts).strip()


def rag_agent(
    state: dict[str, Any],
    retriever_fn: Callable[[str], list[dict[str, Any]]] = retrieve_documents,
) -> dict[str, Any]:
    """Retrieve the most relevant knowledge documents for ticket and log context.

    The existing ChromaDB retriever is reused as a dependency. We do not rewrite the
    vector-store logic here; we simply build the query from the state and delegate to
    the retriever.
    """

    started_at = time.perf_counter()
    query = _build_retrieval_query(state)
    try:
        documents = retriever_fn(query) if callable(retriever_fn) else []
    except Exception as exc:
        documents = []
        errors = append_error(
            state,
            "rag_agent",
            str(exc),
            details="Knowledge-base retrieval failed; the workflow will continue without documents.",
            user_message="No relevant knowledge-base documents were returned. Continuing without retrieval context.",
        )
        trace = add_execution_trace(
            state,
            "rag_agent",
            "Attempted to query the knowledge base and continued without retrieval context after a retrieval failure.",
            decision="fallback",
            started_at=started_at,
        )
        return {"retrieved_documents": documents, "current_step": "rag_agent", "errors": errors, **trace}

    if not documents:
        errors = append_error(
            state,
            "rag_agent",
            "No relevant documents were found for the current issue.",
            details="The retrieval result set is empty, so the workflow continues without additional knowledge-base context.",
            user_message="No relevant documents were found. Continuing without knowledge-base context.",
        )
        trace = add_execution_trace(
            state,
            "rag_agent",
            "Queried the knowledge base with the ticket and log context, but the retrieval set was empty.",
            decision="fallback",
            started_at=started_at,
        )
        return {"retrieved_documents": documents, "current_step": "rag_agent", "errors": errors, **trace}

    trace = add_execution_trace(
        state,
        "rag_agent",
        "Queried the knowledge base and retained the most relevant documents for the incident context.",
        started_at=started_at,
    )
    return {
        "retrieved_documents": documents,
        "current_step": "rag_agent",
        **trace,
    }


__all__ = ["rag_agent"]
