"""Root cause analysis agent that correlates ticket, logs, and retrieval context."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from support_troubleshooting_agent.agents._resilience import add_execution_trace, append_error, invoke_with_retry
from support_troubleshooting_agent.models.llm_factory import get_chat_model


def _coerce_text(value: Any) -> str:
    """Convert ticket/log/doc context into normalized text for the model."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _fallback_root_cause(ticket_summary: str, log_analysis: Any, retrieved_documents: Any) -> dict[str, Any]:
    """Return a structured root-cause payload when the model is unavailable or cannot parse output."""

    combined = " ".join(
        part for part in [
            ticket_summary,
            _coerce_text(log_analysis),
            _coerce_text(retrieved_documents),
        ] if part
    )
    if not combined:
        return {
            "primary_cause": "No sufficient evidence was provided to determine a root cause.",
            "confidence": 0.0,
            "supporting_evidence": [],
            "reasoning": "The workflow did not receive enough ticket, log, or retrieval context.",
        }

    evidence: list[str] = []
    for token in ["504", "timeout", "MongoDB", "slow query", "deployment", "rollback", "database"]:
        if token.lower() in combined.lower():
            evidence.append(token)

    return {
        "primary_cause": "Evidence suggests a deployment or dependency regression is causing the observed failures.",
        "confidence": 0.7,
        "supporting_evidence": evidence[:5] if evidence else ["Ticket, log, and retrieval evidence were correlated but not explicit enough to isolate a single failure mode."],
        "reasoning": "The ticket and logs indicate a system-level failure pattern, and the retrieval context should be used to narrow the root cause further.",
    }


def diagnosis_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Correlate the ticket summary, logs, and retrieved docs and produce a root-cause object."""

    started_at = time.perf_counter()

    ticket_summary = _coerce_text(state.get("ticket_summary"))
    log_analysis = state.get("log_analysis")
    retrieved_documents = state.get("retrieved_documents")

    if not ticket_summary and not log_analysis and not retrieved_documents:
        fallback = {
            "root_cause": {
                "primary_cause": "No evidence supplied.",
                "confidence": 0.0,
                "supporting_evidence": [],
                "reasoning": "No ticket summary, log data, or retrieved documents were provided.",
            },
            "current_step": "diagnosis_agent",
        }
        errors = append_error(
            state,
            "diagnosis_agent",
            "No sufficient incident evidence was available for diagnosis.",
            details="The workflow lacked ticket, log, and document evidence for root-cause analysis.",
            user_message="Not enough evidence was available to determine a root cause. Continuing with the incomplete diagnosis result.",
        )
        trace = add_execution_trace(
            state,
            "diagnosis_agent",
            "Checked the available evidence and detected that the workflow lacked enough data for a reliable root-cause conclusion.",
            decision="fallback",
            started_at=started_at,
        )
        fallback["errors"] = errors
        return {**fallback, **trace}

    try:
        model = get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior incident investigator. Correlate the ticket summary, logs, and retrieved documents to determine the most likely root cause. Return compact JSON only with keys: primary_cause, confidence, supporting_evidence, reasoning.",
                ),
                (
                    "human",
                    "Ticket Summary:\n{ticket_summary}\n\nLog Analysis:\n{log_analysis}\n\nRetrieved Documents:\n{retrieved_documents}",
                ),
            ]
        )

        response = invoke_with_retry(
            model,
            prompt,
            {
                "ticket_summary": ticket_summary,
                "log_analysis": _coerce_text(log_analysis),
                "retrieved_documents": _coerce_text(retrieved_documents),
            },
            "diagnosis_agent",
        )
        content = getattr(response, "content", response)
        text = str(content).strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            result = {
                "root_cause": _fallback_root_cause(ticket_summary, log_analysis, retrieved_documents),
                "current_step": "diagnosis_agent",
            }
            errors = append_error(
                state,
                "diagnosis_agent",
                "The model returned invalid diagnosis JSON.",
                details="The fallback root-cause payload was used because the response could not be parsed.",
                user_message="The root-cause response was incomplete. Continuing with the fallback diagnosis.",
            )
            trace = add_execution_trace(
                state,
                "diagnosis_agent",
                "Correlated the incident evidence and used the fallback diagnosis after the model output was invalid.",
                decision="fallback",
                started_at=started_at,
            )
            result["errors"] = errors
            return {**result, **trace}

        normalized = {
            "primary_cause": str(parsed.get("primary_cause", "Unable to determine root cause.")),
            "confidence": float(parsed.get("confidence", 0.5)),
            "supporting_evidence": parsed.get("supporting_evidence", []) if isinstance(parsed.get("supporting_evidence", []), list) else [str(parsed.get("supporting_evidence", ""))],
            "reasoning": str(parsed.get("reasoning", "The provided evidence was correlated to determine the likely root cause.")),
        }
        trace = add_execution_trace(
            state,
            "diagnosis_agent",
            "Correlated the ticket, logs, and knowledge context to determine the most likely root cause and confidence level.",
            started_at=started_at,
        )
        return {"root_cause": normalized, "current_step": "diagnosis_agent", **trace}
    except Exception as exc:
        fallback = _fallback_root_cause(ticket_summary, log_analysis, retrieved_documents)
        errors = append_error(
            state,
            "diagnosis_agent",
            str(exc),
            details="Root cause correlation failed; fallback payload used so the workflow can continue.",
            user_message="Root-cause analysis could not complete. Continuing with the fallback diagnosis.",
        )
        trace = add_execution_trace(
            state,
            "diagnosis_agent",
            "Encountered a diagnosis error and switched to the fallback root-cause path to maintain workflow continuity.",
            decision="fallback",
            started_at=started_at,
        )
        return {
            "root_cause": fallback,
            "current_step": "diagnosis_agent",
            "errors": errors,
            **trace,
        }


__all__ = ["diagnosis_agent"]
