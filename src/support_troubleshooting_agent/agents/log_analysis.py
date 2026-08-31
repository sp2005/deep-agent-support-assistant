"""Log analysis agent that summarizes runtime health signals from app, nginx, and MongoDB logs."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from support_troubleshooting_agent.agents._resilience import add_execution_trace, append_error, invoke_with_retry
from support_troubleshooting_agent.models.llm_factory import get_chat_model


def _collect_log_text(state: dict[str, Any]) -> str:
    """Collect all available log inputs from the state into a single text block."""

    parts: list[str] = []
    for key in ("application_logs", "nginx_logs", "mongodb_logs"):
        value = state.get(key)
        if value:
            if isinstance(value, str):
                parts.append(f"[{key}]\n{value}")
            else:
                parts.append(f"[{key}]\n{json.dumps(value, ensure_ascii=False, default=str)}")

    return "\n\n".join(parts)


def _fallback_summary(text: str) -> dict[str, Any]:
    """Generate a structured summary without LLM access when logs are absent or parsing fails."""

    normalized = text.strip()
    if not normalized:
        return {
            "summary": "No log data was provided.",
            "anomalies": [],
            "evidence": [],
        }

    anomalies: list[str] = []
    evidence: list[str] = []
    patterns = {
        "exceptions": r"(Exception|ERROR|Traceback|stack trace|FAILURE)",
        "http_4xx": r"\b4\d\d\b",
        "http_5xx": r"\b5\d\d\b",
        "timeouts": r"(timeout|timed out|deadline exceeded|ReadTimeout|Gateway Timeout)",
        "mongo_issues": r"(MongoDB|mongo.*(connect|connection|timeout|failed)|failed to connect to MongoDB)",
        "slow_requests": r"(slow request|slow query|SLOW|duration.*ms|took .*s)",
    }

    for label, pattern in patterns.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            anomalies.append(label.replace("_", " ").title())

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    for line in lines:
        if re.search(r"(Exception|Traceback|ERROR|Timeout|timed out|MongoDB|5\d\d|4\d\d)", line, flags=re.IGNORECASE):
            evidence.append(line[:400])

    if not anomalies:
        anomalies = ["No clear anomalies detected in the provided logs."]
    if not evidence:
        evidence = ["Logs were reviewed but no explicit error signatures were identified."]

    return {
        "summary": "Log review found operational anomalies that should be correlated with the ticket and runtime state.",
        "anomalies": anomalies[:10],
        "evidence": evidence[:10],
    }


def log_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Analyze app, nginx, and MongoDB logs and emit structured log findings."""

    started_at = time.perf_counter()

    state_errors = list(state.get("errors", []) or [])
    log_text = _collect_log_text(state)
    if not log_text.strip():
        errors = append_error(
            state,
            "log_agent",
            "No log files were uploaded or the provided logs were empty.",
            details="The workflow did not receive application, nginx, or MongoDB log data.",
            user_message="No log files were uploaded. Continuing with ticket-only analysis and limited evidence.",
        )
        trace = add_execution_trace(
            state,
            "log_agent",
            "Checked the available log inputs and found no usable log data, so the workflow continued without log evidence.",
            decision="fallback",
            started_at=started_at,
        )
        return {
            "log_analysis": {
                "summary": "No log data was provided.",
                "anomalies": [],
                "evidence": [],
            },
            "current_step": "log_agent",
            "errors": errors,
            **trace,
        }

    try:
        model = get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior observability analyst. Review the supplied logs and detect: exceptions, stack traces, HTTP 4xx, HTTP 5xx, timeout errors, MongoDB connection issues, and slow requests. Return compact JSON only with keys: summary, anomalies, evidence.",
                ),
                ("human", "Logs:\n{logs}"),
            ]
        )

        response = invoke_with_retry(model, prompt, {"logs": log_text}, "log_agent")
        content = getattr(response, "content", response)
        text = str(content).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            errors = append_error(
                state,
                "log_agent",
                "The model returned invalid log-analysis JSON.",
                details="The fallback summary was used because the response could not be parsed as JSON.",
                user_message="Log analysis output was incomplete. Continuing with the fallback summary.",
            )
            trace = add_execution_trace(
                state,
                "log_agent",
                "Reviewed the logs and used the fallback summary because the model response was not valid JSON.",
                decision="fallback",
                started_at=started_at,
            )
            return {
                "log_analysis": _fallback_summary(log_text),
                "current_step": "log_agent",
                "errors": errors,
                **trace,
            }

        normalized = {
            "summary": str(parsed.get("summary", "Log analysis completed."))[:800],
            "anomalies": parsed.get("anomalies", []) if isinstance(parsed.get("anomalies", []), list) else [str(parsed.get("anomalies", ""))],
            "evidence": parsed.get("evidence", []) if isinstance(parsed.get("evidence", []), list) else [str(parsed.get("evidence", ""))],
        }
        trace = add_execution_trace(
            state,
            "log_agent",
            "Reviewed application, nginx, and MongoDB logs and identified the main operational anomalies and evidence points.",
            started_at=started_at,
        )
        return {"log_analysis": normalized, "current_step": "log_agent", "errors": state_errors, **trace}
    except Exception as exc:
        fallback = _fallback_summary(log_text)
        errors = append_error(
            state,
            "log_agent",
            str(exc),
            details="Log analysis failed; the fallback summary was returned so the workflow can continue.",
            user_message="Log analysis could not complete. Continuing with the fallback summary.",
        )
        trace = add_execution_trace(
            state,
            "log_agent",
            "Encountered a log-analysis error and switched to the fallback summary path to keep the workflow moving.",
            decision="fallback",
            started_at=started_at,
        )
        return {
            "log_analysis": fallback,
            "current_step": "log_agent",
            "errors": errors,
            **trace,
        }


__all__ = ["log_agent"]
