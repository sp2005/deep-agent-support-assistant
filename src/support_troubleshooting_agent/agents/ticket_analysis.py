"""Ticket analysis agent that summarizes incoming support tickets into concise JSON."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping

from langchain_core.prompts import ChatPromptTemplate

from support_troubleshooting_agent.agents._resilience import add_execution_trace, append_error, invoke_with_retry
from support_troubleshooting_agent.models.llm_factory import get_chat_model


def _coerce_ticket(ticket: Any) -> str:
    """Normalize a ticket payload into a text block for the model."""

    if isinstance(ticket, str):
        return ticket.strip()
    if isinstance(ticket, Mapping):
        return json.dumps(ticket, ensure_ascii=False, default=str)
    return str(ticket)


def _parse_summary_response(content: Any) -> dict[str, Any]:
    """Parse the model output into a JSON document, with a safe fallback."""

    if isinstance(content, (dict, list)):
        return content if isinstance(content, dict) else {"summary": json.dumps(content, ensure_ascii=False)}

    text = str(content).strip()
    if not text:
        return {
            "summary": "No ticket summary available.",
            "priority": "P4",
            "issue_type": "unknown",
            "customer_impact": "unknown",
            "key_facts": [],
        }

    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return {
        "summary": cleaned[:800],
        "priority": "P4",
        "issue_type": "unknown",
        "customer_impact": "unknown",
        "key_facts": [],
    }


def ticket_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Analyze the incoming support ticket and return a concise JSON summary."""

    started_at = time.perf_counter()

    ticket = state.get("ticket")
    if ticket is None:
        fallback = {
            "summary": "No ticket payload was provided.",
            "priority": "P4",
            "issue_type": "unknown",
            "customer_impact": "unknown",
            "key_facts": [],
        }
        errors = append_error(
            state,
            "ticket_agent",
            "No support ticket was provided.",
            details="The workflow cannot analyze a ticket without the original ticket payload.",
            user_message="No support ticket was uploaded. Continuing with a minimal placeholder summary.",
        )
        trace = add_execution_trace(
            state,
            "ticket_agent",
            "Reviewed the request and applied a safe fallback because no support ticket was provided.",
            decision="fallback",
            started_at=started_at,
        )
        return {"ticket_summary": json.dumps(fallback, ensure_ascii=False), "current_step": "ticket_agent", "errors": errors, **trace}

    try:
        model = get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior support triage assistant. Analyze the support ticket and return concise JSON only. "
                    "Use keys: summary, priority, issue_type, customer_impact, key_facts. "
                    "priority must be one of P1, P2, P3, P4. "
                    "summary should be 1-2 sentences max.",
                ),
                ("human", "Ticket:\n{ticket}"),
            ]
        )

        response = invoke_with_retry(model, prompt, {"ticket": _coerce_ticket(ticket)}, "ticket_agent")
        payload = _parse_summary_response(getattr(response, "content", response))
        summary = {
            "summary": str(payload.get("summary", "Ticket analysis unavailable."))[:800],
            "priority": str(payload.get("priority", "P4")).upper(),
            "issue_type": str(payload.get("issue_type", "unknown")),
            "customer_impact": str(payload.get("customer_impact", "unknown")),
            "key_facts": payload.get("key_facts", []) if isinstance(payload.get("key_facts", []), list) else [str(payload.get("key_facts", ""))],
        }
        trace = add_execution_trace(
            state,
            "ticket_agent",
            "Reviewed the ticket payload and produced a structured triage summary and priority classification.",
            started_at=started_at,
        )
        return {"ticket_summary": json.dumps(summary, ensure_ascii=False), "current_step": "ticket_agent", **trace}
    except Exception as exc:
        fallback = {
            "summary": "Ticket analysis unavailable due to an internal error.",
            "priority": "P4",
            "issue_type": "unknown",
            "customer_impact": "unknown",
            "key_facts": [],
        }
        errors = append_error(
            state,
            "ticket_agent",
            str(exc),
            details="Failed to generate ticket summary; a minimal fallback summary was returned.",
            user_message="Ticket analysis could not complete. Continuing with a safe fallback summary.",
        )
        trace = add_execution_trace(
            state,
            "ticket_agent",
            "Encountered an error while summarizing the ticket and applied the fallback summary path.",
            decision="fallback",
            started_at=started_at,
        )
        return {
            "ticket_summary": json.dumps(fallback, ensure_ascii=False),
            "current_step": "ticket_agent",
            "errors": errors,
            **trace,
        }


__all__ = ["ticket_agent"]
