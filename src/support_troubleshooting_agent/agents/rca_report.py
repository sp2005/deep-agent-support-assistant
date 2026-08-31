"""Final RCA report synthesis agent for the troubleshooting workflow."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from support_troubleshooting_agent.agents._resilience import add_execution_trace, append_error, invoke_with_retry
from support_troubleshooting_agent.models.llm_factory import get_chat_model


def _coerce_text(value: Any) -> str:
    """Normalize values into a text representation for the model."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _fallback_report(state: dict[str, Any]) -> dict[str, Any]:
    """Provide a structured RCA summary when the LLM is unavailable or malformed."""

    ticket_summary = _coerce_text(state.get("ticket_summary"))
    log_analysis = _coerce_text(state.get("log_analysis"))
    root_cause = _coerce_text(state.get("root_cause"))
    recommendations = _coerce_text(state.get("recommendations"))

    title = "Incident RCA Report"
    summary = "The incident was caused by a degradation in the system path evidenced by the support ticket, log anomalies, and root-cause analysis. Immediate remediation steps and preventative measures are documented below."

    return {
        "title": title,
        "summary": summary,
        "root_cause": root_cause or "Root cause not yet conclusively determined.",
        "actions": [
            "Apply the recommended containment and mitigation steps.",
            "Validate the fix against the affected workflow and customer impact.",
        ],
        "evidence": [
            ticket_summary or "Ticket summary unavailable.",
            log_analysis or "Log evidence unavailable.",
            root_cause or "Root cause evidence unavailable.",
            recommendations or "Recommendations unavailable.",
        ],
        "status": "Mitigation in progress",
    }


def report_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Generate the final RCA incident report from all prior workflow evidence."""

    started_at = time.perf_counter()

    ticket_summary = state.get("ticket_summary")
    log_analysis = state.get("log_analysis")
    retrieved_documents = state.get("retrieved_documents")
    root_cause = state.get("root_cause")
    recommendations = state.get("recommendations")

    if not any([ticket_summary, log_analysis, retrieved_documents, root_cause, recommendations]):
        fallback = _fallback_report(state)
        errors = append_error(
            state,
            "report_agent",
            "No investigation evidence was available for the RCA report.",
            details="The final report was generated from the fallback template because the workflow did not have enough data.",
            user_message="Not enough investigation data was available to produce a full report. A fallback report has been generated.",
        )
        trace = add_execution_trace(
            state,
            "report_agent",
            "Checked the collected evidence and generated the fallback RCA because the workflow lacked enough data.",
            decision="fallback",
            started_at=started_at,
        )
        return {"rca_report": fallback, "current_step": "report_agent", "errors": errors, **trace}

    try:
        model = get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior incident commander. Synthesize the full support investigation into a polished professional RCA report. "
                    "Return compact JSON only with keys: title, summary, root_cause, actions, evidence, status. "
                    "The `summary` should be a concise executive summary. `actions` should be a list of action items; `evidence` should be a list of strings."
                ),
                (
                    "human",
                    "Ticket Summary:\n{ticket_summary}\n\nLog Analysis:\n{log_analysis}\n\nRetrieved Documents:\n{retrieved_documents}\n\nRoot Cause:\n{root_cause}\n\nRecommendations:\n{recommendations}",
                ),
            ]
        )

        response = invoke_with_retry(
            model,
            prompt,
            {
                "ticket_summary": _coerce_text(ticket_summary),
                "log_analysis": _coerce_text(log_analysis),
                "retrieved_documents": _coerce_text(retrieved_documents),
                "root_cause": _coerce_text(root_cause),
                "recommendations": _coerce_text(recommendations),
            },
            "report_agent",
        )
        content = getattr(response, "content", response)
        text = str(content).strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            errors = append_error(
                state,
                "report_agent",
                "The model returned invalid RCA JSON.",
                details="The fallback report was used because the response could not be parsed.",
                user_message="The final RCA output was incomplete. A fallback report has been generated.",
            )
            trace = add_execution_trace(
                state,
                "report_agent",
                "Synthesized the final RCA and used the fallback report after the model output was invalid.",
                decision="fallback",
                started_at=started_at,
            )
            return {"rca_report": _fallback_report(state), "current_step": "report_agent", "errors": errors, **trace}

        normalized = {
            "title": str(parsed.get("title", "Incident RCA Report")),
            "summary": str(parsed.get("summary", "The incident was investigated and a mitigation plan was defined.")),
            "root_cause": str(parsed.get("root_cause", "Root cause not conclusively established.")),
            "actions": parsed.get("actions", []) if isinstance(parsed.get("actions", []), list) else [str(parsed.get("actions", ""))],
            "evidence": parsed.get("evidence", []) if isinstance(parsed.get("evidence", []), list) else [str(parsed.get("evidence", ""))],
            "status": str(parsed.get("status", "Mitigation in progress")),
        }
        trace = add_execution_trace(
            state,
            "report_agent",
            "Combined the full investigation evidence into the final RCA summary, action plan, and evidence list.",
            started_at=started_at,
        )
        return {"rca_report": normalized, "current_step": "report_agent", **trace}
    except Exception as exc:
        fallback = _fallback_report(state)
        errors = append_error(
            state,
            "report_agent",
            str(exc),
            details="Final RCA synthesis failed; fallback payload used so the workflow can continue.",
            user_message="The final RCA report could not be generated. A fallback report has been created.",
        )
        trace = add_execution_trace(
            state,
            "report_agent",
            "Encountered an RCA-generation failure and used the fallback final report path to keep the workflow moving.",
            decision="fallback",
            started_at=started_at,
        )
        return {
            "rca_report": fallback,
            "current_step": "report_agent",
            "errors": errors,
            **trace,
        }


__all__ = ["report_agent"]
