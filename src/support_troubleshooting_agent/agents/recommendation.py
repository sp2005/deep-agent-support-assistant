"""Recommendation agent that turns the root cause into actionable response guidance."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from support_troubleshooting_agent.agents._resilience import add_execution_trace, append_error, invoke_with_retry
from support_troubleshooting_agent.models.llm_factory import get_chat_model


def _coerce_text(value: Any) -> str:
    """Normalize a value to a single text payload."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _fallback_recommendations(root_cause: Any) -> dict[str, list[str]]:
    """Structured recommendation payload used when the LLM cannot provide a valid response."""

    cause = _coerce_text(root_cause)
    if not cause:
        cause = "No root cause evidence was supplied."

    return {
        "recommended_actions": [
            "Review the most recent deployment and identify any database or dependency changes that could affect the checkout path.",
            "Confirm the root-cause evidence in the application and database logs before making a production rollback.",
        ],
        "validation_steps": [
            "Verify that error rates fall back to baseline after the mitigation.",
            "Run focused smoke tests against the checkout workflow and database queries.",
        ],
        "escalation_guidance": [
            "Escalate to the database owner or platform engineering if database latency remains elevated after mitigation.",
            "Notify the incident commander if customer impact exceeds the defined SLO threshold.",
        ],
        "preventative_measures": [
            "Add alerting for checkout latency and database slow queries at the application layer.",
            "Document the deployment change and add a pre-deploy validation checklist for database regressions.",
        ],
    }


def recommendation_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Generate actionable recommendations from the root cause analysis."""

    started_at = time.perf_counter()

    root_cause = state.get("root_cause")
    if root_cause is None:
        errors = append_error(
            state,
            "recommendation_agent",
            "No root-cause data was available for recommendation generation.",
            details="The workflow continued without a root-cause input, so fallback recommendations were used.",
            user_message="No root-cause analysis was available. Continuing with fallback guidance.",
        )
        trace = add_execution_trace(
            state,
            "recommendation_agent",
            "Checked for root-cause evidence and used fallback guidance because the diagnosis was missing.",
            decision="fallback",
            started_at=started_at,
        )
        return {"recommendations": _fallback_recommendations({}), "current_step": "recommendation_agent", "errors": errors, **trace}

    try:
        model = get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior incident response engineer. Based on the root cause, generate concise but actionable responses. "
                    "Return JSON only with keys: recommended_actions, validation_steps, escalation_guidance, preventative_measures. "
                    "Each value must be a list of short strings.",
                ),
                ("human", "Root Cause:\n{root_cause}"),
            ]
        )

        response = invoke_with_retry(model, prompt, {"root_cause": _coerce_text(root_cause)}, "recommendation_agent")
        content = getattr(response, "content", response)
        text = str(content).strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            errors = append_error(
                state,
                "recommendation_agent",
                "The model returned invalid recommendation JSON.",
                details="Fallback recommendation guidance was used because the LLM response could not be parsed.",
                user_message="Recommendation output was incomplete. Continuing with fallback guidance.",
            )
            trace = add_execution_trace(
                state,
                "recommendation_agent",
                "Converted the root cause into mitigation guidance and used the fallback recommendations after parsing failed.",
                decision="fallback",
                started_at=started_at,
            )
            return {"recommendations": _fallback_recommendations(root_cause), "current_step": "recommendation_agent", "errors": errors, **trace}

        normalized = {
            "recommended_actions": parsed.get("recommended_actions", []) if isinstance(parsed.get("recommended_actions", []), list) else [str(parsed.get("recommended_actions", ""))],
            "validation_steps": parsed.get("validation_steps", []) if isinstance(parsed.get("validation_steps", []), list) else [str(parsed.get("validation_steps", ""))],
            "escalation_guidance": parsed.get("escalation_guidance", []) if isinstance(parsed.get("escalation_guidance", []), list) else [str(parsed.get("escalation_guidance", ""))],
            "preventative_measures": parsed.get("preventative_measures", []) if isinstance(parsed.get("preventative_measures", []), list) else [str(parsed.get("preventative_measures", ""))],
        }
        trace = add_execution_trace(
            state,
            "recommendation_agent",
            "Used the root-cause evidence to create mitigation, validation, escalation, and preventative guidance.",
            started_at=started_at,
        )
        return {"recommendations": normalized, "current_step": "recommendation_agent", **trace}
    except Exception as exc:
        errors = append_error(
            state,
            "recommendation_agent",
            str(exc),
            details="Recommendation generation failed; the fallback guidance was used so the workflow could continue.",
            user_message="Recommendations could not be generated. Continuing with fallback guidance.",
        )
        trace = add_execution_trace(
            state,
            "recommendation_agent",
            "Encountered a recommendation-generation error and used the fallback guidance path to maintain continuity.",
            decision="fallback",
            started_at=started_at,
        )
        return {
            "recommendations": _fallback_recommendations(root_cause),
            "current_step": "recommendation_agent",
            "errors": errors,
            **trace,
        }


__all__ = ["recommendation_agent"]
