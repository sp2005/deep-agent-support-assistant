"""LangGraph workflow builder for the support troubleshooting pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from .state import SupportTroubleshootingState


def _set_step(state: SupportTroubleshootingState, step_name: str) -> dict[str, Any]:
    """Update the current workflow step without adding any domain logic yet."""

    trace = {
        "current_agent": step_name,
        "current_step": step_name,
        "completed_steps": list(state.get("completed_steps", []) or []) + ([step_name] if step_name not in state.get("completed_steps", []) else []),
        "reasoning_summary": list(state.get("reasoning_summary", []) or []) + [{"agent": step_name, "decision": "advance", "summary": f"Advanced the workflow to {step_name}."}],
        "execution_time": float(state.get("execution_time", 0.0) or 0.0),
    }
    return trace


def _invoke_agent(module_name: str, function_name: str, state: SupportTroubleshootingState) -> dict[str, Any]:
    """Attempt to call an existing agent implementation if it is available.

    If the agent module is not implemented yet, the workflow still remains valid by
    updating the current step in the shared state.
    """

    try:
        module = import_module(f"support_troubleshooting_agent.agents.{module_name}")
    except ImportError:
        return _set_step(state, function_name)

    agent_fn = getattr(module, function_name, None)
    if agent_fn is None:
        agent_fn = getattr(module, f"{function_name}_agent", None)

    if callable(agent_fn):
        result = agent_fn(state)
        if isinstance(result, dict):
            return result

    return _set_step(state, function_name)


def ticket_agent(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Node placeholder for ticket intake and summarization."""

    return _invoke_agent("ticket_analysis", "ticket_agent", state)


def log_agent(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Node placeholder for log analysis."""

    return _invoke_agent("log_analysis", "log_agent", state)


def rag_agent(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Node placeholder for retrieval and knowledge grounding."""

    return _invoke_agent("rag_knowledge", "rag_agent", state)


def diagnosis_agent(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Node placeholder for diagnosis and root-cause analysis."""

    return _invoke_agent("root_cause", "diagnosis_agent", state)


def recommendation_agent(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Node placeholder for recommendation generation."""

    return _invoke_agent("recommendation", "recommendation_agent", state)


def report_agent(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Node placeholder for final RCA report generation."""

    return _invoke_agent("rca_report", "report_agent", state)


def human_review_agent(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Pause before finalizing the RCA and require a human decision.

    Supported actions are: approve, revise, cancel.
    """

    decision = state.get("approval_decision")
    reason = state.get("approval_reason")
    report = state.get("rca_report", {}) or {}

    if decision:
        normalized = str(decision).lower().strip()
        if normalized == "approve":
            report["status"] = "Approved"
            return {
                "approval_status": "approved",
                "approval_decision": "approve",
                "approval_reason": reason or "Approved by human reviewer.",
                "rca_report": report,
                "current_step": "human_review_agent",
            }
        if normalized == "revise":
            report["status"] = "Revision requested"
            return {
                "approval_status": "revision_requested",
                "approval_decision": "revise",
                "approval_reason": reason or "Requested revision before finalization.",
                "rca_report": report,
                "current_step": "human_review_agent",
            }
        report["status"] = "Cancelled"
        return {
            "approval_status": "cancelled",
            "approval_decision": "cancel",
            "approval_reason": reason or "Cancelled by human reviewer.",
            "rca_report": report,
            "current_step": "human_review_agent",
        }

    review_payload = {
        "type": "rca_review",
        "message": "Review the generated RCA report before finalizing it.",
        "allowed_actions": ["approve", "revise", "cancel"],
        "report": report,
    }
    resume = interrupt(review_payload)

    decision = str((resume or {}).get("decision", "cancel")).lower().strip()
    reason = str((resume or {}).get("reason", "No reason provided.")).strip()

    if decision == "approve":
        report["status"] = "Approved"
        return {
            "approval_status": "approved",
            "approval_decision": "approve",
            "approval_reason": reason,
            "rca_report": report,
            "current_step": "human_review_agent",
        }
    if decision == "revise":
        report["status"] = "Revision requested"
        return {
            "approval_status": "revision_requested",
            "approval_decision": "revise",
            "approval_reason": reason,
            "rca_report": report,
            "current_step": "human_review_agent",
        }

    report["status"] = "Cancelled"
    return {
        "approval_status": "cancelled",
        "approval_decision": "cancel",
        "approval_reason": reason,
        "rca_report": report,
        "current_step": "human_review_agent",
    }


def build_workflow() -> CompiledStateGraph:
    """Construct and compile the ordered troubleshooting graph.

    The compiled graph runs investigation agents in sequence and pauses at the
    human review node before the final RCA is approved or cancelled.
    """

    workflow = StateGraph(SupportTroubleshootingState)

    workflow.add_node("ticket_agent", ticket_agent)
    workflow.add_node("log_agent", log_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("diagnosis_agent", diagnosis_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)
    workflow.add_node("report_agent", report_agent)
    workflow.add_node("human_review_agent", human_review_agent)

    workflow.set_entry_point("ticket_agent")

    workflow.add_edge("ticket_agent", "log_agent")
    workflow.add_edge("log_agent", "rag_agent")
    workflow.add_edge("rag_agent", "diagnosis_agent")
    workflow.add_edge("diagnosis_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", "report_agent")
    workflow.add_edge("report_agent", "human_review_agent")
    workflow.add_edge("human_review_agent", END)

    return workflow.compile()


__all__ = [
    "build_workflow",
    "diagnosis_agent",
    "human_review_agent",
    "log_agent",
    "rag_agent",
    "recommendation_agent",
    "report_agent",
    "ticket_agent",
]
