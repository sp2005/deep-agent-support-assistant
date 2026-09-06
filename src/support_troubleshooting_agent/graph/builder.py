"""LangGraph workflow builder for the support troubleshooting pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from .state import SupportTroubleshootingState
from support_troubleshooting_agent.agents._resilience import add_execution_trace


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


def investigation_planner(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Plan investigation needs from the ticket, current state, and available logs."""

    ticket_summary = str(state.get("ticket_summary", "")).strip()
    log_analysis = state.get("log_analysis") or {}
    available_logs = " ".join(
        str(state.get(log_key, ""))
        for log_key in ("application_logs", "nginx_logs", "mongodb_logs")
        if state.get(log_key)
    )
    evidence = " ".join(
        [
            ticket_summary,
            available_logs,
            str(log_analysis.get("summary", "")) if isinstance(log_analysis, dict) else str(log_analysis),
            " ".join(str(item) for item in log_analysis.get("anomalies", [])) if isinstance(log_analysis, dict) else "",
            " ".join(str(item) for item in log_analysis.get("evidence", [])) if isinstance(log_analysis, dict) else "",
        ]
    ).lower()
    retrieval_signals = (
        "unknown",
        "unfamiliar",
        "deployment",
        "configuration",
        "config",
        "dependency",
        "database",
        "mongodb",
        "timeout",
        "504",
        "503",
        "502",
        "index",
        "search",
        "regression",
        "runbook",
    )
    sufficient_signals = (
        "confirmed root cause",
        "confirmed fix",
        "known issue",
        "resolved by",
        "health check passed",
    )
    evidence_is_sufficient = any(signal in evidence for signal in sufficient_signals)
    retrieval_needed = not evidence_is_sufficient and (not evidence or any(signal in evidence for signal in retrieval_signals))
    query = " ".join(part for part in [ticket_summary, available_logs, str(log_analysis)] if part).strip()
    decision = "retrieve_knowledge" if retrieval_needed else "diagnose_from_available_evidence"
    summary = (
        "Planner determined KB lookup required because the incident contains unfamiliar or dependency-related signals."
        if retrieval_needed
        else "Planner determined KB lookup was not required because the available ticket and log evidence was sufficient."
    )
    trace = add_execution_trace(state, "investigation_planner", summary, decision=decision)
    return {
        "retrieval_needed": retrieval_needed,
        "retrieval_query": query,
        "retrieval_attempts": 0,
        "retrieval_sufficient": not retrieval_needed,
        "investigation_decision": decision,
        "current_step": "investigation_planner",
        **trace,
    }


def rag_agent(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Node placeholder for retrieval and knowledge grounding."""

    result = _invoke_agent("rag_knowledge", "rag_agent", state)
    return {
        **result,
        "retrieval_attempts": int(state.get("retrieval_attempts", 0) or 0) + 1,
        "current_step": "rag_agent",
    }


def evaluate_retrieval(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Assess retrieval quality and decide whether to accept, retry, or skip it."""

    documents = state.get("retrieved_documents") or []
    scores = [float(document.get("score", 0.0) or 0.0) for document in documents if isinstance(document, dict)]
    sufficient = bool(documents) and max(scores, default=0.0) >= 0.45
    attempts = int(state.get("retrieval_attempts", 0) or 0)

    if sufficient:
        decision = "use_retrieved_knowledge"
        summary = "Retrieval evaluator found relevant knowledge-base evidence and routed the investigation to diagnosis."
    elif attempts < 2:
        decision = "refine_retrieval_query"
        summary = "Retrieval evaluator found insufficient evidence and routed the query for one refinement attempt."
    else:
        decision = "continue_without_kb_evidence"
        summary = "Retrieval evaluator found insufficient evidence after two attempts and continued without KB evidence."

    trace = add_execution_trace(state, "retrieval_evaluator", summary, decision=decision)
    return {
        "retrieved_documents": [] if decision == "continue_without_kb_evidence" else documents,
        "retrieval_sufficient": sufficient,
        "investigation_decision": decision,
        "current_step": "retrieval_evaluator",
        **trace,
    }


def refine_retrieval_query(state: SupportTroubleshootingState) -> dict[str, Any]:
    """Refine the retrieval query before the single allowed retry."""

    query = str(state.get("retrieval_query", "")).strip()
    log_analysis = state.get("log_analysis") or {}
    anomalies = log_analysis.get("anomalies", []) if isinstance(log_analysis, dict) else []
    refinement = " troubleshooting root cause remediation runbook"
    if anomalies:
        refinement += " " + " ".join(str(item) for item in anomalies[:3])
    refined_query = (query + refinement).strip()
    trace = add_execution_trace(
        state,
        "retrieval_refiner",
        "Refined the knowledge-base query with incident signals and routed it back to retrieval.",
        decision="retry_retrieval",
    )
    return {
        "retrieval_query": refined_query,
        "investigation_decision": "retry_retrieval",
        "current_step": "retrieval_refiner",
        **trace,
    }


def route_after_log(state: SupportTroubleshootingState) -> str:
    """Route to retrieval only when the planner requests knowledge lookup."""

    return "rag_agent" if state.get("retrieval_needed") else "diagnosis_agent"


def route_after_retrieval(state: SupportTroubleshootingState) -> str:
    """Route retrieval results to diagnosis, retry, or evidence-limited diagnosis."""

    if state.get("retrieval_sufficient"):
        return "diagnosis_agent"
    if int(state.get("retrieval_attempts", 0) or 0) < 2:
        return "retrieval_refiner"
    return "diagnosis_agent"


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
    workflow.add_node("investigation_planner", investigation_planner)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("retrieval_evaluator", evaluate_retrieval)
    workflow.add_node("retrieval_refiner", refine_retrieval_query)
    workflow.add_node("diagnosis_agent", diagnosis_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)
    workflow.add_node("report_agent", report_agent)
    workflow.add_node("human_review_agent", human_review_agent)

    workflow.set_entry_point("ticket_agent")

    workflow.add_edge("ticket_agent", "investigation_planner")
    workflow.add_edge("investigation_planner", "log_agent")
    workflow.add_conditional_edges(
        "log_agent",
        route_after_log,
        {"rag_agent": "rag_agent", "diagnosis_agent": "diagnosis_agent"},
    )
    workflow.add_edge("rag_agent", "retrieval_evaluator")
    workflow.add_conditional_edges(
        "retrieval_evaluator",
        route_after_retrieval,
        {"diagnosis_agent": "diagnosis_agent", "retrieval_refiner": "retrieval_refiner"},
    )
    workflow.add_edge("retrieval_refiner", "rag_agent")
    workflow.add_edge("diagnosis_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", "report_agent")
    workflow.add_edge("report_agent", "human_review_agent")
    workflow.add_edge("human_review_agent", END)

    return workflow.compile()


__all__ = [
    "build_workflow",
    "diagnosis_agent",
    "evaluate_retrieval",
    "human_review_agent",
    "investigation_planner",
    "log_agent",
    "rag_agent",
    "recommendation_agent",
    "report_agent",
    "refine_retrieval_query",
    "route_after_log",
    "route_after_retrieval",
    "ticket_agent",
]
