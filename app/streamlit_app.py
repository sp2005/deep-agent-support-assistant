"""Streamlit UI for the support troubleshooting workflow."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from support_troubleshooting_agent.graph.builder import build_workflow, human_review_agent
from support_troubleshooting_agent.models.llm_factory import get_model_configuration


st.set_page_config(page_title="AI Support Investigation", page_icon="🛠️", layout="wide")


def _safe_dict(value: Any) -> dict[str, Any]:
    """Normalize a value into a dictionary when possible."""

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"summary": value}
        except json.JSONDecodeError:
            return {"summary": value}
    return {"summary": str(value)}


def _render_json_block(title: str, content: Any, empty_message: str) -> None:
    """Render a structured result block with a fallback empty state."""

    st.subheader(title)
    if content is None:
        st.info(empty_message)
        return
    if isinstance(content, (list, dict)) and not content:
        st.info(empty_message)
        return
    st.json(content)


def _render_documents(documents: list[dict[str, Any]]) -> None:
    """Render retrieved knowledge documents in a professional card-like format."""

    st.subheader("Retrieved Knowledge")
    if not documents:
        st.info("No relevant documents were returned for this ticket.")
        return

    for index, doc in enumerate(documents[:5], start=1):
        score = doc.get("score")
        with st.container():
            st.markdown(f"### {index}. {doc.get('title', 'Untitled document')}")
            st.caption(f"Source: {doc.get('source', 'Unknown')} | Relevance: {score if score is not None else 'n/a'}")
            st.write(doc.get("content", ""))
            st.markdown("---")


def _render_error_panel(errors: list[dict[str, Any]]) -> None:
    """Render collected workflow errors in a user-friendly panel."""

    if not errors:
        return

    st.subheader("Workflow Errors")
    for error in errors:
        message = error.get("user_message") or error.get("message") or "An error occurred."
        details = error.get("details")
        st.warning(message)
        if details:
            st.caption(details)


def _render_execution_trace(trace: list[dict[str, Any]]) -> None:
    """Render a concise workflow trace without showing raw chain-of-thought content."""

    st.subheader("Workflow Trace")
    if not trace:
        st.info("No workflow execution trace is available yet.")
        return

    for entry in trace:
        agent = entry.get("agent", "unknown_agent")
        decision = entry.get("decision", "completed")
        summary = entry.get("summary", "No observable summary was recorded.")
        elapsed = entry.get("execution_time")
        st.markdown(f"**{agent}** — {decision}")
        st.write(summary)
        if elapsed is not None:
            st.caption(f"Elapsed: {float(elapsed):.3f}s")
        st.markdown("---")


def _run_workflow_with_progress(state: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Execute the graph with status updates and return the current state plus whether a review pause was triggered."""

    workflow = build_workflow()
    current_state = dict(state)
    pending_review = False
    status = st.status("Investigating incident...", expanded=True)
    progress_bar = st.progress(0, text="Starting investigation")

    steps = [
        "ticket_agent",
        "log_agent",
        "rag_agent",
        "diagnosis_agent",
        "recommendation_agent",
        "report_agent",
        "human_review_agent",
    ]

    for index, event in enumerate(workflow.stream(current_state, stream_mode="updates"), start=1):
        if "__interrupt__" in event:
            pending_review = True
            status.update(label="Paused for RCA review", state="complete")
            st.warning("The workflow paused for human review before the final RCA report was finalized.")
            progress_bar.progress(1.0, text="Waiting for human review")
            break

        for step_name, payload in event.items():
            if step_name == "__end__":
                continue
            if isinstance(payload, dict):
                current_state.update(payload)
            display_name = step_name.replace("_agent", "").replace("_", " ").title()
            status.write(f"Completed: {display_name}")

        progress_bar.progress(min(index / len(steps), 1.0), text=f"Running step {min(index, len(steps))}/{len(steps)}")

    return current_state, pending_review


def _apply_review_action(state: dict[str, Any], decision: str, reason: str) -> dict[str, Any]:
    """Apply a human review decision and return the updated workflow state."""

    state["approval_decision"] = decision
    state["approval_reason"] = reason
    state.update(human_review_agent(state))
    return state


def _clear_demo_state() -> None:
    """Clear the current results and force fresh input widgets for the next demo."""

    st.session_state.pop("workflow_state", None)
    st.session_state.pop("pending_review", None)
    st.session_state["input_reset_id"] = st.session_state.get("input_reset_id", 0) + 1


def main() -> None:
    """Render the support investigation interface."""

    st.title("AI Support Investigation")
    st.caption("Paste a ticket or upload supporting records, then run the troubleshooting workflow to investigate the incident.")
    provider_name, model_name = get_model_configuration()
    st.info(f"Configured model: **{provider_name} / {model_name}**")
    input_reset_id = st.session_state.get("input_reset_id", 0)

    with st.sidebar:
        st.header("Inputs")
        ticket_file = st.file_uploader(
            "Upload support ticket",
            type=["json", "txt", "md"],
            key=f"ticket_file_{input_reset_id}",
        )
        with st.expander("Paste ticket instead", expanded=False):
            ticket_text = st.text_area(
                "Support ticket text",
                height=110,
                placeholder="Paste the customer issue or incident summary here...",
                key=f"ticket_text_{input_reset_id}",
            )
        with st.expander("Optional log files", expanded=True):
            app_logs = st.file_uploader(
                "Application logs",
                type=["log", "txt", "csv"],
                key=f"app_logs_{input_reset_id}",
            )
            nginx_logs = st.file_uploader(
                "Nginx logs",
                type=["log", "txt", "csv"],
                key=f"nginx_logs_{input_reset_id}",
            )
            mongodb_logs = st.file_uploader(
                "MongoDB logs",
                type=["log", "txt", "csv"],
                key=f"mongodb_logs_{input_reset_id}",
            )
        investigate = st.button("Investigate", type="primary")
        clear_results = st.button("Clear results / New demo")

    if clear_results:
        _clear_demo_state()
        st.rerun()

    if investigate:
        ticket_value = ticket_text.strip()
        if ticket_file is not None:
            ticket_value = ticket_file.read().decode("utf-8", errors="replace")

        if not ticket_value:
            st.warning("Please provide a support ticket before running the investigation.")
            return

        initial_state: dict[str, Any] = {"ticket": {"ticket_text": ticket_value}}

        if app_logs is not None:
            initial_state["application_logs"] = app_logs.read().decode("utf-8", errors="replace")
        if nginx_logs is not None:
            initial_state["nginx_logs"] = nginx_logs.read().decode("utf-8", errors="replace")
        if mongodb_logs is not None:
            initial_state["mongodb_logs"] = mongodb_logs.read().decode("utf-8", errors="replace")

        st.subheader("Incident Inputs")
        st.json({
            "ticket_preview": ticket_value[:400],
            "app_logs_uploaded": app_logs is not None,
            "nginx_logs_uploaded": nginx_logs is not None,
            "mongodb_logs_uploaded": mongodb_logs is not None,
        })

        final_state, pending_review = _run_workflow_with_progress(initial_state)
        st.session_state["workflow_state"] = final_state
        st.session_state["pending_review"] = pending_review
    elif "workflow_state" not in st.session_state:
        st.info("Provide a ticket and relevant logs, then click Investigate to begin the review.")
        return

    final_state = st.session_state["workflow_state"]
    pending_review = st.session_state.get("pending_review", False)

    errors = final_state.get("errors") or []
    current_agent = final_state.get("current_agent")
    execution_time = final_state.get("execution_time")
    reasoning_summary = final_state.get("reasoning_summary") or []

    if current_agent or execution_time is not None:
        st.subheader("Execution Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current agent", str(current_agent or "pending"))
        with col2:
            st.metric("Total execution time", f"{float(execution_time or 0.0):.3f}s")

    _render_execution_trace(reasoning_summary)

    if errors:
        _render_error_panel(errors)

    ticket_summary = _safe_dict(final_state.get("ticket_summary"))
    log_analysis = final_state.get("log_analysis")
    retrieved_documents = final_state.get("retrieved_documents") or []
    root_cause = final_state.get("root_cause")
    recommendations = final_state.get("recommendations")
    rca_report = final_state.get("rca_report")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        _render_json_block("Ticket Summary", ticket_summary, "No ticket summary was generated.")
    with col2:
        _render_json_block("Root Cause", root_cause, "No root cause was determined.")

    _render_documents(retrieved_documents)

    _render_json_block("Log Analysis", log_analysis, "No log analysis was produced.")

    _render_json_block("Recommendations", recommendations, "No recommendations were generated.")
    _render_json_block("RCA Report", rca_report, "No RCA report was generated.")

    if pending_review:
        st.markdown("---")
        st.subheader("Human Review")
        st.info("Review the final RCA before it is finalized.")
        review_reason = st.text_input("Review note", placeholder="Optional reason for approve, revise, or cancel")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Approve", width="stretch"):
                _apply_review_action(final_state, "approve", review_reason or "Approved in the UI.")
                st.session_state["workflow_state"] = final_state
                st.session_state["pending_review"] = False
                st.success("RCA approved and finalized.")
                st.json(final_state.get("rca_report"))
        with col2:
            if st.button("Revise", width="stretch"):
                _apply_review_action(final_state, "revise", review_reason or "Requested revision before finalization.")
                st.session_state["workflow_state"] = final_state
                st.session_state["pending_review"] = False
                st.warning("RCA revision requested.")
                st.json(final_state.get("rca_report"))
        with col3:
            if st.button("Cancel", width="stretch"):
                _apply_review_action(final_state, "cancel", review_reason or "Cancelled in the UI.")
                st.session_state["workflow_state"] = final_state
                st.session_state["pending_review"] = False
                st.error("RCA workflow cancelled.")
                st.json(final_state.get("rca_report"))


if __name__ == "__main__":
    main()
