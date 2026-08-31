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


def _render_list(items: Any, empty_message: str = "No items were recorded.") -> None:
    """Render a list of human-readable bullet points."""

    values = items if isinstance(items, list) else [items] if items else []
    if not values:
        st.caption(empty_message)
        return
    for item in values:
        st.markdown(f"- {item}")


def _render_ticket_summary(summary: dict[str, Any]) -> None:
    """Render ticket triage fields as readable metadata and key facts."""

    st.subheader("Ticket Summary")
    with st.container(border=True):
        st.write(summary.get("summary", "No ticket summary was generated."))
        metadata = st.columns(3)
        metadata[0].metric("Priority", summary.get("priority", "Unknown"))
        metadata[1].metric("Issue type", summary.get("issue_type", "Unknown"))
        metadata[2].metric("Customer impact", summary.get("customer_impact", "Unknown"))
        with st.expander("Key facts", expanded=True):
            _render_list(summary.get("key_facts"), "No key facts were extracted.")


def _render_log_analysis(log_analysis: Any) -> None:
    """Render detected log signals without exposing the raw log payload."""

    st.subheader("Log Analysis")
    if not log_analysis:
        st.info("No log analysis was produced.")
        return
    with st.container(border=True):
        st.write(log_analysis.get("summary", "Log analysis completed."))
        anomaly_col, evidence_col = st.columns(2)
        with anomaly_col:
            st.markdown("**Detected signals**")
            _render_list(log_analysis.get("anomalies"), "No clear anomalies detected.")
        with evidence_col:
            st.markdown("**Supporting evidence**")
            _render_list(log_analysis.get("evidence"), "No explicit evidence lines were recorded.")


def _render_root_cause(root_cause: Any) -> None:
    """Render the root-cause conclusion and evidence, excluding private reasoning text."""

    st.subheader("Root Cause")
    if not root_cause:
        st.info("No root cause was determined.")
        return
    with st.container(border=True):
        st.markdown(f"**{root_cause.get('primary_cause', 'Root cause not conclusively established.')}**")
        confidence = float(root_cause.get("confidence", 0.0) or 0.0)
        st.progress(max(0.0, min(confidence, 1.0)), text=f"Confidence: {confidence:.0%}")
        with st.expander("Supporting evidence", expanded=True):
            _render_list(root_cause.get("supporting_evidence"), "No supporting evidence was recorded.")


def _render_recommendations(recommendations: Any) -> None:
    """Render recommendation categories as action-oriented sections."""

    st.subheader("Recommendations")
    if not recommendations:
        st.info("No recommendations were generated.")
        return
    labels = {
        "recommended_actions": "Recommended actions",
        "validation_steps": "Validation steps",
        "escalation_guidance": "Escalation guidance",
        "preventative_measures": "Preventative measures",
    }
    with st.container(border=True):
        for key, label in labels.items():
            with st.expander(label, expanded=key == "recommended_actions"):
                _render_list(recommendations.get(key), "No items were recorded.")


def _render_rca_report(report: Any) -> None:
    """Render the final RCA as an executive summary, actions, and evidence."""

    st.subheader("RCA Report")
    if not report:
        st.info("No RCA report was generated.")
        return
    with st.container(border=True):
        st.markdown(f"### {report.get('title', 'Incident RCA Report')}")
        st.badge(report.get("status", "In progress"), icon=":material/flag:")
        st.write(report.get("summary", "No executive summary was generated."))
        with st.expander("Root cause", expanded=True):
            st.write(report.get("root_cause", "Root cause not conclusively established."))
        with st.expander("Action plan", expanded=True):
            _render_list(report.get("actions"), "No action items were recorded.")
        with st.expander("Evidence"):
            _render_list(report.get("evidence"), "No evidence was recorded.")


def _render_agent_status(completed_steps: list[str], current_agent: str | None) -> None:
    """Render progress badges for the workflow's observable agent stages."""

    agent_labels = [
        ("ticket_agent", "Ticket Analysis"),
        ("log_agent", "Log Analysis"),
        ("rag_agent", "RAG Knowledge"),
        ("diagnosis_agent", "Root Cause"),
        ("recommendation_agent", "Recommendation"),
        ("report_agent", "RCA Report"),
    ]
    st.markdown("**Agent progress**")
    badges = []
    for agent_name, label in agent_labels:
        if agent_name in completed_steps:
            badges.append(f":green-badge[✓ {label}]")
        elif agent_name == current_agent:
            badges.append(f":orange-badge[● {label}]")
        else:
            badges.append(f":gray-badge[○ {label}]")
    st.markdown(" ".join(badges))


def _render_input_summary(ticket_value: str, app_logs: Any, nginx_logs: Any, mongodb_logs: Any) -> None:
    """Render uploaded input metadata without displaying raw ticket or log contents."""

    st.subheader("Incident inputs")
    with st.container(border=True):
        st.caption(f"Ticket loaded: {len(ticket_value):,} characters")
        input_cols = st.columns(3)
        for column, label, uploaded in [
            (input_cols[0], "Application logs", app_logs),
            (input_cols[1], "Nginx logs", nginx_logs),
            (input_cols[2], "MongoDB logs", mongodb_logs),
        ]:
            with column:
                if uploaded is not None:
                    st.badge("Uploaded", icon=":material/check:", color="green")
                    st.caption(uploaded.name)
                else:
                    st.badge("Not uploaded", icon=":material/remove:", color="gray")
                st.caption(label)


def _render_documents(documents: list[dict[str, Any]]) -> None:
    """Render retrieved knowledge documents in a professional card-like format."""

    st.subheader("Retrieved Knowledge")
    if not documents:
        st.info("No matching knowledge-base articles were found. The investigation continued using ticket and log evidence.")
        return

    for index, doc in enumerate(documents[:5], start=1):
        score = doc.get("score")
        with st.expander(f"{index}. {doc.get('title', 'Untitled document')}", expanded=index == 1):
            st.caption(f"Source: {doc.get('source', 'Unknown')} | Relevance: {score if score is not None else 'n/a'}")
            st.write(doc.get("content", ""))


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

    st.subheader("Workflow trace")
    if not trace:
        st.info("No workflow execution trace is available yet.")
        return

    with st.expander(f"{len(trace)} observable agent steps", expanded=False):
        lines = []
        for entry in trace:
            agent = entry.get("agent", "unknown_agent")
            decision = entry.get("decision", "completed")
            summary = entry.get("summary", "No observable summary was recorded.")
            elapsed = entry.get("execution_time")
            timing = f" ({float(elapsed):.3f}s)" if elapsed is not None else ""
            lines.append(f"- **{agent}** · `{decision}`{timing}: {summary}")
        st.markdown("\n".join(lines))


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

    st.title("Deep Agent Support Assistant")
    st.subheader("Multi-Agent AI for Production Incident Investigation and Root Cause Analysis")
    st.write(
        "Upload a support ticket and optional application logs. The LangGraph workflow coordinates multiple specialized AI agents to analyze the incident, retrieve relevant knowledge, determine the most likely root cause, and generate a structured RCA report for human review."
    )
    with st.container(horizontal=True, gap="small"):
        st.badge("🤖 LangGraph", color="blue")
        st.badge("📚 RAG", color="green")
        st.badge("🔄 Multi-Agent", color="orange")
        st.badge("🧠 OpenAI / Ollama", color="violet")
        st.badge("👤 Human Review", color="gray")
    provider_name, model_name = get_model_configuration()
    st.caption(f"Active model: **{provider_name} / {model_name}**")
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

        _render_input_summary(ticket_value, app_logs, nginx_logs, mongodb_logs)

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
    completed_steps = final_state.get("completed_steps") or []

    st.subheader("Execution Summary")
    with st.container(border=True):
        summary_cols = st.columns(4)
        summary_cols[0].metric("Total time", f"{float(execution_time or 0.0):.3f}s")
        summary_cols[1].metric("Current agent", str(current_agent or "Pending"))
        summary_cols[2].metric("Completed agents", str(len(completed_steps)))
        summary_cols[3].metric("Model", f"{provider_name} / {model_name}")
        _render_agent_status(completed_steps, current_agent)

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
        _render_ticket_summary(ticket_summary)
    with col2:
        _render_root_cause(root_cause)

    _render_documents(retrieved_documents)

    _render_log_analysis(log_analysis)

    _render_recommendations(recommendations)
    _render_rca_report(rca_report)

    if pending_review:
        st.markdown("---")
        st.subheader("Human Review")
        st.info("Review the proposed RCA before final approval. You can approve it, request changes, or cancel the investigation.", icon=":material/rate_review:")
        review_reason = st.text_input("Review note", placeholder="Optional reason for approve, revise, or cancel")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Approve Report", width="stretch", type="primary"):
                _apply_review_action(final_state, "approve", review_reason or "Approved in the UI.")
                st.session_state["workflow_state"] = final_state
                st.session_state["pending_review"] = False
                st.success("RCA report approved and finalized.", icon=":material/check_circle:")
                st.rerun()
        with col2:
            if st.button("Request Revision", width="stretch"):
                _apply_review_action(final_state, "revise", review_reason or "Requested revision before finalization.")
                st.session_state["workflow_state"] = final_state
                st.session_state["pending_review"] = False
                st.toast("RCA revision requested.")
                st.rerun()
        with col3:
            if st.button("Cancel Investigation", width="stretch"):
                _apply_review_action(final_state, "cancel", review_reason or "Cancelled in the UI.")
                st.session_state["workflow_state"] = final_state
                st.session_state["pending_review"] = False
                st.error("Investigation cancelled.", icon=":material/cancel:")
                st.rerun()


if __name__ == "__main__":
    main()
