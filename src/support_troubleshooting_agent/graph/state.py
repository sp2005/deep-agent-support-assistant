"""Shared LangGraph state and supporting types for the support troubleshooting workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class UploadedFile(TypedDict):
    """Metadata for a file uploaded into the workflow."""

    filename: str
    content_type: str
    size: int
    uploaded_at: str
    path: str


class RetrievedDocument(TypedDict):
    """A single knowledge-base or ticket-document retrieval result."""

    id: str
    title: str
    source: str
    content: str
    score: float
    metadata: dict[str, Any]


class LogAnalysis(TypedDict):
    """Structured output from log analysis."""

    summary: str
    anomalies: list[str]
    evidence: list[str]


class RootCause(TypedDict):
    """Structured root-cause analysis output."""

    primary_cause: str
    confidence: float
    supporting_evidence: list[str]


class Recommendation(TypedDict):
    """Single corrective recommendation."""

    recommendation: str
    priority: str
    rationale: str


class RCAReport(TypedDict):
    """Final root-cause and corrective-action summary."""

    title: str
    summary: str
    root_cause: str
    actions: list[str]
    evidence: list[str]
    status: str


class GraphError(TypedDict):
    """Structured workflow error payload."""

    step: str
    message: str
    details: str


class SupportTroubleshootingState(TypedDict, total=False):
    """Shared, incremental state for the troubleshooting LangGraph workflow.

    All fields are optional to allow the graph to build the state progressively
    across each node execution without requiring a fully-populated payload at the
    start of the workflow.
    """

    ticket: dict[str, Any]
    ticket_summary: str
    uploaded_files: list[UploadedFile]
    retrieved_documents: list[RetrievedDocument]
    log_analysis: LogAnalysis
    root_cause: RootCause
    recommendations: list[Recommendation]
    rca_report: RCAReport
    errors: list[GraphError]
    current_step: str


__all__ = [
    "GraphError",
    "LogAnalysis",
    "RCAReport",
    "Recommendation",
    "RetrievedDocument",
    "RootCause",
    "SupportTroubleshootingState",
    "UploadedFile",
]
