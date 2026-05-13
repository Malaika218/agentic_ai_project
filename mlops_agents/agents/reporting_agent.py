"""
agents/reporting_agent.py

Reporting Agent — generates a structured markdown incident report,
saves the full incident to the RAG incidents collection, dispatches
Slack/email notifications, and performs adaptive threshold learning.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from mlops_agents.rag.store import RAGStore
from mlops_agents.tools.mcp_tools import send_slack_notification, send_email_alert
from mlops_agents.agents.threshold_manager import run_threshold_update

logger = logging.getLogger(__name__)

from dotenv import load_dotenv


load_dotenv()


# ---------------------------------------------------------------------------
# Threshold learning config
# ---------------------------------------------------------------------------

THRESHOLD_LIMITS = {
    "accuracy_major": (0.6, 0.85),
    "accuracy_critical": (0.5, 0.75),

    "drift_major": (0.1, 0.7),
    "drift_critical": (0.3, 0.9),

    "latency_major_ms": (500, 3000),
    "latency_critical_ms": (1000, 5000),

    "error_rate_major": (0.01, 0.2),
    "error_rate_critical": (0.05, 0.3),
}

MAX_THRESHOLD_DELTA = 0.02


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(state: AgentState, historical_stats: dict) -> str:
    """Generate a structured markdown incident report."""

    metrics: dict = state.get("metrics") or {}
    diag_json: dict = state.get("diagnosis_json") or {}
    similar = state.get("similar_incidents") or []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_id = metrics.get("model_id", "unknown")
    environment = metrics.get("environment", "production")
    severity = state.get("severity", "none")

    report_lines = [
        f"# Incident Report — {model_id}",
        "",
        f"**Date:** {now}  ",
        f"**Model:** `{model_id}` v{metrics.get('model_version', 'unknown')}  ",
        f"**Environment:** {environment}  ",
        f"**Severity:** {severity.upper()}  ",
        f"**Incident ID:** `{state.get('incident_id', 'pending')}`",
        "",
        "---",
        "",
        "## Metrics at Detection",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {metrics.get('accuracy', 'N/A')} |",
        f"| Drift Score | {metrics.get('drift_score', 'N/A')} |",
        f"| p99 Latency | {metrics.get('latency_p99_ms', 'N/A')} ms |",
        f"| p95 Latency | {metrics.get('latency_p95_ms', 'N/A')} ms |",
        f"| Error Rate | {metrics.get('error_rate', 'N/A')} |",
        f"| Predictions (window) | {metrics.get('prediction_count', 'N/A')} |",
        "",
        "---",
        "",
        "## Root Cause Analysis",
        "",
        f"**Root Cause:** {state.get('diagnosis', 'N/A')}  ",
        f"**Confidence:** {diag_json.get('confidence', 'N/A')}  ",
        "",
        "**Evidence:**",
    ]

    for ev in diag_json.get("evidence", []):
        report_lines.append(f"- {ev}")

    report_lines += [
        "",
        f"**Reasoning:** {diag_json.get('reasoning', 'N/A')}",
        "",
        "---",
        "",
        "## Remediation",
        "",
        f"**Action:** `{state.get('remediation_action', 'N/A')}`  ",
        f"**Status:** {state.get('remediation_status', 'N/A')}  ",
        f"**Detail:** {state.get('remediation_detail', 'N/A')}  ",
        f"**Human Approved:** {state.get('human_approved', False)}",
        "",
        "---",
        "",
        "## Historical Context",
        "",
        f"**Total past incidents:** {historical_stats.get('total', 0)}  ",
    ]

    sev_dist = historical_stats.get("severity_distribution", {})
    if sev_dist:
        report_lines.append(f"**Severity distribution:** {sev_dist}  ")

    act_dist = historical_stats.get("action_distribution", {})
    if act_dist:
        report_lines.append(f"**Most common actions:** {act_dist}  ")

    success_rate = historical_stats.get("remediation_success_rate")
    if success_rate is not None:
        report_lines.append(
            f"**Historical remediation success rate:** {success_rate:.1%}"
        )

    if similar:
        report_lines += [
            "",
            "### Most Similar Past Incidents",
        ]

        for i, inc in enumerate(similar[:3], 1):
            meta = inc.get("metadata", {})

            report_lines.append(
                f"{i}. **{meta.get('severity', 'N/A').upper()}** — "
                f"{meta.get('recommended_action', 'N/A')} → "
                f"{meta.get('remediation_status', 'N/A')} "
                f"(similarity: {inc.get('distance', 0):.3f})"
            )

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def _llm_executive_summary(state: AgentState, report: str) -> str:
    """
    Ask the LLM to write a concise executive summary for Slack.
    """

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You write concise executive summaries of MLOps incidents."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Write a concise 3-sentence summary:\n\n{report[:1500]}"
                    )
                ),
            ]
        )

        return response.content.strip()

    except Exception as exc:
        logger.warning("LLM summary failed: %s — using template", exc)

        metrics: dict = state.get("metrics") or {}

        return (
            f"Model `{metrics.get('model_id', 'unknown')}` "
            f"({state.get('severity', '?').upper()}) "
            f"in {metrics.get('environment', 'production')} degraded. "
            f"Root cause: {state.get('diagnosis', 'unknown')}. "
            f"Remediation status: {state.get('remediation_status', 'N/A')}."
        )

# ---------------------------------------------------------------------------
# Reporting Agent
# ---------------------------------------------------------------------------

def reporting_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Reporting Agent.
    """

    metrics: dict = state.get("metrics") or {}

    severity: str = state.get("severity", "none")
    model_id: str = metrics.get("model_id", "unknown")
    environment: str = metrics.get("environment", "production")

    logger.info(
        "Reporting Agent: generating report for %s (%s)",
        model_id,
        environment,
    )

    # ------------------------------------------------------------------
    # Historical stats
    # ------------------------------------------------------------------

    historical_stats = rag.get_incident_stats(
        model_id=model_id,
        environment=environment,
    )

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------

    report = _build_report(state, historical_stats)

    # ------------------------------------------------------------------
    # Save incident
    # ------------------------------------------------------------------

    state_with_report = {
        **state,
        "report": report,
    }

    incident_id = rag.save_incident(state_with_report)

    logger.info("Saved incident to RAG: %s", incident_id)

    report = report.replace("pending", incident_id, 1)

    # ------------------------------------------------------------------
    # Slack
    # ------------------------------------------------------------------

    notifications: list[str] = []

    slack_enabled = (
        os.getenv("SLACK_NOTIFICATIONS_ENABLED", "true").lower() == "true"
    )

    if slack_enabled:

        summary = _llm_executive_summary(state, report)

        slack_result = send_slack_notification(
            message=summary,
            severity=severity,
            incident_id=incident_id,
        )

        if slack_result.get("status") == "success":
            notifications.append("slack")
        else:
            logger.warning(
                "Slack notification failed: %s",
                slack_result.get("detail"),
            )

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    email_min_severity = os.getenv("EMAIL_MIN_SEVERITY", "major")

    severity_rank = {
        "none": 0,
        "minor": 1,
        "major": 2,
        "critical": 3,
    }

    should_email = (
        severity_rank.get(severity, 0)
        >= severity_rank.get(email_min_severity, 2)
    )

    email_enabled = (
        os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "true").lower() == "true"
    )

    if should_email and email_enabled:

        email_result = send_email_alert(
            subject=(
                f"[MLOps {severity.upper()}] "
                f"{model_id} degradation — {incident_id}"
            ),
            body=report,
            severity=severity,
            incident_id=incident_id,
        )

        if email_result.get("status") == "success":
            notifications.append("email")
        else:
            logger.warning(
                "Email alert failed: %s",
                email_result.get("detail"),
            )

    logger.info(
        "Reporting complete. incident_id=%s notifications=%s",
        incident_id,
        notifications,
    )

    # ------------------------------------------------------------------
    # Threshold learning
    # ------------------------------------------------------------------

    try:
        run_threshold_update(state, rag)
    except Exception as exc:
        logger.error(
            "Threshold learning failed: %s",
            exc,
        )

    return {
        **state,
        "report": report,
        "incident_id": incident_id,
        "notifications_sent": notifications,
        "messages": state.get("messages", [])
        + [
            HumanMessage(
                content=(
                    f"[Reporting] incident_id={incident_id} "
                    f"notifications={notifications}"
                )
            )
        ],
    }
