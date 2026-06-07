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

from mlops_agents.llm_manager import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from typing import Optional

from pydantic import BaseModel, Field

from mlops_agents.state import AgentState
from mlops_agents.rag.store import RAGStore
from mlops_agents.tools.mcp_tools import send_slack_notification, send_email_alert
from mlops_agents.tools.token_tracker import TokenUsageHandler
from mlops_agents.tools.alert_decider import decide_slack_alert
from mlops_agents.agents.threshold_manager import run_threshold_update

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(state: AgentState, historical_stats: dict) -> str:
    """Generate a structured markdown incident report."""

    metrics: dict = state.get("metrics") or {}
    diag_json: dict = state.get("diagnosis_json") or {}
    similar = state.get("similar_incidents") or []
    metadata: dict = metrics.get("metadata") or {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    # Monitor stores model_id/environment at top-level state and inside
    # metrics["metadata"]; they are NOT keys of `metrics` itself.
    model_id = state.get("model_id") or metadata.get("model_name", "unknown")
    environment = state.get("environment") or "production"
    model_version = state.get("model_version") or metadata.get("model_version", "unknown")
    severity = state.get("severity", "none")

    report_lines = [
        f"# Incident Report — {model_id}",
        "",
        f"**Date:** {now}  ",
        f"**Model:** `{model_id}` v{model_version}  ",
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

def _llm_executive_summary(state: AgentState, report: str, tracker: TokenUsageHandler) -> str:
    """
    Ask the LLM to write a concise executive summary for Slack.
    Token usage is routed into the caller-provided tracker.
    """

    llm = get_llm(temperature=0)

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
            ],
            config={"callbacks": [tracker]},
        )

        return response.content.strip()

    except Exception as exc:
        logger.info("LLM summary failed: %s — using template", exc)

        metrics: dict = state.get("metrics") or {}

        return (
            f"Model `{metrics.get('model_id', 'unknown')}` "
            f"({state.get('severity', '?').upper()}) "
            f"in {metrics.get('environment', 'production')} degraded. "
            f"Root cause: {state.get('diagnosis', 'unknown')}. "
            f"Remediation status: {state.get('remediation_status', 'N/A')}."
        )

# ---------------------------------------------------------------------------
# Post-mortem evaluator
# ---------------------------------------------------------------------------

class PostMortemEvaluation(BaseModel):
    """LLM-gated decision on whether to promote an incident to a runbook."""

    should_save: bool = Field(
        description=(
            "True iff this incident teaches something the existing runbooks "
            "do NOT already capture. Default false when in doubt — we'd rather "
            "miss a near-duplicate than flood the collection with noise."
        )
    )
    reason: str = Field(
        description="One-sentence rationale for the decision (kept either way)."
    )
    title: Optional[str] = Field(
        default=None,
        description="Short title for the post-mortem; populated only if should_save=True.",
    )
    summary: Optional[str] = Field(
        default=None,
        description=(
            "Markdown body for the post-mortem (populated only if should_save=True). "
            "Should structure: ## Symptom, ## Root Cause, ## Action Taken, ## Outcome, "
            "## Lessons Learned. Keep it under 800 chars — concise > exhaustive."
        ),
    )
    tags: Optional[str] = Field(
        default=None,
        description="Comma-separated tags for retrieval (e.g. 'concept_drift,retrain,recall').",
    )


def _format_runbook_for_prompt(rb: dict) -> str:
    """Render a single existing runbook row for the novelty prompt."""
    meta = rb.get("metadata") or {}
    doc = (rb.get("document") or "")[:300]
    return (
        f"- type={meta.get('doc_type', '?')} title={meta.get('title', '?')!r} "
        f"tags={meta.get('tags', '')!r} distance={rb.get('distance', 0):.3f}\n"
        f"  excerpt: {doc.replace(chr(10), ' ')}…"
    )


def _evaluate_post_mortem(
    state: AgentState,
    incident_id: str,
    similar_runbooks: list,
    tracker: TokenUsageHandler,
) -> Optional[PostMortemEvaluation]:
    """
    Ask the LLM whether this incident is novel enough to deserve a post-mortem
    entry in the runbooks collection. Returns the parsed evaluation, or None
    on failure (so the caller can degrade gracefully — never break reporting).
    """
    diag_json = state.get("diagnosis_json") or {}
    metrics: dict = state.get("metrics") or {}
    drifted = state.get("drifted_features") or []

    incident_summary = (
        f"Severity: {state.get('severity', 'unknown')}\n"
        f"Root cause: {state.get('diagnosis', 'N/A')}\n"
        f"Category: {diag_json.get('root_cause_category', 'unknown')}\n"
        f"Recommended action: {state.get('recommended_action', 'N/A')}\n"
        f"Remediation status: {state.get('remediation_status', 'N/A')}\n"
        f"Drifted features ({len(drifted)}): {drifted[:8]}\n"
        f"Key metrics: accuracy={metrics.get('accuracy')} recall={metrics.get('recall')} "
        f"roc_auc={metrics.get('roc_auc')} fraud_rate={metrics.get('fraud_rate')}\n"
        f"Diagnosis reasoning: {diag_json.get('reasoning', 'N/A')[:400]}"
    )

    if similar_runbooks:
        existing_block = "\n".join(
            _format_runbook_for_prompt(rb) for rb in similar_runbooks
        )
    else:
        existing_block = "(no existing runbooks or post-mortems retrieved)"

    prompt = f"""You are a curator for an ML incident runbook collection.

Your job: decide whether the CURRENT INCIDENT teaches an operationally useful
lesson that the EXISTING ENTRIES do NOT already cover. Bias toward "no" —
duplicates and near-duplicates hurt retrieval quality more than missed entries.

Save a post-mortem ONLY if at least one of these holds:
  1. The root cause / drift pattern is genuinely new vs the existing entries.
  2. The action taken was unusual or surprising for this kind of incident.
  3. The remediation FAILED in a way prior entries don't document.

Do NOT save if:
  - Existing entries already describe this pattern with similar root cause + action.
  - The incident was minor / routine recovery without new insight.
  - Information would just duplicate the diagnosis output verbatim.

CURRENT INCIDENT (id={incident_id}):
{incident_summary}

EXISTING ENTRIES (most similar, top {len(similar_runbooks)}):
{existing_block}

Return a structured decision. If should_save=true, write a CONCISE markdown
body (≤ 800 chars) that future operators can use as a playbook."""

    try:
        llm = get_llm(temperature=0).with_structured_output(PostMortemEvaluation)
        return llm.invoke(
            [
                SystemMessage(content="You curate ML incident runbooks. Be selective — quality over quantity."),
                HumanMessage(content=prompt),
            ],
            config={"callbacks": [tracker]},
        )
    except Exception as exc:
        logger.info("[Reporting] post-mortem evaluator failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Reporting Agent
# ---------------------------------------------------------------------------

def reporting_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Reporting Agent.
    Aggregates diagnosis metrics, strips telemetry histogram payloads for vector safety,
    persists history snapshots to RAG memory, and triggers threshold optimizations.
    """
    metrics: dict = state.get("metrics") or {}
    severity: str = state.get("severity", "none")
    
    # Extract structural names using our updated metadata object patterns
    metadata = metrics.get("metadata") or {}
    model_id: str = state.get("model_id") or metadata.get("model_name", "unknown")
    environment: str = state.get("environment", "production")

    logger.info(
        "[Reporting] starting — model_id=%s environment=%s severity=%s",
        model_id, environment, severity,
    )

    # 1. Historical Stats retrieval from vector layer
    historical_stats = rag.get_incident_stats(
        model_id=model_id,
        environment=environment,
    )

    # 2. Build report string context payload layout
    report = _build_report(state, historical_stats)

    # Generate an LLM executive summary and prepend to the report before saving.
    reporting_tracker = TokenUsageHandler()
    try:
        executive_summary = _llm_executive_summary(state, report, reporting_tracker)
    except Exception as err:
        logger.info("Executive summary generation failed: %s", err)
        executive_summary = None

    if executive_summary:
        report = (
            f"## Executive Summary\n\n{executive_summary}\n\n---\n\n" + report
        )

    # 3. CRITICAL STRUCTURAL ADJUSTMENT: Prepare a vector-optimized state copy.
    # We strip out the heavy histogram arrays so we don't bloat the vector database metadata attributes.
    sanitized_metrics = {
        k: v for k, v in metrics.items()
        if k not in ["reference_histograms", "production_histograms"]
    }
    # Track metadata summary stats instead of heavy matrices
    sanitized_metrics["had_reference_histograms"] = "reference_histograms" in metrics
    sanitized_metrics["had_production_histograms"] = "production_histograms" in metrics

    vector_safe_state = {
        **state,
        "metrics": sanitized_metrics,
        "report": report,
        "executive_summary": executive_summary or "",
    }

    # Save clean, optimized document footprint to RAG
    incident_id = rag.save_incident(vector_safe_state)
    logger.info(
        "[Reporting] persisted incident — id=%s report_len=%d chars",
        incident_id, len(report),
    )

    # Replace temp tokens with valid incident identifiers
    report = report.replace("pending", incident_id, 1)

    # ── Post-mortem evaluation (LLM-gated runbook write) ────────────────────
    # Major/critical incidents may be worth promoting to the runbooks
    # collection as a `post_mortem` entry — but only if they teach something
    # new vs existing runbooks. The LLM gate prevents duplicate flooding.
    postmortem_enabled = os.getenv("POSTMORTEM_ENABLED", "true").lower() == "true"
    postmortem_id: Optional[str] = None
    if postmortem_enabled and severity in ("major", "critical"):
        try:
            query_text = (
                f"{state.get('diagnosis', '')} "
                f"severity={severity} action={state.get('recommended_action', '')}"
            )
            similar = rag.query_runbooks(query_text=query_text, n_results=5) or []
            logger.info(
                "[Reporting] post-mortem: querying novelty against %d existing entries",
                len(similar),
            )
            evaluation = _evaluate_post_mortem(
                state, incident_id, similar, reporting_tracker
            )
            if evaluation is None:
                logger.info("[Reporting] post-mortem: evaluator returned no result — skipping")
            elif not evaluation.should_save:
                logger.info(
                    "[Reporting] post-mortem: NOT saving — %s",
                    evaluation.reason,
                )
            elif not evaluation.summary:
                logger.info(
                    "[Reporting] post-mortem: should_save=True but summary empty — skipping"
                )
            else:
                diag_cat = (state.get("diagnosis_json") or {}).get("root_cause_category", "unknown")
                postmortem_id = rag.ingest_runbook({
                    "title": evaluation.title or f"Post-mortem: {state.get('diagnosis', incident_id)[:60]}",
                    "content": evaluation.summary,
                    "doc_type": "post_mortem",
                    "tags": evaluation.tags or f"{severity},{state.get('recommended_action', '')},{diag_cat}",
                    "author": "reporting_agent",
                    "source_url": f"incident://{incident_id}",
                })
                logger.info(
                    "[Reporting] post-mortem saved — runbook_id=%s reason=%s",
                    postmortem_id, evaluation.reason,
                )
        except Exception as exc:
            logger.info("[Reporting] post-mortem flow failed (non-fatal): %s", exc)

    # 4. Slack Channels Notification Handlers (Disabled via Env Flags based on your context)
    notifications: list[str] = []
    slack_enabled = os.getenv("SLACK_NOTIFICATIONS_ENABLED", "false").lower() == "true"

    if slack_enabled:
        try:
            # Ask LLM whether this incident should trigger Slack.
            decision = decide_slack_alert(state, report, reporting_tracker)
            if not decision.get("alert"):
                logger.info("LLM decided not to alert Slack: %s", decision.get("reason"))
            else:
                # Use existing executive summary if available, otherwise generate one.
                summary = executive_summary or _llm_executive_summary(state, report, reporting_tracker)
                slack_result = send_slack_notification(
                    message=summary,
                    severity=severity,
                    incident_id=incident_id,
                )
                if slack_result.get("status") == "success":
                    notifications.append("slack")
        except Exception as err:
            logger.info("Slack reporting interface bypassed or error encountered: %s", err)

    # 5. Email Alert Dispatch Handling (Disabled via Env Flags based on your context)
    email_min_severity = os.getenv("EMAIL_MIN_SEVERITY", "major")
    severity_rank = {"none": 0, "minor": 1, "major": 2, "critical": 3}
    
    should_email = severity_rank.get(severity, 0) >= severity_rank.get(email_min_severity, 2)
    email_enabled = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower() == "true"

    if should_email and email_enabled:
        try:
            email_result = send_email_alert(
                subject=f"[MLOps {severity.upper()}] {model_id} degradation — {incident_id}",
                body=report,
                severity=severity,
                incident_id=incident_id,
            )
            if email_result.get("status") == "success":
                notifications.append("email")
        except Exception as err:
            logger.info("Email reporting interface bypassed or error encountered: %s", err)

    logger.info(
        "[Reporting] notifications dispatched — incident_id=%s notifications=%s",
        incident_id, notifications or "none",
    )

    # 6. Threshold learning (Feeds tuned boundary rules back into RAG for Monitor Node ingestion)
    logger.info("[Reporting] starting threshold optimization — model_id=%s environment=%s", model_id, environment)
    threshold_tracker = TokenUsageHandler()
    try:
        run_threshold_update(state, rag, tracker=threshold_tracker)
    except Exception as exc:
        logger.info("Dynamic threshold tuning optimizer failed: %s", exc)

    logger.info(
        "[Reporting] complete — incident_id=%s notifications=%s",
        incident_id, notifications or "none",
    )

    reporting_summary = reporting_tracker.summary()
    threshold_summary = threshold_tracker.summary()
    logger.info(
        "[Reporting] token usage — reporting=%s threshold_manager=%s",
        reporting_summary, threshold_summary,
    )

    return {
        **state,
        "report": report,
        "incident_id": incident_id,
        "postmortem_runbook_id": postmortem_id,  # None when LLM gate said skip
        "notifications_sent": notifications,
        "token_usage": {
            "reporting":         reporting_summary,
            "threshold_manager": threshold_summary,
        },
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Reporting] Compiled incident_id={incident_id}. Actions completed.")
        ]
    }
