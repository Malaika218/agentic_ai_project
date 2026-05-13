"""
agents/remediation_agent.py

Remediation Agent — dispatches the appropriate tool based on the
recommended_action from the Diagnosis Agent.

Updated with Dry Run support for local testing with chaos_model_server.py.
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage

from state import AgentState
from mlops_agents.tools.mcp_tools import (
    trigger_retraining_pipeline,
    rollback_deployment,
    scale_deployment,
    open_github_issue,
)

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def remediation_agent(state: AgentState) -> AgentState:
    """
    LangGraph node — Remediation Agent.
    
    If DRY_RUN=true, it logs the intent but skips actual tool execution.
    """
    metrics: dict = state.get("metrics") or {}
    model_id: str = metrics.get("model_id", os.getenv("DEFAULT_MODEL_ID", "unknown"))
    environment: str = metrics.get("environment", os.getenv("DEFAULT_ENVIRONMENT", "production"))
    action: str = state.get("recommended_action", "investigate")
    diagnosis: str = state.get("diagnosis", "")
    severity: str = state.get("severity", "minor")

    # Check for Dry Run mode
    is_dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    logger.info(
        "Remediation Agent: %s action='%s' for %s (%s)",
        "[DRY RUN]" if is_dry_run else "Executing",
        action, model_id, environment,
    )

    if is_dry_run:
        result = {
            "status": "dry_run_success",
            "detail": f"DRY RUN: Would have triggered '{action}' for {model_id} due to: {diagnosis[:50]}..."
        }
    else:
        # Standard deterministic dispatch
        if action == "retrain":
            prescription = state.get("retrain_prescription") or {}
            metrics      = state.get("metrics") or {}

            # derive data window from drift onset if available
            drift_onset  = state.get("drift_onset_at")
            if drift_onset and not prescription.get("window_days"):
                from datetime import datetime, timezone
                onset_dt     = datetime.fromisoformat(drift_onset)
                days_drifting = (datetime.now(timezone.utc) - onset_dt).days
                prescription["window_days"] = max(14, days_drifting + 7)

            result = trigger_retraining_pipeline(
                model_id=model_id,
                environment=environment,
                reason=diagnosis,
                severity=severity,
                prescription=prescription,
                current_metrics=metrics,
            )
        elif action == "rollback":
            result = rollback_deployment(model_id=model_id, environment=environment, reason=diagnosis)
        elif action == "scale":
            result = scale_deployment(model_id=model_id, environment=environment)
        elif action == "investigate":
            result = open_github_issue(model_id=model_id, environment=environment, diagnosis=diagnosis, severity=severity, metrics=metrics)
        else:
            result = {"status": "failed", "detail": f"Unknown action '{action}'."}

    status: str = result.get("status", "failed")
    detail: str = result.get("detail", "")

    return {
        **state,
        "remediation_action": action,
        "remediation_status": status,
        "remediation_detail": detail,
        "messages": state.get("messages", [])
        + [
            HumanMessage(
                content=f"[Remediation] {'(DRY RUN) ' if is_dry_run else ''}action={action} status={status} — {detail}"
            )
        ],
    }