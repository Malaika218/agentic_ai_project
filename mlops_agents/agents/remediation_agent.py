# agents/remediation_agent.py

from __future__ import annotations

import logging
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.messages import HumanMessage

from mlops_agents.state import AgentState
from mlops_agents.tools.mcp_tools import (
    trigger_retraining_pipeline,
    rollback_deployment,
    scale_deployment,
    open_github_issue,
)

# NEW IMPORTS: For promoting model versions post-retrain
import mlflow
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def remediation_agent(state: AgentState) -> AgentState:
    """
    LangGraph node — Remediation Agent.
    Dispatches tools based on structured recommendations from the Diagnosis Agent.
    """
    metrics: dict = state.get("metrics") or {}

    active_dataset = metrics.get("active_dataset", "unknown")
    
    # 1. FIXED: Extract using the uniform key 'remediation_action' set by the Diagnosis Agent
    action: str = state.get("remediation_action", "none")
    diagnosis: str = state.get("diagnosis", "")
    severity: str = state.get("severity", "minor")
    
    # Normalize model IDs using the metadata dictionary from our new initialization state
    metadata = metrics.get("metadata") or {}
    model_id: str = state.get("model_id") or metadata.get("model_name", os.getenv("DEFAULT_MODEL_ID", "unknown"))
    environment: str = state.get("environment", os.getenv("DEFAULT_ENVIRONMENT", "production"))

    is_dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    logger.info(
        "[Remediation] starting — action=%s model_id=%s environment=%s severity=%s dry_run=%s",
        action, model_id, environment, severity, is_dry_run,
    )

    if is_dry_run:
        result = {
            "status": "dry_run_success",
            "detail": f"DRY RUN: Would have triggered '{action}' for {model_id} due to: {diagnosis[:50]}..."
        }
    else:
        # 2. CHANGED: Structured matching using the tokens emitted by the Pydantic Diagnosis schema
        if action == "trigger_retraining":
            prescription = state.get("retrain_prescription") or {}

            # Core retraining tool call
            result = trigger_retraining_pipeline(
                model_id=model_id,
                environment=environment,
                reason=diagnosis,
                severity=severity,
                prescription=prescription,
                current_metrics=metrics,
                active_dataset=active_dataset,
            )

        elif action == "scale_infrastructure":
            result = scale_deployment(model_id=model_id, environment=environment)

        elif action == "rollback":
            result = rollback_deployment(model_id=model_id, environment=environment)
            
        elif action == "none":
            result = {"status": "skipped", "detail": "Diagnosis requested no remediation actions. System state within normal bounds."}
            
        elif action == "investigate":
            result = open_github_issue(model_id=model_id, environment=environment, diagnosis=diagnosis, severity=severity, metrics=metrics)
            
        else:
            # Fallback handler for unmapped schema tokens
            result = {"status": "failed", "detail": f"Unrecognized remediation routing instruction sequence: '{action}'."}

    status: str = result.get("status", "failed")
    detail: str = result.get("detail", "")

    logger.info(
        "[Remediation] complete — action=%s status=%s detail=%s",
        action, status, detail[:200] if detail else "",
    )

    return {
        **state,
        "remediation_status": status,
        "remediation_detail": detail,
        "messages": state.get("messages", [])
        + [
            HumanMessage(
                content=f"[Remediation] {'(DRY RUN) ' if is_dry_run else ''}action={action} status={status} — {detail}"
            )
        ],
    }