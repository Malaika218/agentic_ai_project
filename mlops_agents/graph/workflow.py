"""
graph/workflow.py

LangGraph StateGraph for the MLOps Multi-Agent pipeline.

Flow:
    [monitor] → route_by_severity
        ├── "none"     → END
        ├── "minor"    → [diagnosis] → [remediation] → [reporting] → END
        ├── "critical" → [diagnosis] → [human_approval] → [remediation] → [reporting] → END
        └── "major"    → [diagnosis] → [human_approval] → [remediation] → [reporting] → END

Human-in-the-loop uses LangGraph's dynamic `interrupt()` function (the recommended
HITL pattern — static `interrupt_before` breakpoints are intended for debugging).
The orchestrator resumes the paused thread with `Command(resume=True/False)`.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

# Single checkpointer instance shared across all build_graph() calls so that
# checkpointed state survives between the initial interrupted run and resume.
_checkpointer = MemorySaver()

from mlops_agents.state import AgentState
from mlops_agents.rag.store import RAGStore
from mlops_agents.agents.monitor_agent import monitor_agent
from mlops_agents.agents.diagnosis_agent import diagnosis_agent
from mlops_agents.agents.remediation_agent import remediation_agent
from mlops_agents.agents.reporting_agent import reporting_agent

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_by_severity(
    state: AgentState,
) -> Literal["diagnosis", "__end__"]:
    """
    After the Monitor Agent runs, decide what to do next.

    - "none"                    → end the pipeline
    - "minor" | "major" | "critical" → proceed to diagnosis
    """
    severity = state.get("severity", "none")
    if severity == "none":
        logger.info("Severity=none — pipeline ends after monitor.")
        return "__end__"
    return "diagnosis"


def route_after_diagnosis(
    state: AgentState,
) -> Literal["human_approval", "remediation"]:
    """
    After diagnosis, major and critical incidents require human approval.
    Minor incidents go straight to remediation.
    """
    severity = state.get("severity", "minor")
    if severity in ("critical", "major"):
        logger.info("[LangGraph Workflow] High severity incident — routing to human_approval node.")
        return "human_approval"
    return "remediation"


# ---------------------------------------------------------------------------
# Human approval node
# ---------------------------------------------------------------------------

def human_approval_node(state: AgentState) -> dict:
    """
    Human-in-the-loop checkpoint.

    Calls LangGraph's `interrupt()` to pause execution. The payload below
    surfaces to the orchestrator as `event["__interrupt__"][0].value` so the
    dashboard knows what it's approving. The orchestrator resumes the thread
    with `Command(resume=True)` (approve) or `Command(resume=False)` (reject);
    that boolean becomes the return value of `interrupt()`.

    NOTE: LangGraph re-runs the entire node from the top on resume, so this
    function must stay side-effect-free up to the `interrupt()` call.
    """
    # HITL-disabled escape hatch (stable per-process — env var doesn't change
    # mid-run, so the rule against conditionally skipping `interrupt()` calls
    # within a node isn't violated).
    if os.getenv("HUMAN_IN_THE_LOOP", "true").lower() != "true":
        logger.info("HUMAN_IN_THE_LOOP disabled — auto-approving.")
        return {"human_approved": True}

    metrics: dict = state.get("metrics") or {}
    approved = interrupt(
        {
            "message": (
                f"Human approval required for {state.get('severity', 'high')}-severity "
                f"incident on {state.get('model_id', 'unknown')} "
                f"({state.get('environment', 'production')})."
            ),
            "model_id": state.get("model_id"),
            "environment": state.get("environment"),
            "severity": state.get("severity"),
            "diagnosis": state.get("diagnosis", ""),
            "recommended_action": state.get("recommended_action", ""),
            "incident_metrics": metrics,
        }
    )

    logger.info("human_approval_node resumed — approved=%s", approved)
    return {"human_approved": bool(approved)}


def route_after_approval(
    state: AgentState,
) -> Literal["remediation", "__end__"]:
    """Route after human approval — reject sends to END."""
    if state.get("human_approved", False):
        return "remediation"
    logger.info("Human rejected the remediation — ending pipeline.")
    return "__end__"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(rag: RAGStore | None = None) -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Args:
        rag: An initialised RAGStore instance. If None, a new one is created.
             Pass an existing instance to share the ChromaDB connection across runs.

    Returns:
        A compiled LangGraph app.
    """
    if rag is None:
        logger.info("No RAGStore provided — creating a new one.")
        rag = RAGStore()

    # Bind RAG to agents that need it
    monitor_node = partial(monitor_agent, rag=rag)
    diagnosis_node = partial(diagnosis_agent, rag=rag)
    reporting_node = partial(reporting_agent, rag=rag)

    # Build graph
    builder = StateGraph(AgentState)

    builder.add_node("monitor", monitor_node)
    builder.add_node("diagnosis", diagnosis_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("remediation", remediation_agent)
    builder.add_node("reporting", reporting_node)

    # Entry point
    builder.set_entry_point("monitor")

    # Edges
    builder.add_conditional_edges(
        "monitor",
        route_by_severity,
        {"diagnosis": "diagnosis", "__end__": END},
    )

    builder.add_conditional_edges(
        "diagnosis",
        route_after_diagnosis,
        {"human_approval": "human_approval", "remediation": "remediation"},
    )

    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"remediation": "remediation", "__end__": END},
    )

    builder.add_edge("remediation", "reporting")
    builder.add_edge("reporting", END)

    # No static `interrupt_before`: the dynamic `interrupt()` call inside
    # human_approval_node handles pausing per the recommended HITL pattern.
    app = builder.compile(checkpointer=_checkpointer)

    logger.info("LangGraph compiled successfully.")
    return app
