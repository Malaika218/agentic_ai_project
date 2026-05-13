"""
graph/workflow.py

LangGraph StateGraph for the MLOps Multi-Agent pipeline.

Flow:
    [monitor] → route_by_severity
        ├── "none"     → END
        ├── "minor"    → [diagnosis] → [remediation] → [reporting] → END
        ├── "critical" → [diagnosis] → [remediation] → [reporting] → END
        └── "major"    → [diagnosis] → [human_approval] → [remediation] → [reporting] → END

The RAGStore is instantiated once and injected into each node via functools.partial
so agents don't need to create their own connections.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
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
    After diagnosis, major incidents require human approval.
    Minor and critical go straight to remediation.
    """
    severity = state.get("severity", "minor")
    if severity == "major":
        logger.info("Severity=major — routing to human_approval node.")
        return "human_approval"
    return "remediation"


# ---------------------------------------------------------------------------
# Human approval node
# ---------------------------------------------------------------------------

def human_approval_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop checkpoint for major incidents.

    Production behaviour:
        - Raises GraphInterrupt so LangGraph pauses the thread.
        - The external orchestrator (Slack webhook, dashboard) resumes the graph
          with {"human_approved": True/False} injected into the state.

    The GraphInterrupt import is guarded so the graph can still be instantiated
    in environments where the checkpoint backend is not configured.
    """
    try:
        from langgraph.errors import GraphInterrupt  # available in langgraph ≥ 0.1

        metrics: dict = state.get("metrics") or {}
        raise GraphInterrupt(
            {
                "message": (
                    f"Human approval required for MAJOR incident on "
                    f"{metrics.get('model_id', 'unknown')} "
                    f"({metrics.get('environment', 'production')})."
                ),
                "diagnosis": state.get("diagnosis", ""),
                "recommended_action": state.get("recommended_action", ""),
                "incident_metrics": metrics,
            }
        )
    except ImportError:
        # Fallback: auto-approve (for environments without checkpoint backend)
        logger.warning(
            "GraphInterrupt not available — auto-approving major incident. "
            "Install langgraph>=0.1 and configure a checkpoint backend for "
            "production human-in-the-loop behaviour."
        )
        return {**state, "human_approved": True}


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

    # Use MemorySaver for thread checkpointing (swap for Redis/Postgres in production)
    checkpointer = MemorySaver()
    app = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"]
        if os.getenv("HUMAN_IN_THE_LOOP", "true").lower() == "true"
        else [],
    )

    logger.info("LangGraph compiled successfully.")
    return app
