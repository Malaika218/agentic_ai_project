"""
api/main.py

FastAPI backend — wraps the LangGraph CLI pipeline with HTTP endpoints
so the Streamlit dashboard can trigger runs, poll status, and handle
human-in-the-loop approvals without touching the terminal.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /runs                      trigger a new pipeline run
    GET  /runs                      list all runs + status
    GET  /runs/{thread_id}          single run detail
    POST /runs/{thread_id}/approve  resume paused major-severity thread
    POST /runs/{thread_id}/reject   reject + end paused thread
    GET  /metrics/current           latest metrics snapshot from model server
    GET  /incidents                 ChromaDB incident history
    GET  /health                    Ollama + ChromaDB + model server reachability
    POST /datasets/create           run dataset_generator.py for all scenarios
    GET  /datasets                  list available datasets on disk
    POST /generator/start           start transaction generator on a named dataset
    POST /generator/stop            stop the generator subprocess
    GET  /generator/status          is generator running, which dataset, how many sent
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests as http
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api")

# ── config ────────────────────────────────────────────────────────────────────
MODEL_SERVER_URL = os.getenv("FRAUD_MODEL_MCP_URL", "http://localhost:8080")
OLLAMA_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PROJECT_ROOT     = Path(__file__).parent.parent
DATASETS_DIR     = PROJECT_ROOT / "data" / "datasets"
ACTIVE_DATASET_FILE = PROJECT_ROOT / "data" / "active_dataset.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    from mlops_agents.rag.init_collections import init
    init()
    yield

app = FastAPI(
    title="MLOps Agent API",
    description="HTTP wrapper around the LangGraph MLOps pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# ── in-memory run registry ────────────────────────────────────────────────────
runs: dict[str, dict[str, Any]] = {}

# ── in-memory generator state ─────────────────────────────────────────────────
_generator_proc: subprocess.Popen | None = None
_generator_state: dict[str, Any] = {
    "running":  False,
    "dataset":  None,
    "pid":      None,
    "started_at": None,
}

# ── pydantic models ───────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    model_id:    str = os.getenv("DEFAULT_MODEL_ID", "fraud-classifier-v1")
    environment: str = "production"


class GeneratorStartRequest(BaseModel):
    dataset: str = "baseline"
    rate:    float = 2.0
    error_rate: float = 0.0
    seed_n:  int = 500

# ── pipeline execution helpers ────────────────────────────────────────────────

def _build_graph():
    from mlops_agents.rag.store import RAGStore
    from mlops_agents.graph.workflow import build_graph
    rag = RAGStore()
    return build_graph(rag=rag)


async def _execute_pipeline(thread_id: str, model_id: str, environment: str):
    runs[thread_id]["status"]        = "running"
    runs[thread_id]["started_at"]    = datetime.now(timezone.utc).isoformat()
    runs[thread_id]["current_agent"] = "monitor"

    try:
        app_graph = _build_graph()
        config    = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "model_id":    model_id,
            "environment": environment,
            "messages":    [],
        }

        for event in app_graph.stream(initial_state, config=config):
            node_name = list(event.keys())[0]
            node_out  = event[node_name] or {}

            runs[thread_id]["current_agent"] = node_name

            for field in [
                "severity", "diagnosis", "recommended_action", "remediation_status",
                "incident_id", "report", "remediation_action", "remediation_detail",
                "diagnosis_json", "retrain_prescription", "drifted_features",
                "similar_incidents", "relevant_runbooks", "notifications_sent",
            ]:
                if field in node_out:
                    runs[thread_id][field] = node_out[field]

            if "messages" in node_out:
                runs[thread_id]["messages"] = [
                    m.content if hasattr(m, "content") else str(m)
                    for m in node_out["messages"]
                ]

        runs[thread_id]["status"]       = "completed"
        runs[thread_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as exc:
        exc_name = type(exc).__name__
        if "GraphInterrupt" in exc_name or "interrupt" in str(exc).lower():
            runs[thread_id]["status"] = "awaiting_approval"
            logger.info("Pipeline paused at human approval — thread %s", thread_id)
        else:
            runs[thread_id]["status"] = "failed"
            runs[thread_id]["error"]  = str(exc)
            logger.error("Pipeline failed for thread %s: %s", thread_id, exc)


async def _resume_pipeline(thread_id: str, approved: bool):
    runs[thread_id]["status"]        = "running"
    runs[thread_id]["current_agent"] = "remediation"

    try:
        app_graph    = _build_graph()
        config       = {"configurable": {"thread_id": thread_id}}
        resume_state = {"human_approved": approved}

        for event in app_graph.stream(resume_state, config=config):
            node_name = list(event.keys())[0]
            node_out  = event[node_name] or {}
            runs[thread_id]["current_agent"] = node_name

            for field in [
                "remediation_status", "incident_id", "report", "remediation_action",
                "remediation_detail", "diagnosis_json", "retrain_prescription",
                "drifted_features", "similar_incidents", "relevant_runbooks",
                "notifications_sent",
            ]:
                if field in node_out:
                    runs[thread_id][field] = node_out[field]

            if "messages" in node_out:
                runs[thread_id]["messages"] = [
                    m.content if hasattr(m, "content") else str(m)
                    for m in node_out["messages"]
                ]

        runs[thread_id]["status"]         = "completed"
        runs[thread_id]["completed_at"]   = datetime.now(timezone.utc).isoformat()
        runs[thread_id]["human_approved"] = approved

    except Exception as exc:
        runs[thread_id]["status"] = "failed"
        runs[thread_id]["error"]  = str(exc)
        logger.error("Resume failed for thread %s: %s", thread_id, exc)

# ── pipeline run endpoints ────────────────────────────────────────────────────

@app.post("/runs", status_code=202)
async def trigger_run(req: RunRequest, background_tasks: BackgroundTasks):
    """Trigger a new monitoring pipeline run. Returns immediately with thread_id."""
    thread_id = str(uuid.uuid4())

    runs[thread_id] = {
        "thread_id":            thread_id,
        "model_id":             req.model_id,
        "environment":          req.environment,
        "status":               "queued",
        "severity":             None,
        "current_agent":        None,
        "diagnosis":            None,
        "recommended_action":   None,
        "remediation_status":   None,
        "incident_id":          None,
        "report":               None,
        "human_approved":       None,
        "error":                None,
        "created_at":           datetime.now(timezone.utc).isoformat(),
        "started_at":           None,
        "completed_at":         None,
        "remediation_action":   None,
        "remediation_detail":   None,
        "diagnosis_json":       None,
        "retrain_prescription": None,
        "drifted_features":     [],
        "similar_incidents":    [],
        "relevant_runbooks":    [],
        "notifications_sent":   [],
        "messages":             [],
    }

    background_tasks.add_task(
        _execute_pipeline, thread_id, req.model_id, req.environment
    )

    logger.info("Queued run %s for model=%s env=%s", thread_id, req.model_id, req.environment)
    return {"thread_id": thread_id, "status": "queued"}


@app.get("/runs")
async def list_runs():
    """List all runs sorted newest first."""
    return sorted(
        runs.values(),
        key=lambda r: r.get("created_at", ""),
        reverse=True,
    )


@app.get("/runs/{thread_id}")
async def get_run(thread_id: str):
    """Get the current state of a single pipeline run."""
    if thread_id not in runs:
        raise HTTPException(404, f"Thread {thread_id} not found")
    return runs[thread_id]


@app.post("/runs/{thread_id}/approve", status_code=202)
async def approve_run(thread_id: str, background_tasks: BackgroundTasks):
    """Approve a pipeline paused at the human approval checkpoint."""
    run = runs.get(thread_id)
    if not run:
        raise HTTPException(404, f"Thread {thread_id} not found")
    if run["status"] != "awaiting_approval":
        raise HTTPException(400, f"Run is not awaiting approval — status: {run['status']}")

    background_tasks.add_task(_resume_pipeline, thread_id, True)
    return {"thread_id": thread_id, "status": "resuming", "approved": True}


@app.post("/runs/{thread_id}/reject", status_code=202)
async def reject_run(thread_id: str, background_tasks: BackgroundTasks):
    """Reject a pipeline paused at the human approval checkpoint."""
    run = runs.get(thread_id)
    if not run:
        raise HTTPException(404, f"Thread {thread_id} not found")
    if run["status"] != "awaiting_approval":
        raise HTTPException(400, f"Run is not awaiting approval — status: {run['status']}")

    background_tasks.add_task(_resume_pipeline, thread_id, False)
    return {"thread_id": thread_id, "status": "resuming", "approved": False}

# ── metrics endpoint ──────────────────────────────────────────────────────────

@app.get("/metrics/current")
async def current_metrics(model_id: str = None):
    """Fetch live metrics from the model server via MCP."""
    try:
        r = http.post(
            f"{MODEL_SERVER_URL}/mcp/call",
            json={"tool": "get_current_metrics", "params": {}},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Could not reach model server: {e}")

# ── incidents endpoint ────────────────────────────────────────────────────────

@app.get("/incidents")
async def list_incidents(limit: int = 50, severity: str = None):
    """Query ChromaDB for recent incidents."""
    try:
        from mlops_agents.rag.store import RAGStore
        rag   = RAGStore()
        where = {"severity": severity} if severity else None

        results = rag._incidents.get(
            where=where,
            include=["metadatas"],
            limit=limit,
        )
        metas = results.get("metadatas") or []
        return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)

    except Exception as e:
        raise HTTPException(500, f"ChromaDB query failed: {e}")

# ── dataset endpoints ─────────────────────────────────────────────────────────

@app.post("/datasets/create", status_code=202)
async def create_datasets(background_tasks: BackgroundTasks):
    """
    Run dataset_generator.py to create all scenario CSVs in data/datasets/.
    This is a one-time setup step.
    """
    script_path = PROJECT_ROOT / "scripts" / "dataset_generator.py"
    if not script_path.exists():
        raise HTTPException(404, f"dataset_generator.py not found at {script_path}")

    def _run_generator():
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            )
            if result.returncode != 0:
                logger.error("dataset_generator.py failed:\n%s", result.stderr)
            else:
                logger.info("dataset_generator.py completed successfully")
        except Exception as exc:
            logger.error("Failed to run dataset_generator.py: %s", exc)

    background_tasks.add_task(_run_generator)
    return {"status": "started", "message": "Dataset generation running in background"}


@app.get("/datasets")
async def list_datasets():
    """List available datasets on disk with their metadata."""
    if not DATASETS_DIR.exists():
        return []

    datasets = []
    for csv_path in sorted(DATASETS_DIR.glob("*.csv")):
        meta_path = csv_path.with_suffix(".json")
        entry: dict[str, Any] = {"name": csv_path.stem, "csv": csv_path.name}

        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    entry.update(json.load(f))
            except Exception:
                pass
        else:
            # fallback: just report row count
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, usecols=["Class"])
                entry["rows"] = len(df)
                entry["fraud_rate"] = float(df["Class"].mean())
            except Exception:
                entry["rows"] = None

        # mark which dataset is active
        active_name = _read_active_dataset()
        entry["active"] = (csv_path.stem == active_name)
        datasets.append(entry)

    return datasets

# ── generator endpoints ───────────────────────────────────────────────────────

def _read_active_dataset() -> str | None:
    """Read the currently active dataset name from the shared state file."""
    try:
        if ACTIVE_DATASET_FILE.exists():
            with open(ACTIVE_DATASET_FILE) as f:
                return json.load(f).get("dataset")
    except Exception:
        pass
    return None


def _write_active_dataset(name: str) -> None:
    """Write the active dataset name to the shared state file."""
    ACTIVE_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_DATASET_FILE, "w") as f:
        json.dump({"dataset": name, "updated_at": datetime.now(timezone.utc).isoformat()}, f)


def _proc_alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


@app.post("/generator/start", status_code=202)
async def start_generator(req: GeneratorStartRequest):
    """
    Start the transaction generator subprocess replaying rows from a named dataset.
    Writes the dataset name to data/active_dataset.json so the generator and Drift Lab
    stay in sync.
    """
    global _generator_proc, _generator_state

    if _proc_alive(_generator_proc):
        raise HTTPException(409, "Generator is already running. Stop it first.")

    dataset_csv = DATASETS_DIR / f"{req.dataset}.csv"
    if not dataset_csv.exists():
        raise HTTPException(404, f"Dataset not found: {dataset_csv}. Run POST /datasets/create first.")

    script_path = PROJECT_ROOT / "scripts" / "transaction_generator.py"
    if not script_path.exists():
        raise HTTPException(404, f"transaction_generator.py not found at {script_path}")

    # Write active dataset so generator and UI are in sync
    _write_active_dataset(req.dataset)

    cmd = [
        sys.executable, str(script_path),
        "--dataset",    req.dataset,
        "--rate",       str(req.rate),
        "--error-rate", str(req.error_rate),
        "--seed-n",     str(req.seed_n),
        "--quiet",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
        _generator_proc = proc
        _generator_state = {
            "running":    True,
            "dataset":    req.dataset,
            "pid":        proc.pid,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Generator started — pid=%d dataset=%s", proc.pid, req.dataset)
    except Exception as e:
        raise HTTPException(500, f"Failed to start generator: {e}")

    return {"status": "started", "pid": _generator_state["pid"], "dataset": req.dataset}


@app.post("/generator/stop")
async def stop_generator():
    """Stop the transaction generator subprocess."""
    global _generator_proc, _generator_state

    if not _proc_alive(_generator_proc):
        _generator_state["running"] = False
        return {"status": "not_running"}

    try:
        _generator_proc.terminate()
        _generator_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _generator_proc.kill()
    except Exception as e:
        logger.warning("Error stopping generator: %s", e)

    pid = _generator_state.get("pid")
    _generator_proc = None
    _generator_state = {
        "running":    False,
        "dataset":    _generator_state.get("dataset"),
        "pid":        None,
        "started_at": None,
    }
    logger.info("Generator stopped — pid=%s", pid)
    return {"status": "stopped"}


@app.get("/generator/status")
async def generator_status():
    """Return whether the generator is running, which dataset, and PID."""
    global _generator_proc, _generator_state

    alive = _proc_alive(_generator_proc)
    if not alive and _generator_state["running"]:
        # Process died on its own
        _generator_state["running"] = False
        _generator_state["pid"]     = None

    return {
        "running":    _generator_state["running"] and alive,
        "dataset":    _generator_state.get("dataset"),
        "pid":        _generator_state.get("pid") if alive else None,
        "started_at": _generator_state.get("started_at"),
        "active_dataset_file": _read_active_dataset(),
    }

# ── health endpoint ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check reachability of all dependent services."""
    results = {}

    try:
        r = http.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        results["ollama"] = r.ok
    except Exception:
        results["ollama"] = False

    try:
        r = http.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        results["model_server"] = r.ok
    except Exception:
        results["model_server"] = False

    try:
        from mlops_agents.rag.store import RAGStore
        rag = RAGStore()
        results["chromadb"] = rag._incidents.count() >= 0
    except Exception:
        results["chromadb"] = False

    results["mlflow"] = results["model_server"]

    overall = all(results.values())
    return {
        "status":    "ok" if overall else "degraded",
        "services":  results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info",
    )