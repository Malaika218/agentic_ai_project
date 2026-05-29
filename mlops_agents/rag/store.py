"""
rag/store.py

Production ChromaDB RAG store.

Three collections:
  - incidents       : full incident records for Diagnosis Agent retrieval
  - metrics_history : lightweight metric snapshots for trend analysis
  - runbooks        : institutional knowledge (runbooks, post-mortems, playbooks)

All public methods are synchronous and safe to call from within LangGraph nodes.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

MetricsSnapshot = dict[str, Any]
IncidentRecord = dict[str, Any]
RunbookDoc = dict[str, Any]


# ---------------------------------------------------------------------------
# RAGStore
# ---------------------------------------------------------------------------

class RAGStore:
    """
    Singleton-style wrapper around three ChromaDB collections.

    Instantiate once (e.g. in graph/workflow.py) and pass as a dependency
    into each agent node.
    """

    def __init__(self) -> None:
        self._client = self._build_client()
        self._embed_fn = self._build_embed_fn()
        self._incidents = self._get_or_create("incidents")
        self._metrics = self._get_or_create("metrics_history")
        self._runbooks = self._get_or_create("runbooks")
        logger.info("RAGStore initialised.")

    # ------------------------------------------------------------------
    # Client / embedding setup
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client() -> chromadb.ClientAPI:
        host = os.getenv("CHROMA_HOST", "")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./rag_data")

        if host:
            return chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False),
            )
        os.makedirs(persist_dir, exist_ok=True)
        return chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    @staticmethod
    def _build_embed_fn() -> embedding_functions.EmbeddingFunction:
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return embedding_functions.OllamaEmbeddingFunction(
            model_name=model,
            url=f"{base}/api/embeddings",
        )

    def _get_or_create(self, name: str) -> chromadb.Collection:
        try:
            return self._client.get_collection(
                name=name, embedding_function=self._embed_fn
            )
        except Exception:
            return self._client.create_collection(
                name=name,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_meta(value: Any) -> str | int | float | bool:
        """ChromaDB metadata values must be scalar — serialise anything else."""
        if isinstance(value, (str, int, float, bool)):
            return value
        return json.dumps(value)

    def _flatten_meta(self, d: dict) -> dict:
        return {k: self._safe_meta(v) for k, v in d.items() if v is not None}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Coerce a value to float, returning `default` when value is None or invalid."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Coerce a value to int, returning `default` when value is None or invalid."""
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except Exception:
                return default

    
    def save_dynamic_thresholds(self, model_id: str, thresholds: dict) -> str:
        """
        Persist a new version of dynamic thresholds.
        """
        threshold_id = str(uuid.uuid4())
        now = self._now_iso()

        metadata = self._flatten_meta(
            {
                "threshold_id": threshold_id,
                "model_id": model_id,
                "type": "threshold_config",
                "thresholds": json.dumps(thresholds),
                "updated_at": now,
            }
        )

        doc = f"Threshold config for model {model_id} updated at {now}"

        self._metrics.add(
            ids=[threshold_id],
            documents=[doc],
            metadatas=[metadata],
        )

        logger.info("Saved dynamic thresholds for model=%s", model_id)
        return threshold_id
    # ------------------------------------------------------------------
    # ── INCIDENTS collection ──────────────────────────────────────────
    # ------------------------------------------------------------------

    def save_incident(self, state: dict) -> str:
        """
        Persist a completed incident to the incidents collection.

        The embedding document is a rich natural-language summary so that
        future semantic queries surface truly similar incidents.

        Args:
            state: The final LangGraph AgentState dict.

        Returns:
            The incident_id (UUID) assigned to this record.
        """
        incident_id = str(uuid.uuid4())
        metrics: dict = state.get("metrics") or {}
        created_at = self._now_iso()

        # Build a descriptive text document for embedding
        doc = self._incident_to_text(state, metrics)

        metadata = self._flatten_meta(
            {
                "incident_id": incident_id,
                "model_id": metrics.get("model_id", "unknown"),
                "model_version": metrics.get("model_version", "unknown"),
                "environment": metrics.get("environment", "production"),
                "severity": state.get("severity", "none"),
                "recommended_action": state.get("recommended_action", ""),
                "remediation_status": state.get("remediation_status", ""),
                "human_approved": bool(state.get("human_approved", False)),
                "accuracy": float(metrics.get("accuracy")) if metrics.get("accuracy") is not None else 0.0,
                "drift_score": float(metrics.get("drift_score")) if metrics.get("drift_score") is not None else 0.0,
                "latency_p99_ms": float(metrics.get("latency_p99_ms")) if metrics.get("latency_p99_ms") is not None else 0.0,
                "error_rate": float(metrics.get("error_rate")) if metrics.get("error_rate") is not None else 0.0,
                "created_at": created_at,
                "resolved_at": self._now_iso(),
                # Store the full JSON payload for retrieval
                "raw_payload": json.dumps(
                    {
                        "metrics": metrics,
                        "diagnosis": state.get("diagnosis", ""),
                        "recommended_action": state.get("recommended_action", ""),
                        "remediation_action": state.get("remediation_action", ""),
                        "remediation_status": state.get("remediation_status", ""),
                        "report": state.get("report", ""),
                    }
                ),
            }
        )

        self._incidents.add(
            ids=[incident_id],
            documents=[doc],
            metadatas=[metadata],
        )
        logger.info("Saved incident %s (severity=%s)", incident_id, state.get("severity"))
        return incident_id

    def query_similar_incidents(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most semantically similar past incidents.

        Args:
            query_text: Natural language description of the current situation.
            n_results:  Number of results to return.
            where:      Optional ChromaDB metadata filter dict.

        Returns:
            List of dicts, each containing:
                - document  : the stored text
                - metadata  : the metadata dict (raw_payload is pre-parsed JSON)
                - distance  : cosine distance (lower = more similar)
        """
        count = self._incidents.count()
        if count == 0:
            return []

        results = self._incidents.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        incidents = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Deserialise raw_payload for callers
            if "raw_payload" in meta:
                try:
                    meta["raw_payload"] = json.loads(meta["raw_payload"])
                except json.JSONDecodeError:
                    pass
            incidents.append({"document": doc, "metadata": meta, "distance": dist})

        return incidents

    def get_incident_stats(
        self,
        model_id: str | None = None,
        environment: str | None = None,
        limit: int = 100,
    ) -> dict:
        """
        Aggregate statistics across stored incidents for the Reporting Agent.

        Returns a summary dict with counts, rates, and common actions.
        """
        filters: dict[str, Any] = {}
        if model_id:
            filters["model_id"] = model_id
        if environment:
            filters["environment"] = environment

        results = self._incidents.get(
            where=self._build_where_clause(filters),
            include=["metadatas"],
            limit=limit,
        )
        metas = results.get("metadatas") or []
        if not metas:
            return {"total": 0}

        severity_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        remediation_success = 0

        for m in metas:
            sev = m.get("severity", "none")
            act = m.get("recommended_action", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            action_counts[act] = action_counts.get(act, 0) + 1
            if m.get("remediation_status") == "success":
                remediation_success += 1

        return {
            "total": len(metas),
            "severity_distribution": severity_counts,
            "action_distribution": action_counts,
            "remediation_success_rate": round(remediation_success / len(metas), 3),
        }

    # ------------------------------------------------------------------
    # ── METRICS HISTORY collection ────────────────────────────────────
    # ------------------------------------------------------------------

    def save_metrics_snapshot(self, metrics: dict, severity: str) -> str:
        """
        Save a lightweight metrics snapshot after every monitor cycle.

        Args:
            metrics:  The raw metrics dict from the Monitor Agent.
            severity: Classified severity for this snapshot.

        Returns:
            The snapshot_id assigned.
        """
        snapshot_id = str(uuid.uuid4())
        sampled_at = self._now_iso()

        doc = self._metrics_to_text(metrics, severity)

        metadata = self._flatten_meta(
            {
                "snapshot_id": snapshot_id,
                "model_id": metrics.get("model_id", "unknown"),
                "model_version": metrics.get("model_version", "unknown"),
                "environment": metrics.get("environment", "production"),
                "severity": severity,
                "accuracy": self._safe_float(metrics.get("accuracy")),
                "drift_score": self._safe_float(metrics.get("drift_score")),
                "latency_p99_ms": self._safe_float(metrics.get("latency_p99_ms")),
                "error_rate": self._safe_float(metrics.get("error_rate")),
                "prediction_count": self._safe_int(metrics.get("prediction_count")),
                "sampled_at": sampled_at,
            }
        )

        self._metrics.add(
            ids=[snapshot_id],
            documents=[doc],
            metadatas=[metadata],
        )
        logger.debug("Saved metrics snapshot %s (severity=%s)", snapshot_id, severity)
        return snapshot_id

    @staticmethod
    def _build_where_clause(filters: dict[str, Any]) -> dict[str, Any] | None:
        """Build a ChromaDB where clause for one or more metadata filters."""
        if not filters:
            return None
        if len(filters) == 1:
            return filters
        return {"$and": [{k: v} for k, v in filters.items()]}

    def query_recent_metrics(
        self,
        model_id: str,
        n_results: int = 20,
        environment: str | None = None,
    ) -> list[dict]:
        """
        Retrieve recent metric snapshots for a model (trend window).

        Returns list of metadata dicts sorted newest-first by sampled_at.
        """
        count = self._metrics.count()
        if count == 0:
            return []

        filters: dict[str, Any] = {"model_id": model_id}
        if environment:
            filters["environment"] = environment

        results = self._metrics.get(
            where=self._build_where_clause(filters),
            include=["metadatas"],
            limit=n_results,
        )
        metas = results.get("metadatas") or []
        return sorted(metas, key=lambda m: m.get("sampled_at", ""), reverse=True)

    def query_trend_window(
        self,
        current_metrics: dict,
        n_results: int = 10,
    ) -> list[dict]:
        """
        Semantic similarity search over metrics history for anomaly context.

        Useful for finding past periods that looked similar to the current state.
        """
        count = self._metrics.count()
        if count == 0:
            return []

        query_text = self._metrics_to_text(current_metrics, severity="")

        results = self._metrics.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            include=["metadatas", "distances"],
        )

        return [
            {"metadata": m, "distance": d}
            for m, d in zip(
                results["metadatas"][0], results["distances"][0]
            )
        ]

    # ------------------------------------------------------------------
    # ── RUNBOOKS collection ───────────────────────────────────────────
    # ------------------------------------------------------------------

    def ingest_runbook(self, doc: RunbookDoc) -> str:
        """
        Ingest a single runbook / post-mortem / playbook document.

        Args:
            doc: Dict with keys:
                    - title      (required)
                    - content    (required) — full text to embed
                    - doc_type   — runbook | post_mortem | playbook | note
                    - tags       — comma-separated string
                    - author     — author name or team slug
                    - source_url — Confluence / GitHub URL
                    - doc_id     — supply to overwrite; auto-generated if absent

        Returns:
            doc_id
        """
        doc_id = doc.get("doc_id") or str(uuid.uuid4())
        doc_id = str(doc_id)  # ensure it's a string for ChromaDB
        now = self._now_iso()

        metadata = self._flatten_meta(
            {
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "doc_type": doc.get("doc_type", "note"),
                "tags": doc.get("tags", ""),
                "author": doc.get("author", ""),
                "source_url": doc.get("source_url", ""),
                "created_at": doc.get("created_at", now),
                "updated_at": now,
            }
        )

        # Upsert — overwrite if doc_id already exists
        self._runbooks.upsert(
            ids=[doc_id],
            documents=[doc["content"]],
            metadatas=[metadata],
        )
        logger.info("Ingested runbook '%s' (%s)", doc.get("title"), doc_id)
        return doc_id

    def query_runbooks(
        self,
        query_text: str,
        n_results: int = 3,
        doc_type: str | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant runbooks for the current diagnosis context.

        Args:
            query_text: Current incident description / diagnosis text.
            n_results:  Number of results.
            doc_type:   Optional filter — runbook | post_mortem | playbook | note.

        Returns:
            List of dicts: {document, metadata, distance}
        """
        count = self._runbooks.count()
        if count == 0:
            return []

        where = {"doc_type": doc_type} if doc_type else None

        results = self._runbooks.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ------------------------------------------------------------------
    # Text document builders (for embedding quality)
    # ------------------------------------------------------------------

    @staticmethod
    def _incident_to_text(state: dict, metrics: dict) -> str:
        """Build a rich natural-language representation of an incident."""
        lines = [
            f"Incident severity: {state.get('severity', 'unknown')}.",
            f"Model: {metrics.get('model_id', 'unknown')} "
            f"version {metrics.get('model_version', 'unknown')} "
            f"in {metrics.get('environment', 'production')} environment.",
            f"Accuracy: {metrics.get('accuracy', 'N/A')}, "
            f"drift score: {metrics.get('drift_score', 'N/A')}, "
            f"p99 latency: {metrics.get('latency_p99_ms', 'N/A')} ms, "
            f"error rate: {metrics.get('error_rate', 'N/A')}.",
        ]

        if state.get("diagnosis"):
            lines.append(f"Diagnosis: {state['diagnosis']}")

        if state.get("recommended_action"):
            lines.append(f"Recommended action: {state['recommended_action']}.")

        if state.get("remediation_status"):
            lines.append(f"Remediation outcome: {state['remediation_status']}.")

        if state.get("report"):
            # Include just the first 400 chars of the report to keep embeddings focused
            lines.append(f"Report excerpt: {str(state['report'])[:400]}")

        return " ".join(lines)

    @staticmethod
    def _metrics_to_text(metrics: dict, severity: str) -> str:
        """Build a short natural-language summary of a metrics snapshot."""
        return (
            f"Model {metrics.get('model_id', 'unknown')} "
            f"version {metrics.get('model_version', 'unknown')} "
            f"environment {metrics.get('environment', 'production')}. "
            f"Severity: {severity or 'unknown'}. "
            f"Accuracy {metrics.get('accuracy', 'N/A')}, "
            f"drift {metrics.get('drift_score', 'N/A')}, "
            f"latency p99 {metrics.get('latency_p99_ms', 'N/A')} ms, "
            f"error rate {metrics.get('error_rate', 'N/A')}, "
            f"predictions {metrics.get('prediction_count', 'N/A')}."
        )
    
    def get_dynamic_thresholds(self, model_id: str) -> dict | None:
        """
        Retrieve the latest learned thresholds for a model.

        Thresholds are stored in the metrics_history collection with:
            metadata["type"] = "threshold_config"

        Returns:
            dict of thresholds OR None if not found
        """
        try:
            results = self._metrics.get(
                where=self._build_where_clause(
                    {
                        "model_id": model_id,
                        "type": "threshold_config",
                    }
                ),
                include=["metadatas"],
                limit=5,  # small window, we'll sort anyway
            )

            metas = results.get("metadatas") or []
            if not metas:
                return None

            # Sort newest first
            metas_sorted = sorted(
                metas,
                key=lambda m: m.get("updated_at", m.get("sampled_at", "")),
                reverse=True,
            )

            latest = metas_sorted[0]

            raw = latest.get("thresholds")

            if not raw:
                logger.warning("Threshold record found but no 'thresholds' field")
                return None

            # thresholds might be stored as JSON string
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    logger.error("Failed to parse stored thresholds JSON")
                    return None

            if not isinstance(raw, dict):
                logger.error("Invalid thresholds format (not dict)")
                return None

            return raw

        except Exception as e:
            logger.error("Error retrieving dynamic thresholds: %s", e)
            return None
