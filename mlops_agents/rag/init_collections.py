"""
rag/init_collections.py

ChromaDB collection schema initialisation script.
Run once before first use, or safely re-run (idempotent).

Usage:
    python -m rag.init_collections
    # or directly:
    python rag/init_collections.py
"""

from dotenv import load_dotenv
import logging
import os
import sys
from datetime import datetime
from mlops_agents.rag.store import RAGStore
 

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("rag.init_collections")


load_dotenv()


# ---------------------------------------------------------------------------
# Collection definitions
# ---------------------------------------------------------------------------

COLLECTIONS: list[dict] = [
    {
        "name": "incidents",
        "description": (
            "Full incident records — one document per resolved incident. "
            "Each document contains a serialised JSON payload with metrics, "
            "diagnosis, remediation outcome, and a human-readable report. "
            "Used by the Diagnosis Agent to retrieve the top-k most similar "
            "past incidents for in-context learning."
        ),
        "metadata_schema": {
            # Fields stored in the ChromaDB metadata dict for each document.
            # ChromaDB metadata values must be str | int | float | bool.
            "incident_id":          "str   — UUID v4",
            "model_id":             "str   — deployed model identifier",
            "model_version":        "str   — semver or tag",
            "environment":          "str   — production | staging | canary",
            "severity":             "str   — none | minor | major | critical",
            "recommended_action":   "str   — retrain | rollback | scale | investigate",
            "remediation_status":   "str   — success | failed | skipped",
            "human_approved":       "bool  — whether human approval was required",
            "accuracy":             "float — model accuracy at incident time",
            "drift_score":          "float — data drift score at incident time",
            "latency_p99_ms":       "float — p99 latency in ms at incident time",
            "error_rate":           "float — error rate fraction (0–1)",
            "created_at":           "str   — ISO-8601 UTC timestamp",
            "resolved_at":          "str   — ISO-8601 UTC timestamp or empty string",
        },
    },
    {
        "name": "metrics_history",
        "description": (
            "Lightweight metric snapshots saved after every monitor cycle — "
            "including healthy runs. Used by the Monitor Agent to retrieve "
            "recent trend windows for anomaly context, and by the Reporting "
            "Agent to generate aggregate statistics."
        ),
        "metadata_schema": {
            "snapshot_id":      "str   — UUID v4",
            "model_id":         "str   — deployed model identifier",
            "model_version":    "str   — semver or tag",
            "environment":      "str   — production | staging | canary",
            "severity":         "str   — none | minor | major | critical",
            "accuracy":         "float — model accuracy",
            "drift_score":      "float — data drift score",
            "latency_p99_ms":   "float — p99 latency in ms",
            "error_rate":       "float — error rate fraction (0–1)",
            "prediction_count": "int   — number of predictions in window",
            "sampled_at":       "str   — ISO-8601 UTC timestamp",
        },
    },
    {
        "name": "runbooks",
        "description": (
            "Institutional knowledge base — runbooks, post-mortems, "
            "remediation playbooks, and engineering notes ingested "
            "offline. Queried by the Diagnosis Agent to ground its "
            "recommendations in org-specific procedures."
        ),
        "metadata_schema": {
            "doc_id":       "str — UUID v4 or slug",
            "title":        "str — human-readable document title",
            "doc_type":     "str — runbook | post_mortem | playbook | note",
            "tags":         "str — comma-separated keyword tags",
            "author":       "str — author name or team slug",
            "source_url":   "str — Confluence / GitHub URL or empty string",
            "created_at":   "str — ISO-8601 date",
            "updated_at":   "str — ISO-8601 date",
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_chroma_client() -> chromadb.ClientAPI:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./rag_data")
    host = os.getenv("CHROMA_HOST", "")
    port = int(os.getenv("CHROMA_PORT", "8000"))

    if host:
        logger.info("Connecting to remote ChromaDB at %s:%s", host, port)
        return chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )

    logger.info("Using local PersistentClient at '%s'", persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_embedding_function() -> embedding_functions.EmbeddingFunction:
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.info("Embedding model: %s  (Ollama @ %s)", embed_model, ollama_url)
    return embedding_functions.OllamaEmbeddingFunction(
        model_name=embed_model,
        url=f"{ollama_url}/api/embeddings",
    )


# ---------------------------------------------------------------------------
# Main init
# ---------------------------------------------------------------------------

def init_collections(reset: bool = False) -> dict[str, chromadb.Collection]:
    """
    Create (or verify) all ChromaDB collections.

    Args:
        reset: If True, delete and recreate every collection.
               USE WITH CAUTION — destroys all stored data.

    Returns:
        Dict mapping collection name → Collection object.
    """
    client = _get_chroma_client()
    embed_fn = _get_embedding_function()
    created: dict[str, chromadb.Collection] = {}

    for spec in COLLECTIONS:
        name = spec["name"]

        if reset:
            try:
                client.delete_collection(name)
                logger.warning("Deleted collection '%s' (reset=True)", name)
            except Exception:
                pass  # did not exist yet

        try:
            col = client.get_collection(name=name, embedding_function=embed_fn)
            count = col.count()
            logger.info(
                "Collection '%s' already exists — %d documents. Skipping creation.",
                name,
                count,
            )
        except Exception:
            col = client.create_collection(
                name=name,
                embedding_function=embed_fn,
                metadata={
                    "description": spec["description"],
                    "schema_version": "1.0",
                    "created_at": datetime.utcnow().isoformat(),
                    "hnsw:space": "cosine",          # cosine similarity for RAG
                },
            )
            logger.info("Created collection '%s'.", name)

        created[name] = col

    logger.info("Initialisation complete. Collections: %s",
                list(created.keys()))
    return created


def seed_runbooks() -> None:
    RUNBOOKS = [

        {
            "doc_id": 1,
            "title":    "Concept Drift — Fraud Pattern Shift Runbook",
            "doc_type": "runbook",
            "tags":     "concept_drift,recall,retrain,fraud,drift_period_only",
            "author":   "ml-platform-team",
            "content":  """# Concept Drift — Fraud Pattern Shift Runbook
 
## What It Is
Concept drift occurs when the statistical relationship between input features and the fraud label changes.
Fraudsters change techniques. Previously V14 high = fraud. Now fraud exploits different feature combinations.
The model is confidently wrong — not uncertain.
 
## How to Identify
- recall < 0.55 (key signal — model missing fraud)
- precision remains high (model still confident, just wrong)
- drift_score moderate (0.20–0.45) — inputs look similar but labels changed
- accuracy drop lags recall drop by several cycles
- prediction confidence distribution remains bimodal (model not uncertain, just wrong)
 
## How to Distinguish from Data Drift
- Data drift: drift_score high (>0.45), model uncertain, probabilities cluster near 0.5
- Concept drift: drift_score moderate, precision high, recall collapsed
- Mixed: both signals present simultaneously
 
## Remediation Steps
1. Confirm recall < 0.55 for at least 2 consecutive monitor cycles
2. Retrain with strategy: drift_period_only
3. Set drift_period_weight=2.0 (upsample recent data where new pattern is present)
4. Set optimize_for=recall — catching fraud is more important than precision
5. Set target_recall=0.82 minimum
6. Deploy with canary strategy at 10% traffic
7. Monitor recall for 2 hours before promoting to 100%
 
## Prescription Template
- data_strategy: drift_period_only
- window_days: 14 (focus on recent period where new pattern emerged)
- drift_period_weight: 2.0
- optimize_for: recall
- target_recall: 0.82
- target_roc_auc: 0.88
- deployment_strategy: canary
- canary_traffic_pct: 10
 
## Post-Remediation Checks
- recall should recover to > 0.80 within 2 monitor cycles
- if recall recovers but precision drops below 0.40, widen window_days to 21
- if recall does not recover, the new fraud pattern may require feature engineering — escalate to data science team
 
## Historical Outcomes
- 2024-Q1: V14/V17 pattern shift. drift_period_only + recall optimization resolved in 3 cycles.
- 2024-Q3: Amount-based fraud pattern. Required feature engineering alongside retrain.
""",
        },

        {
            "doc_id": 2,
            "title":    "Data Drift — Covariate Shift Runbook",
            "doc_type": "runbook",
            "tags":     "data_drift,covariate_shift,retrain,recent_window,psi",
            "author":   "ml-platform-team",
            "content":  """# Data Drift — Covariate Shift Runbook
 
## What It Is
The statistical distribution of input features changes. A new merchant category enters the platform.
Geographic expansion brings different transaction patterns. Seasonal effects shift Amount distribution.
The model sees unfamiliar inputs — not a label relationship change.
 
## How to Identify
- drift_score > 0.35 (primary signal)
- accuracy degrades gradually (not suddenly)
- recall and accuracy degrade together (model uncertain, not wrong)
- prediction confidence distribution shifts toward 0.5 (uncertainty rising)
- latency unaffected
 
## Remediation Steps
1. Identify which features drifted — check per-feature PSI if available
2. Retrain with strategy: recent_window
3. window_days: derive from drift onset. Recent drift (< 7 days) use 14. Gradual (weeks) use 30–60.
4. Set drift_period_weight=1.5 if drift_score 0.35–0.55, 2.0 if > 0.55
5. refit_preprocessors=true if Amount or Time features drifted (scaler needs update)
6. Deploy with canary for drift_score > 0.45, immediate for 0.35–0.45
 
## Prescription Template
- data_strategy: recent_window
- window_days: 21 (adjust based on drift onset)
- drift_period_weight: 1.5–2.0
- refit_preprocessors: true if Amount_scaled or Time_scaled in drifted_features
- optimize_for: recall
- deployment_strategy: canary (drift_score > 0.45) or immediate
 
## Common Causes
- New merchant category onboarded (Amount distribution shifts right)
- Geographic expansion (V-feature distributions shift)
- Seasonal patterns (Time feature distribution shifts)
- Upstream data pipeline schema change (sudden step-function drift)
 
## If Drift Recurs Within 7 Days
Suspect upstream data pipeline change rather than natural distribution shift.
Check: feature encoding, normalization pipeline, data source schema.
Action: investigate rather than retrain again.
 
## Post-Remediation Checks
- drift_score should fall below 0.20 within 3 monitor cycles
- if drift persists, the new distribution may be a permanent shift — update reference distribution
""",
        },

        {
            "doc_id": 3,
            "title":    "Latency Spike — Infrastructure Runbook",
            "doc_type": "runbook",
            "tags":     "latency,infrastructure,scale,kubernetes,not_retrain",
            "author":   "ml-platform-team",
            "content":  """# Latency Spike — Infrastructure Runbook
 
## What It Is
p99 or p95 latency exceeds threshold while model accuracy and drift remain healthy.
This is an infrastructure problem, not a model quality problem. Do not retrain.
 
## How to Identify
- latency_ms > 1000 (major) or > 2000 (critical)
- accuracy within normal range (> 0.80)
- drift_score within normal range (< 0.25)
- error_rate may be rising due to timeouts, not model failures
 
## CRITICAL
If accuracy is also degraded, do not assume latency is the root cause.
High latency + low accuracy = possible model complexity issue from recent retrain.
Check model version and retrain date before scaling.
 
## Remediation Steps
1. Confirm accuracy and drift are healthy before acting
2. Action: scale (double replica count)
3. Check pod CPU/memory — OOMKill causes latency spikes
4. Check if a recent deployment changed model complexity (GBM depth increase)
5. Do NOT retrain — retraining does not fix serving infrastructure
 
## Prescription
- recommended_action: scale
- no retrain prescription needed
- if scaling does not resolve within 5 minutes, investigate pod logs
 
## Kubernetes Commands (if manual intervention needed)
kubectl get pods -n ml-production
kubectl top pods -n ml-production
kubectl describe pod <pod-name> -n ml-production
 
## Post-Remediation Checks
- latency should recover within 3 minutes of scaling
- if latency persists after scaling, check node-level metrics
- consider adding HPA (Horizontal Pod Autoscaler) if spikes are recurring
 
## When to Escalate
- latency spike coincides with a recent deployment → rollback
- latency spike with OOMKill → increase memory limits
- latency spike with no other signals and no recent changes → investigate
""",
        },

        {
            "doc_id": 4,
            "title":    "High Error Rate — Model Serving Failure Runbook",
            "doc_type": "runbook",
            "tags":     "error_rate,serving,schema,investigate,malformed",
            "author":   "ml-platform-team",
            "content":  """# High Error Rate — Model Serving Failure Runbook
 
## What It Is
error_rate > 0.05 (major) or > 0.10 (critical).
Inference requests are failing — not returning wrong predictions, failing entirely.
Distinct from accuracy degradation which is wrong predictions not failed ones.
 
## How to Identify
- error_rate rising (this is computed from real inference exceptions, not estimated)
- may accompany latency spike (failed requests still consume time)
- accuracy metrics may be misleading (failed requests excluded from accuracy computation)
 
## Common Causes
1. Schema change upstream — transaction payload missing fields or wrong types
2. NaN or null values in features — upstream data pipeline bug
3. Model artifact corruption — unlikely but possible after bad retrain
4. Memory error during inference — model too large for pod memory
 
## Remediation Decision Tree
- error_rate high + accuracy healthy + recent schema change → investigate (open GitHub issue)
- error_rate high + accuracy healthy + no schema change → investigate (check serving logs)
- error_rate high + accuracy degraded + recent retrain → rollback
- error_rate high + latency high + accuracy healthy → scale first, then investigate
 
## What NOT to Do
- Do not retrain to fix error_rate — retrain does not fix serving failures
- Do not rollback without confirming accuracy also degraded
 
## Prescription
- recommended_action: investigate
- open GitHub issue with error logs and recent deployment history
- include sample of malformed requests if available
 
## Post-Remediation Checks
- error_rate should drop to < 0.01 within 2 monitor cycles after fix
- monitor for 24 hours after schema fix to confirm stability
""",
        },

        {
            "doc_id": 5,
            "title":    "Post-Retrain Regression — Rollback Runbook",
            "doc_type": "runbook",
            "tags":     "rollback,regression,retrain,deployment,helm",
            "author":   "ml-platform-team",
            "content":  """# Post-Retrain Regression — Rollback Runbook
 
## What It Is
Accuracy or recall degrades immediately after a new model deployment.
The retrained model performs worse than the model it replaced.
Distinct from gradual drift — regression is sudden and correlates with deployment event.
 
## How to Identify
- accuracy drops within 1–2 monitor cycles of a deployment
- previous model_version tag in metrics_history shows healthy metrics
- current model_version tag shows degraded metrics
- drift_score may be low (not a data problem — model problem)
 
## Remediation Steps
1. Confirm degradation started after deployment (check metrics_history timestamps vs deployment time)
2. Confirm drift_score is NOT the primary signal (if drift high, retrain again with better prescription)
3. Action: rollback
4. Helm rollback to previous revision
5. Verify metrics recover within 2 monitor cycles
 
## When Rollback is Correct
- accuracy dropped > 0.08 points immediately after deployment
- recall dropped > 0.10 points immediately after deployment
- no corresponding drift_score increase
 
## When Rollback is WRONG
- drift is high — rolling back to the old model returns a model that already failed on new distribution
- error_rate is the primary signal — rollback won't fix serving infrastructure
 
## Post-Rollback Checks
- metrics should return to pre-deployment baseline within 2 cycles
- if metrics do not recover after rollback, the degradation is not deployment-related — investigate
 
## Root Cause Analysis After Rollback
- Review retrain prescription: was window_days appropriate?
- Review validation gate: did roc_auc and recall pass thresholds?
- If validation passed but production failed: test set may not represent production distribution
- Increase canary_traffic_pct monitoring period before next promotion
""",
        },

        {
            "doc_id": 6,
            "title":    "Gradual Accuracy Decay — Model Staleness Runbook",
            "doc_type": "runbook",
            "tags":     "staleness,accuracy,gradual,retrain,full_history,weighted_recent",
            "author":   "ml-platform-team",
            "content":  """# Gradual Accuracy Decay — Model Staleness Runbook
 
## What It Is
Accuracy declines slowly over weeks. No sudden event. No clear drift signal.
The model was trained months ago. Fraud patterns evolve. Model becomes increasingly stale.
 
## How to Identify
- accuracy declining 0.01–0.03 per week over 3+ weeks
- drift_score mildly elevated (0.15–0.30) but not alarming
- recall declining alongside accuracy (not a concept drift pattern)
- metrics_history trend shows sustained downward slope
 
## How to Distinguish from Concept/Data Drift
- Staleness: slow decline over weeks, mild drift signal, recall and accuracy decay together
- Concept drift: sudden recall collapse, precision stays high
- Data drift: drift_score > 0.35 is the primary signal
 
## Remediation Steps
1. Confirm trend is sustained — at least 3 consecutive monitor cycles declining
2. Check days since last retrain — if > 30 days, staleness is likely primary cause
3. Retrain with strategy: weighted_recent or full_history
4. weighted_recent: use all historical data but upsample last 30 days
5. full_history: if fraud patterns have shifted seasonally and old data is still relevant
6. Upgrade model architecture if appropriate — LogisticRegression → GradientBoosting
 
## Prescription Template
- data_strategy: weighted_recent
- window_days: 60 (captures recent trend without discarding history)
- drift_period_weight: 1.5
- optimize_for: recall
- target_recall: 0.80
- deployment_strategy: canary (gradual promotion, monitor for regression)
 
## Scheduling Note
Consider proactive retraining on a schedule (every 30 days) rather than waiting for degradation.
Reactive retraining for staleness always results in some period of degraded performance.
 
## Post-Remediation Checks
- accuracy should recover to > 0.88 within 3 monitor cycles
- if accuracy does not recover, staleness was not the primary cause — investigate drift
""",
        },

        {
            "doc_id": 7,
            "title":    "Mixed Drift — Simultaneous Data and Concept Drift Runbook",
            "doc_type": "runbook",
            "tags":     "mixed_drift,concept_drift,data_drift,critical,drift_period_only",
            "author":   "ml-platform-team",
            "content":  """# Mixed Drift — Simultaneous Data and Concept Drift Runbook
 
## What It Is
Both the input distribution and the fraud label relationship have changed simultaneously.
The most severe and hardest-to-diagnose failure mode.
Typically occurs during major platform changes or large-scale fraud pattern shifts.
 
## How to Identify
- drift_score high (> 0.45) AND recall collapsed (< 0.55)
- accuracy severely degraded
- prediction confidence distribution unpredictable
- severity: critical in almost all cases
 
## Why It's Hard to Diagnose
- drift_score rising could be data drift alone
- recall collapse could be concept drift alone
- together they are mixed and require aggressive remediation
 
## Remediation Steps
1. Do not attempt to separate the signals — treat as mixed
2. data_strategy: drift_period_only (focus entirely on recent data where both shifts occurred)
3. drift_period_weight: 2.0 (upsample aggressively)
4. window_days: 14 (tight window — only the period after both shifts)
5. optimize_for: recall (fraud safety is priority)
6. deployment_strategy: canary with extended shadow period (4 hours minimum)
7. Do NOT use full_history — historical data reflects neither the new distribution nor the new patterns
 
## Prescription Template
- data_strategy: drift_period_only
- window_days: 14
- drift_period_weight: 2.0
- optimize_for: recall
- target_recall: 0.82
- target_roc_auc: 0.88
- deployment_strategy: canary
- canary_traffic_pct: 10
- shadow_period_hours: 4
 
## If Retrain Fails Validation Gate
- roc_auc or recall below threshold after training on drift period
- the drift period may be too short to train a good model
- increase window_days to 21 and retrain again
- if still failing: escalate to data science team for feature engineering
 
## Post-Remediation Checks
- both drift_score and recall must recover
- drift_score < 0.20 AND recall > 0.80 before considering resolved
- monitor for 48 hours after full promotion — mixed drift can recur
""",
        },

        {
            "doc_id": 8,
            "title":    "2024-Q3 Production Post-Mortem — Amount Feature Drift",
            "doc_type": "post_mortem",
            "tags":     "post_mortem,data_drift,amount,merchant,retrain,2024",
            "author":   "oncall-team",
            "content":  """# Post-Mortem: Amount Feature Drift — 2024 Q3
 
## Incident Summary
Duration: 4 hours 22 minutes
Severity: Major
Impact: fraud detection recall dropped from 0.88 to 0.61
Root cause: New premium merchant category onboarded — Amount distribution shifted significantly
 
## Timeline
- 14:00 UTC: New merchant category enabled in production
- 14:35 UTC: Monitor agent detected drift_score=0.41 → major severity
- 14:38 UTC: Human approval requested (major severity gate)
- 14:51 UTC: Approval given, retrain dispatched
- 16:47 UTC: Retrain completed, model hot-reloaded
- 18:22 UTC: Metrics confirmed recovered, incident closed
 
## Root Cause
Premium merchant transactions average $2,400 vs $85 baseline.
Amount_scaled feature shifted +3.2 standard deviations from training distribution.
Model was trained almost entirely on sub-$500 transactions.
High-value legitimate transactions were being flagged as fraud (false positives).
Recall dropped because the fraud boundary shifted with the Amount distribution.
 
## What Worked
- Monitor agent detected drift within 35 minutes of merchant enablement
- RAG retrieved similar historical incident (2023-Q4 geographic expansion)
- Diagnosis correctly identified data_drift with Amount_scaled as primary feature
- Prescription used recent_window with refit_preprocessors=true (scaler needed update)
 
## What Did Not Work
- Human approval added 13 minutes — major severity gate was correct but slow
- Retrain took 1h56m — GradientBoosting on full window was slower than needed
- Should have used drift_period_only with shorter window (14 days sufficient)
 
## Action Items
- Add Amount_scaled to feature drift monitoring dashboard
- Pre-stage scaler refit when new merchant categories are onboarded
- Consider reducing major severity approval window to 5 minutes for known drift patterns
- Runbook updated with Amount-specific prescription
 
## Prescription That Resolved It
- data_strategy: recent_window
- window_days: 30
- drift_period_weight: 1.5
- refit_preprocessors: true
- drifted_features: [Amount_scaled]
- optimize_for: recall
- deployment_strategy: canary
""",
        },

    ]
    print("Connecting to ChromaDB...")
    rag = RAGStore()

    if rag._runbooks.count() > 0:
        print(f"Runbooks collection already has {rag._runbooks.count()} documents.")
        print("Skipping seeding to avoid duplicates.")
        return
    
    for doc in RUNBOOKS:
        doc_id = rag.ingest_runbook(doc)
        print(f"  ✓ {doc['title'][:60]}")
        print(f"    type={doc['doc_type']} tags={doc['tags'][:50]}")
        print(f"    id={doc_id[:32]}…\n")

    total = rag._runbooks.count()
    print(f"Done. ChromaDB runbooks collection now has {total} documents.")
    print("\nTest a query to verify retrieval:")
    print("  python scripts/inject_metrics.py --list")
    print("  Or use the Runbooks page → Test Query tab in the dashboard.")

def init() -> None:
    print("Running initialization script for RAG system")
    print("Initializing collections")
    init_collections()
    print("Seeding starter runbooks")
    seed_runbooks()
    print("Initialization script now complete")

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialise ChromaDB collections for the MLOps agent system."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all collections. Destroys all stored data.",
    )
    args = parser.parse_args()

    if args.reset:
        confirm = input(
            "\n⚠️  --reset will DELETE all collection data. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    init_collections(reset=args.reset)
    seed_runbooks()

    print("\n✅  ChromaDB collections ready.")
