# Agentic MLOps — Self-Healing Fraud Detection

A production-style MLOps demo where deployed models **monitor themselves, diagnose drift, get human-approved on critical incidents, retrain, and hot-swap** — fully autonomous within guardrails. Built on LangGraph, MLflow, ChromaDB, with pluggable LLM backends (Ollama local / Google Gemini).

For the full architecture, components, state stores, and both healthy + remediation flows, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## The self-healing loop

```
   Generator ──▶ Model Server ◀── Watcher (MLflow polling)
       │              │                   ▲
       │              ▼                   │ new model + scalers
       │           Telemetry              │
       │              │                   │
       ▼              ▼                   │
   Real metrics + labels in window        │
       │                                  │
       ▼                                  │
   Monitor Agent ──▶ (none) ──▶ END       │
       │                                  │
       ▼ (minor/major/critical)           │
   Diagnosis Agent ──▶ Histogram-Drift    │
       │                                  │
       ▼ (critical/major)                 │
   Human Approval (HITL interrupt)        │
       │                                  │
       ▼ (approved)                       │
   Remediation ──▶ train.py subprocess ───┘
       │
       ▼
   Reporting ──▶ ChromaDB + threshold-manager
```

| Severity | Routing | What runs |
|---|---|---|
| `none`  | Monitor → END | Pipeline exits early |
| `minor` | Monitor → Diagnosis → Remediation → Reporting → END | No HITL |
| `major` / `critical` | Monitor → Diagnosis → **HITL** → Remediation → Reporting → END | Paused at human approval |

Every run persists to ChromaDB. Future diagnoses retrieve similar past incidents, runbooks, and metrics trends.

---

## Quickstart

One command launches everything:

```bash
git clone <repo-url> agentic_mlops && cd agentic_mlops
cp .env.example .env  # if you don't have one yet — see "Configuration" below
./setup.sh
```

`./setup.sh` does in order:

1. Creates/verifies `.venv` and runs `pip install -r requirements.txt`
2. Loads `.env` (safe parser — tolerates comments, quoted values, whitespace)
3. Validates `LLM_PROVIDER` (`ollama` default, `google` requires `GOOGLE_API_KEY`)
4. Pings Ollama and pulls `llama3.2:1b` + `nomic-embed-text` if missing
5. Verifies `mlops_agents/data/creditcard.csv` exists (instructs you on the `kaggle` download if not)
6. Detects your terminal emulator and launches three services in separate windows:
   - **Model Server** on `:8080`
   - **FastAPI** on `:8000`
   - **Streamlit Dashboard** on `:8501`

Flags:

```bash
./setup.sh --check          # preflight only — no launch
./setup.sh --no-install     # skip pip install (faster reruns)
./setup.sh --no-pull        # skip Ollama model pulls
./setup.sh --terminal=tmux  # force tmux instead of GUI emulator
```

Open `http://localhost:8501` to drive the system.

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 (3.12 tested) | Runtime |
| Ollama | latest | Local LLM inference (skip if using Gemini) |
| `creditcard.csv` | Kaggle `mlg-ulb/creditcardfraud` | Training dataset |
| Databricks MLflow workspace | optional | Default tracking backend |
| `kubectl` / `helm` | latest | Only if you wire up rollback / scale tools |

### Dataset

The trainer and scenario generators need the Kaggle dataset at `mlops_agents/data/creditcard.csv`. Download it once:

```bash
kaggle datasets download mlg-ulb/creditcardfraud -p ./mlops_agents/data/ --unzip
```

`./setup.sh` checks for the file and refuses to launch services until it's present.

---

## Configuration

Everything is `.env`-driven. Key vars:

```env
# LLM backend — switch with one line
LLM_PROVIDER=ollama                  # ollama | google

# Ollama (when LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Google Gemini (when LLM_PROVIDER=google)
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-2.0-flash

# Model identity
DEFAULT_MODEL_ID=main.default.fraud_classifier_v1
DEFAULT_ENVIRONMENT=production

# MLflow / Databricks Unity Catalog
MLFLOW_TRACKING_URI=https://<workspace>.cloud.databricks.com
MLFLOW_TRACKING_TOKEN=...
MLFLOW_EXPERIMENT_NAME=/Shared/fraud-detection

# Behaviour flags
LOCAL_MODE=true                      # spawn train.py locally vs. dispatch GitHub Actions
HUMAN_IN_THE_LOOP=true               # dynamic interrupt() in workflow
DRY_RUN=false

# Severity thresholds (deterministic floor — adaptive layer tunes from here)
THRESHOLD_ACCURACY_MAJOR=0.72
THRESHOLD_ACCURACY_CRITICAL=0.65
THRESHOLD_RECALL_MAJOR=0.75
THRESHOLD_RECALL_CRITICAL=0.60
THRESHOLD_ROC_AUC_MAJOR=0.85
THRESHOLD_ROC_AUC_CRITICAL=0.75
THRESHOLD_LATENCY_MAJOR_MS=1000
THRESHOLD_LATENCY_CRITICAL_MS=2000
THRESHOLD_ERROR_RATE_MAJOR=0.05
THRESHOLD_ERROR_RATE_CRITICAL=0.10

# Trainer
MIN_PRECISION=0.10                   # precision floor for the threshold tuner
RETRAIN_LOCK_MAX_AGE_SECONDS=7200    # zombie-lock guard

# Optional cost overrides (per 1M tokens, USD)
# LLM_COST_GEMINI_2_0_FLASH_INPUT=0.10
# LLM_COST_GEMINI_2_0_FLASH_OUTPUT=0.40

# Optional notifications
SLACK_NOTIFICATIONS_ENABLED=false
EMAIL_NOTIFICATIONS_ENABLED=false
EMAIL_MIN_SEVERITY=major
```

---

## Dashboard tour

The Streamlit dashboard at `:8501` is the operator UI.

| Page | Purpose |
|---|---|
| **Overview** | Trigger pipeline runs, see live model metrics, watch the active run unfold (status / severity / current agent), browse run history with per-run **token + cost totals**, view streaming training logs while retraining is in flight |
| **Incidents** | Browse the ChromaDB incident archive |
| **Approvals** | Approve or reject paused (critical-severity) runs — drives the LangGraph `interrupt()` resume |
| **Runbooks** | Add new runbooks for the RAG store |
| **Drift Lab** | Activate one of the scenario CSVs (`baseline` / `concept_drift` / `data_drift_amount` / …), start/stop the transaction generator at a chosen rate, watch live drift signals |

---

## Three-server architecture

| Service | Port | What it does | Hot-reload |
|---|---|---|---|
| **Model Server** (`fraud_model_server/model_server/server.py`) | 8080 | Serves `/predict`, captures predictions + labels in a 5K sliding window, computes live metrics, polls MLflow every 60s for new retrains and atomically swaps model + scalers + metadata + clears the deque | ✓ |
| **FastAPI Orchestrator** (`api/main.py`) | 8000 | HTTP wrapper around the LangGraph pipeline; manages run lifecycle, approvals, generator subprocess, retrain log tailing | — |
| **Streamlit Dashboard** (`dashboards/app.py`) | 8501 | Operator UI, polls the API | — |

The LangGraph workflow lives inside the FastAPI process. Each run is a checkpointed thread; HITL pauses are handled by `langgraph.types.interrupt()` and resumed via `Command(resume=True/False)`.

---

## Agents and tools

| Agent | Role | Determinism strategy |
|---|---|---|
| **Monitor** | Classify severity, narrate the breach | `severity_classifier` tool computes severity deterministically from metrics + thresholds; LLM only writes the human-readable reasoning |
| **Diagnosis** | Find root cause, prescribe retrain parameters | `histogram_drift` tool computes per-feature PSI / KS / mean-shift in NumPy; LLM consumes a structured summary, not raw histograms |
| **Remediation** | Dispatch action (retrain / rollback / scale / investigate) | Pure Python action dispatcher; subprocess management with zombie + age guards |
| **Reporting** | Generate report, persist incident, adapt thresholds | Threshold adaptation has a precision floor and bounded clamping |
| **Threshold Manager** | LLM-proposed threshold deltas, bounded + validated | Pydantic ±delta validation rejects out-of-bounds; clamping prevents drift into nonsense ranges |

Per-agent LLM token usage and cost are tracked via `mlops_agents/tools/token_tracker.py` (a LangChain `BaseCallbackHandler`) and displayed on the Overview page.

---

## Retraining

When the diagnosis agent prescribes `retrain` and a human approves:

1. The remediation agent spawns `train.py` as a subprocess via `mcp_tools.trigger_retraining_pipeline`. Stdout/stderr stream to `data/logs/retrain/<ts>-<model>.log` (unbuffered).
2. The dashboard's Overview page tails that log in a code panel while the process is alive (live indicator + auto-refresh).
3. `train.py` refits scalers if requested, fits a SMOTE-balanced LogisticRegression, tunes the decision threshold with a precision floor (default `MIN_PRECISION=0.10`) and F2 fallback if the precision floor + recall target are jointly unachievable. Result: never deploys a "predict-everything-as-fraud" model.
4. The new version is registered in MLflow, `champion` alias is promoted automatically.
5. Model Server's watcher detects the new run within 60s, atomically swaps model + both scalers + metadata, and clears the prediction window so subsequent metrics reflect the new model only.

Retraining lockfile has stale-detection guards: zombies, age > `RETRAIN_LOCK_MAX_AGE_SECONDS` (default 2h), unparseable timestamps, corrupt JSON — all self-heal.

---

## ChromaDB collections

| Collection | Holds | Read by |
|---|---|---|
| `incidents` | Full incident records (metrics, diagnosis, remediation, report) | Diagnosis Agent |
| `runs` | Pipeline runs with status, severity, prescription, token usage | FastAPI orchestrator |
| `runbooks` | Runbooks / playbooks ingested offline | Diagnosis Agent |
| `metrics_history` | Periodic metrics snapshots for trend detection | Monitor Agent, threshold manager |
| `dynamic_thresholds` | Adapted severity thresholds per model | Monitor Agent |

Default persists to `./rag_data/`. To use a remote ChromaDB:

```env
CHROMA_HOST=chroma.internal.yourdomain.com
CHROMA_PORT=8000
```

---

## Drift scenarios

Pre-built CSVs in `mlops_agents/data/datasets/` exercise different drift modes:

| Scenario | What changes vs. baseline | Expected severity |
|---|---|---|
| `baseline` | Nothing — original distribution | `none` |
| `concept_drift` | Label-to-feature relationship shifts | `critical` (recall + roc_auc collapse) |
| `data_drift_amount` | `Amount` feature distribution shifts | `major` / `critical` depending on window |
| `mixed` | Both | `critical` |

Drift Lab → click "Activate" on a scenario → start the generator → trigger a monitor run from Overview → watch the loop fire.

---

## CLI fallback (legacy)

The dashboard is the primary interface, but `main.py` still supports headless CLI flows:

```bash
python main.py init                              # one-time ChromaDB init
python main.py run --model-id <id> --environment production
python main.py resume --thread-id <id> --approve
```

These bypass the dashboard but use the same LangGraph workflow.

---

## Troubleshooting

**Ollama not reachable**
```bash
ollama serve   # or: systemctl --user start ollama
ollama list    # confirm llama3.2:1b and nomic-embed-text are present
```

**`creditcard.csv NOT FOUND`**
```bash
kaggle datasets download mlg-ulb/creditcardfraud -p ./mlops_agents/data/ --unzip
```

**`numpy.ufunc has no attribute __module__` during model load**
NumPy/SciPy version skew. Confirm `numpy<2.3` in `requirements.txt`, reinstall venv:
```bash
.venv/bin/pip install -r requirements.txt --upgrade
```

**Retrain stuck — lockfile shows zombie PID**
The new staleness guard handles this automatically on the next retrain attempt. To force-clear:
```bash
rm fraud_model_server/model_server/model/.retrain.lock
```

**Streamlit warning: "widget created with default value but also set via session state"**
Already fixed in Drift Lab and Overview pages. If you see it elsewhere, the pattern is: drop the `index=` arg from selectboxes that drive session_state from the API.

**MLflow run shows "Run already active" tracebacks**
Telemetry Worker collision with metrics-snapshot run — fixed in `server.py`. The worker now uses `client.log_artifact(run_id=...)` directly instead of opening a new active run.

**Model server reports `version 17` but metrics look like `v16`**
Sliding window still has pre-swap records. After my fixes the deque is cleared on swap; this only happens for old runs predating that change. Wait for the window to refill or restart the model server.

---

## Tech stack

| Component | Tech |
|---|---|
| Agent orchestration | LangGraph (StateGraph, MemorySaver, dynamic `interrupt()`) |
| LLM | Ollama (`llama3.2:1b`) or Google Gemini, pluggable via `LLM_PROVIDER` |
| Embeddings | `nomic-embed-text` via Ollama / Gemini embeddings |
| Vector DB | ChromaDB (PersistentClient or HttpClient) |
| Model | scikit-learn LogisticRegression, SMOTE-balanced |
| Model registry | MLflow / Databricks Unity Catalog |
| Drift detection | NumPy PSI / KS / mean-shift (deterministic; no LLM in the math path) |
| Token tracking | LangChain `BaseCallbackHandler` (provider-agnostic counts, model-specific cost lookup) |
| API | FastAPI + uvicorn |
| Dashboard | Streamlit |
| Notifications | Slack webhooks, SMTP, SendGrid (all optional) |

---

## Further reading

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — full component / state-store / flow reference for the architecture diagram
- `mlops_agents/agents/*.py` — agent implementations
- `mlops_agents/tools/severity_classifier.py` — deterministic severity rules
- `mlops_agents/tools/histogram_drift.py` — PSI / KS computation
- `mlops_agents/tools/token_tracker.py` — token + cost callback
- `fraud_model_server/model_server/server.py` — serving + hot-reload + telemetry
- `fraud_model_server/model_server/scripts/train.py` — trainer with precision-floor threshold tuner
