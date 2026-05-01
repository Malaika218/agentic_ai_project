# 🤖 Multi-Agent MLOps Monitoring & Autonomous Remediation System

A production-grade, agentic AI pipeline that **monitors deployed ML models, diagnoses degradation, autonomously remediates, and learns from every incident** — powered by LangGraph, Ollama, and ChromaDB.

---

## How It Works

```
[Monitor Agent] → route_by_severity
    ├── "none"     → END  (healthy, nothing to do)
    ├── "minor"    → [Diagnosis] → [Remediation] → [Reporting] → END
    ├── "critical" → [Diagnosis] → [Remediation] → [Reporting] → END
    └── "major"    → [Diagnosis] → [Human Approval] → [Remediation] → [Reporting] → END
```

Each run saves the incident to ChromaDB. Future diagnosis queries retrieve similar past incidents, runbooks, and metrics trends — making the system **progressively smarter over time**.

---

## Project Structure

```
mlops_agents/
├── main.py                        # Entry point — CLI for run / resume / init
├── state.py                       # Shared LangGraph AgentState TypedDict
├── requirements.txt
├── .env.example                   # All configuration options (copy to .env)
│
├── agents/
│   ├── monitor_agent.py           # Fetches real-time metrics, classifies severity
│   ├── diagnosis_agent.py         # RAG-enriched LLM root cause analysis
│   ├── remediation_agent.py       # Dispatches retrain / rollback / scale / investigate
│   └── reporting_agent.py         # Generates report, saves to RAG, sends notifications
│
├── graph/
│   └── workflow.py                # LangGraph StateGraph + conditional routing
│
├── rag/
│   ├── store.py                   # ChromaDB RAG store (3 collections)
│   └── init_collections.py        # One-time schema initialisation script
│
├── tools/
│   ├── metrics_source.py          # Prometheus / Azure Monitor / MLflow adapters
│   └── mcp_tools.py               # GitHub Actions, Kubernetes, Helm, Slack, email
│
└── scripts/
    └── ingest_runbooks.py         # Offline bulk-ingest runbooks into ChromaDB
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| Ollama | Latest | Local LLM inference |
| `llama3.2:1b` model | — | Pulled via Ollama |
| `nomic-embed-text` model | — | Pulled via Ollama |
| kubectl + helm | Latest | For remediation tools (optional for dev) |

---

## Quickstart

### 1. Clone & set up the environment

```bash
git clone https://github.com/your-org/mlops_agents.git
cd mlops_agents

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# Linux / Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download installer from https://ollama.com

# Pull the required models
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

```env
# Which metrics source to use
METRICS_SOURCE=prometheus          # or: azure | mlflow

# The model you want to monitor
DEFAULT_MODEL_ID=your-model-name
DEFAULT_ENVIRONMENT=production

# Prometheus (if METRICS_SOURCE=prometheus)
PROMETHEUS_URL=http://localhost:9090

# GitHub (needed for retrain + investigate actions)
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
GITHUB_OWNER=your-org
GITHUB_REPO=your-repo
GITHUB_RETRAIN_WORKFLOW_ID=retrain.yml

# Kubernetes (needed for rollback + scale actions)
K8S_NAMESPACE=ml-production

# Slack notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Email alerts
SMTP_HOST=smtp.yourdomain.com
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=your-password
EMAIL_FROM=alerts@yourdomain.com
EMAIL_TO_CRITICAL=oncall@yourdomain.com
EMAIL_TO_MAJOR=ml-team@yourdomain.com
```

> All other fields have sensible defaults. See `.env.example` for the full list.

### 4. Initialise ChromaDB collections

```bash
python main.py init
```

This creates the three ChromaDB collections (`incidents`, `metrics_history`, `runbooks`) in `./rag_data/`. Safe to re-run — it is idempotent.

### 5. (Optional) Ingest runbooks

```bash
# Ingest all .md / .txt files from a directory
python scripts/ingest_runbooks.py --dir ./docs/runbooks

# Ingest from a JSON manifest
python scripts/ingest_runbooks.py --manifest ./docs/runbooks.json

# Ingest a single file
python scripts/ingest_runbooks.py --file ./docs/retrain-runbook.md \
    --title "Retraining Runbook" --doc-type runbook --tags "retrain,drift"
```

### 6. Run the pipeline

```bash
python main.py run --model-id your-model-name --environment production
```

---

## All CLI Commands

```bash
# Initialise ChromaDB collections (run once)
python main.py init

# Run a full monitoring cycle
python main.py run --model-id <model-id> --environment <production|staging|canary>

# Run with a specific thread ID (for tracking / resuming)
python main.py run --model-id fraud-classifier-v2 --thread-id my-thread-123

# Resume a pipeline paused at human approval (major severity)
python main.py resume --thread-id <thread-id> --approve
python main.py resume --thread-id <thread-id> --reject

# Skip the Ollama connectivity check (useful in CI)
python main.py run --model-id fraud-classifier-v2 --skip-connectivity-check
```

---

## Metrics Sources

Set `METRICS_SOURCE` in `.env` to one of the following:

### `prometheus` (default)

```env
METRICS_SOURCE=prometheus
PROMETHEUS_URL=http://prometheus.monitoring.svc.cluster.local:9090
PROMETHEUS_BEARER_TOKEN=           # optional
```

Your Prometheus instance must expose these metrics from your ML serving layer:

```
mlops_model_accuracy{model_id="...", environment="..."}
mlops_model_drift_score{...}
mlops_request_latency_seconds_bucket{...}   # histogram
mlops_request_errors_total{...}
mlops_predictions_total{...}
```

### `azure`

```env
METRICS_SOURCE=azure
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
AZURE_SUBSCRIPTION_ID=...
AZURE_RESOURCE_GROUP=...
AZURE_MONITOR_WORKSPACE_ID=...
AZURE_ML_WORKSPACE_NAME=...
```

Install the extra dependencies:

```bash
pip install azure-identity azure-monitor-query azure-ai-ml
```

### `mlflow`

```env
METRICS_SOURCE=mlflow
MLFLOW_TRACKING_URI=http://mlflow.svc:5000
MLFLOW_EXPERIMENT_NAME=production-model-evals
```

Install the extra dependency:

```bash
pip install mlflow
```

---

## Remediation Actions

| Action | When triggered | What it does |
|--------|---------------|--------------|
| `retrain` | High drift / stale model | Dispatches GitHub Actions `workflow_dispatch` |
| `rollback` | Recent deployment regression | Runs `helm rollback` via subprocess |
| `scale` | Latency spike, accuracy healthy | Scales Kubernetes deployment replicas |
| `investigate` | Ambiguous root cause | Opens a GitHub Issue with full context |

---

## Severity Thresholds

All thresholds are configurable via `.env` — no code changes needed:

```env
THRESHOLD_ACCURACY_CRITICAL=0.65
THRESHOLD_ACCURACY_MAJOR=0.72
THRESHOLD_ACCURACY_MINOR=0.80
THRESHOLD_DRIFT_CRITICAL=0.60
THRESHOLD_DRIFT_MAJOR=0.35
THRESHOLD_DRIFT_MINOR=0.20
THRESHOLD_LATENCY_CRITICAL_MS=2000
THRESHOLD_LATENCY_MAJOR_MS=1000
THRESHOLD_ERROR_RATE_CRITICAL=0.10
THRESHOLD_ERROR_RATE_MAJOR=0.05
```

Grey-zone cases (between thresholds) are automatically escalated to the LLM for classification.

---

## Human-in-the-Loop

For **major** severity incidents, the pipeline pauses before executing remediation:

```
[Diagnosis] → ⏸ PAUSED — human approval required → [Remediation]
```

You will see in the terminal:

```
Pipeline paused at human approval checkpoint.
Thread ID: abc-123
Resume with: python main.py resume --thread-id abc-123 --approve
```

Then either approve or reject:

```bash
python main.py resume --thread-id abc-123 --approve   # proceed with remediation
python main.py resume --thread-id abc-123 --reject    # cancel, end pipeline
```

To disable human approval (auto-approve all major incidents):

```env
HUMAN_IN_THE_LOOP=false
```

---

## ChromaDB RAG Collections

| Collection | Contents | Queried by |
|------------|----------|-----------|
| `incidents` | Full incident records — metrics, diagnosis, remediation outcome, report | Diagnosis Agent |
| `metrics_history` | Lightweight metric snapshots from every monitor cycle | Monitor Agent, Diagnosis Agent |
| `runbooks` | Runbooks, post-mortems, playbooks ingested offline | Diagnosis Agent |

ChromaDB persists to `./rag_data/` by default. To use a remote ChromaDB server:

```env
CHROMA_HOST=chroma.internal.yourdomain.com
CHROMA_PORT=8000
```

---

## Notifications

**Slack** — configure one of:

```env
# Option A: Incoming Webhook (simpler)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ

# Option B: Bot Token with per-severity channel routing
SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx
SLACK_SEVERITY_CHANNELS={"critical":"#incidents","major":"#mlops-alerts"}
```

**Email** — configure SMTP or SendGrid:

```env
# SMTP (default)
ALERT_EMAIL_PROVIDER=smtp
SMTP_HOST=smtp.yourdomain.com
SMTP_PORT=587
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=...

# SendGrid
ALERT_EMAIL_PROVIDER=sendgrid
SENDGRID_API_KEY=SG.xxxxxxxxxxxxx
```

Set the minimum severity that triggers emails (default: `major`):

```env
EMAIL_MIN_SEVERITY=major    # emails sent for major + critical only
```

---

## Troubleshooting

**`Cannot reach Ollama`**
```bash
# Make sure Ollama is running
ollama serve

# Verify the model is pulled
ollama list
```

**`MetricsSourceError`**
- Check your `METRICS_SOURCE` env var is set correctly
- Verify the endpoint is reachable from your machine
- For Prometheus: confirm your ML serving layer is exporting the expected metric names

**`ChromaDB collection not found`**
```bash
# Re-run the init script
python main.py init
```

**`ModuleNotFoundError`**
```bash
# Make sure you activated your virtualenv
source venv/bin/activate
pip install -r requirements.txt
```

**Pipeline paused unexpectedly**
- This is expected for `major` severity — see [Human-in-the-Loop](#human-in-the-loop) above
- Use `python main.py resume --thread-id <id> --approve` to continue

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph (StateGraph, MemorySaver, GraphInterrupt) |
| LLM Inference | Ollama — `llama3.2:1b` (INT4, ~2 GB) |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector Database | ChromaDB (local PersistentClient or remote HttpClient) |
| Metrics Sources | Prometheus, Azure Monitor, MLflow |
| Remediation Tools | Kubernetes Python client, Helm CLI, GitHub REST API v3 |
| Notifications | Slack Blocks API, SMTP, SendGrid |
| Deployment | Docker, Helm, Azure Kubernetes Service (AKS) |
