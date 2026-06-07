#!/usr/bin/env bash
# setup.sh — bring the agentic-MLOps demo up locally.
#
# 1. Creates/verifies the .venv
# 2. Installs requirements
# 3. Loads .env and verifies critical vars
# 4. Verifies Ollama is reachable and pulls llama3.2:1b + nomic-embed-text
#    (or whatever OLLAMA_MODEL / OLLAMA_EMBED_MODEL is set to)
# 5. Launches model server (8080), FastAPI (8000), Streamlit (8501) — each in a separate terminal.
#
# Usage:
#   ./setup.sh                  Full setup + launch
#   ./setup.sh --check          Preflight only — no launch
#   ./setup.sh --no-install     Skip pip install (faster reruns)
#   ./setup.sh --no-pull        Skip Ollama model pulls
#   ./setup.sh --terminal=tmux  Force tmux instead of GUI emulator
#   ./setup.sh --help

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ── colours ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; CYAN=$'\e[36m'; DIM=$'\e[2m'; NC=$'\e[0m'
else
    RED=; GREEN=; YELLOW=; CYAN=; DIM=; NC=
fi

log()      { printf "${CYAN}[setup]${NC} %s\n" "$*"; }
ok()       { printf "${GREEN}[ ok ]${NC} %s\n" "$*"; }
warn()     { printf "${YELLOW}[warn]${NC} %s\n" "$*"; }
err()      { printf "${RED}[fail]${NC} %s\n" "$*" >&2; }
section()  { printf "\n${DIM}─── %s ───${NC}\n" "$*"; }

# ── flags ────────────────────────────────────────────────────────────────────
CHECK_ONLY=false
NO_INSTALL=false
NO_PULL=false
FORCE_TERMINAL=""

for arg in "$@"; do
    case "$arg" in
        --check)      CHECK_ONLY=true ;;
        --no-install) NO_INSTALL=true ;;
        --no-pull)    NO_PULL=true ;;
        --terminal=*) FORCE_TERMINAL="${arg#*=}" ;;
        --help|-h)
            sed -n '/^# setup.sh/,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            err "Unknown flag: $arg (use --help)"; exit 2 ;;
    esac
done

# ── 1. Python venv ───────────────────────────────────────────────────────────
section "Python environment"
VENV="$PROJECT_ROOT/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
    warn "venv not found at $VENV — creating with system python3..."
    command -v python3 >/dev/null || { err "python3 not found in PATH"; exit 1; }
    python3 -m venv "$VENV"
    ok "Created venv"
fi
ok "venv: $VENV"
ok "python: $($PY --version)"

# ── 2. Install requirements ──────────────────────────────────────────────────
section "Requirements"
if $NO_INSTALL; then
    warn "Skipping pip install (--no-install)"
else
    [[ -f "$PROJECT_ROOT/requirements.txt" ]] || { err "requirements.txt missing"; exit 1; }
    log "Upgrading pip..."
    "$PY" -m pip install --upgrade pip --quiet
    log "Installing requirements (this may take a minute)..."
    "$PY" -m pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
    ok "Requirements installed"
fi

# ── 3. Load .env ─────────────────────────────────────────────────────────────
# Safer than `source`: handles comments, surrounding whitespace, quoted values,
# and silently skips lines that aren't `KEY=value`. Doesn't try to expand shell
# substitutions (the runtime services use python-dotenv which does that).
load_env_safely() {
    local file="$1"
    while IFS= read -r raw || [[ -n "$raw" ]]; do
        # strip inline comments after #, trim whitespace
        local line="${raw%%#*}"
        line="$(echo "$line" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
        [[ -z "$line" ]] && continue
        [[ "$line" != *"="* ]] && continue
        local key="${line%%=*}"
        local val="${line#*=}"
        # trim around = and strip surrounding quotes
        key="$(echo "$key" | sed -E 's/[[:space:]]+$//')"
        val="$(echo "$val" | sed -E 's/^[[:space:]]+//')"
        if [[ "$val" =~ ^\"(.*)\"$ ]] || [[ "$val" =~ ^\'(.*)\'$ ]]; then
            val="${BASH_REMATCH[1]}"
        fi
        # only export valid identifier names
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        export "$key=$val"
    done < "$file"
}

section "Environment"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    load_env_safely "$PROJECT_ROOT/.env"
    ok ".env loaded"
else
    warn ".env not found — falling back to defaults"
fi

# Variables with safe defaults
LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
API_PORT="${API_PORT:-8000}"
MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-8080}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"

# Provider-specific gates
case "$LLM_PROVIDER" in
    ollama)
        ok "LLM_PROVIDER=ollama  (chat=$OLLAMA_MODEL, embed=$OLLAMA_EMBED_MODEL)"
        ;;
    google)
        if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
            err "LLM_PROVIDER=google but GOOGLE_API_KEY is unset"
            exit 1
        fi
        ok "LLM_PROVIDER=google  (model=${GOOGLE_MODEL:-gemini-2.0-flash})"
        ;;
    *)
        err "Unknown LLM_PROVIDER=$LLM_PROVIDER  (expected: ollama | google)"
        exit 1
        ;;
esac

# Soft-warn on missing infra vars (non-fatal — defaults work for local dev).
# Always return 0 — `set -e` would otherwise abort if the var is set
# (the `[[ -z ... ]]` test returns 1 in that case).
warn_missing() {
    local var="$1"; local why="$2"
    if [[ -z "${!var:-}" ]]; then
        warn "$var unset — $why"
    fi
    return 0
}
warn_missing MLFLOW_TRACKING_URI "MLflow retrains + metrics snapshots won't reach the server"
warn_missing DEFAULT_MODEL_ID    "trigger runs will default to main.default.fraud_classifier_v1"
warn_missing CHROMA_PERSIST_DIR  "ChromaDB will use './rag_data' (CWD-relative — beware running from api/)"

# ── 4. Ollama ────────────────────────────────────────────────────────────────
if [[ "$LLM_PROVIDER" == "ollama" || -n "${OLLAMA_EMBED_MODEL:-}" ]]; then
    section "Ollama"
    log "Checking Ollama at $OLLAMA_BASE_URL..."
    if ! curl -sf --max-time 5 "$OLLAMA_BASE_URL/api/tags" >/dev/null; then
        err "Ollama unreachable at $OLLAMA_BASE_URL"
        err "  Start the daemon:  ollama serve   (or systemctl --user start ollama)"
        err "  Install:           https://ollama.com/download"
        exit 1
    fi
    ok "Ollama reachable"

    ensure_model() {
        local model="$1"
        # /api/tags returns {"models":[{"name":"<model>",...},...]} — string match
        if curl -sf "$OLLAMA_BASE_URL/api/tags" | grep -q "\"name\":\"${model}\""; then
            ok "model present: $model"
            return 0
        fi
        if $NO_PULL; then
            warn "model NOT present: $model  (skipping pull — --no-pull)"
            return 0
        fi
        if ! command -v ollama >/dev/null; then
            err "ollama CLI not on PATH but model '$model' missing — install ollama or set --no-pull"
            return 1
        fi
        log "pulling $model..."
        ollama pull "$model"
        ok "pulled: $model"
    }

    ensure_model "$OLLAMA_MODEL"        || exit 1
    ensure_model "$OLLAMA_EMBED_MODEL"  || exit 1
fi

# ── 5. Dataset (Kaggle creditcardfraud) ──────────────────────────────────────
# train.py and the scenario generators need the raw 284K-row creditcard.csv
# from Kaggle. We don't download it automatically — just check and instruct.
section "Dataset"
DATASET_PATH="$PROJECT_ROOT/mlops_agents/data/creditcard.csv"
if [[ -f "$DATASET_PATH" ]]; then
    ok "creditcard.csv present at $DATASET_PATH"
else
    err "creditcard.csv NOT FOUND at $DATASET_PATH"
    err "Download it from Kaggle (requires authenticated kaggle CLI):"
    err ""
    err "  kaggle datasets download mlg-ulb/creditcardfraud -p ./mlops_agents/data/ --unzip"
    err ""
    err "Trainer and scenario generators will fail until this file is present."
    exit 1
fi

# ── 6. Preflight only? ───────────────────────────────────────────────────────
if $CHECK_ONLY; then
    section "Preflight complete"
    ok "All checks passed (--check)"
    exit 0
fi

# ── 6. Detect terminal emulator ──────────────────────────────────────────────
section "Launching services"

detect_terminal() {
    if [[ -n "$FORCE_TERMINAL" ]]; then
        echo "$FORCE_TERMINAL"
        return
    fi
    for t in gnome-terminal konsole kitty alacritty terminator wezterm tilix xterm; do
        if command -v "$t" >/dev/null 2>&1; then
            echo "$t"
            return
        fi
    done
    if command -v tmux >/dev/null 2>&1; then echo "tmux"; return; fi
    echo "background"
}

TERMINAL="$(detect_terminal)"
ok "terminal: $TERMINAL"

# ── 7. Service definitions ───────────────────────────────────────────────────
# Each runs `cd PROJECT_ROOT && source .venv/bin/activate && <cmd>`
SVC_MODEL_CMD="cd '$PROJECT_ROOT' && source .venv/bin/activate && python fraud_model_server/model_server/server.py"
SVC_API_CMD="cd '$PROJECT_ROOT' && source .venv/bin/activate && uvicorn api.main:app --host 0.0.0.0 --port $API_PORT --reload"
SVC_DASH_CMD="cd '$PROJECT_ROOT' && source .venv/bin/activate && streamlit run dashboards/app.py --server.port $DASHBOARD_PORT"

launch_in_terminal() {
    local title="$1"; shift
    local cmd="$1"
    # Inner command keeps the terminal open after the process exits so logs are inspectable.
    local wrapped="$cmd; echo; echo '[$title] exited — press enter to close'; read"
    case "$TERMINAL" in
        gnome-terminal)
            gnome-terminal --title="$title" -- bash -c "$wrapped" >/dev/null 2>&1 &
            ;;
        konsole)
            konsole --new-tab -p "tabtitle=$title" -e bash -c "$wrapped" >/dev/null 2>&1 &
            ;;
        kitty)
            kitty --title "$title" bash -c "$wrapped" >/dev/null 2>&1 &
            ;;
        alacritty)
            alacritty --title "$title" -e bash -c "$wrapped" >/dev/null 2>&1 &
            ;;
        terminator)
            terminator --title="$title" -x bash -c "$wrapped" >/dev/null 2>&1 &
            ;;
        wezterm)
            wezterm start --always-new-process --class "$title" -- bash -c "$wrapped" >/dev/null 2>&1 &
            ;;
        tilix)
            tilix --title "$title" -e bash -c "$wrapped" >/dev/null 2>&1 &
            ;;
        xterm)
            xterm -title "$title" -e bash -c "$wrapped" >/dev/null 2>&1 &
            ;;
        tmux)
            if ! tmux has-session -t mlops 2>/dev/null; then
                tmux new-session -d -s mlops -n "$title" "$cmd"
            else
                tmux new-window -t mlops -n "$title" "$cmd"
            fi
            ;;
        background)
            local logfile="/tmp/mlops-${title}.log"
            warn "no terminal emulator found — $title → $logfile"
            nohup bash -c "$cmd" >"$logfile" 2>&1 &
            ;;
    esac
    sleep 0.4
}

launch_in_terminal "model-server" "$SVC_MODEL_CMD"
launch_in_terminal "api"          "$SVC_API_CMD"
launch_in_terminal "dashboard"    "$SVC_DASH_CMD"

# ── 8. Done ──────────────────────────────────────────────────────────────────
section "Done"
case "$TERMINAL" in
    tmux)
        ok "Launched in tmux session 'mlops' — attach with:  ${CYAN}tmux attach -t mlops${NC}"
        ;;
    background)
        ok "Services backgrounded — logs at /tmp/mlops-*.log"
        ;;
    *)
        ok "Three terminal windows opened ($TERMINAL)"
        ;;
esac

log "Endpoints:"
log "  Model server : http://localhost:$MODEL_SERVER_PORT"
log "  API          : http://localhost:$API_PORT"
log "  Dashboard    : http://localhost:$DASHBOARD_PORT"
log "  Health probe : curl -s http://localhost:$API_PORT/health | python3 -m json.tool"
