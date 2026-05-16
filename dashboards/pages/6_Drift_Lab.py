"""
dashboard/pages/6_Drift_Lab.py

Drift Lab — dataset-centric UI (Option B).

Replaces all inference-time drift injection with genuine dataset activation:
  • Shows available datasets with metadata (rows, fraud_rate, expected_severity)
  • "Create Datasets" one-time setup button
  • "Activate" button per dataset — writes data/active_dataset.json
  • Start / stop transaction generator subprocess on the active dataset
  • Live metrics (PSI-based drift score, accuracy proxy, latency, error rate)
  • Last agent run output (unchanged structure)
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from utils.session import init_session

init_session()

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

st.set_page_config(page_title="Drift Lab · MLOps", page_icon="⬡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0b0e;--surface:#111318;--surface2:#181b22;--border:#1f2330;
--text:#e8eaf0;--text2:#8b91a8;--text3:#555c72;--accent:#00d4ff;--green:#00e5a0;
--amber:#ffb800;--red:#ff4560;--purple:#9b59ff;
--mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
.stButton>button{background:transparent!important;border:1px solid #2a2f3d!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.8rem!important;border-radius:4px!important;transition:all 0.15s!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
.stButton>button[kind="primary"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#000!important;font-weight:600!important;}
[data-baseweb="select"]>div{background:var(--surface2)!important;border-color:var(--border)!important;color:var(--text)!important;}
[data-testid="stRadio"] label{font-family:var(--mono)!important;font-size:0.82rem!important;color:#8b91a8!important;}
[data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;}
[data-baseweb="tab"]{background:transparent!important;color:#555c72!important;font-family:var(--mono)!important;font-size:0.75rem!important;letter-spacing:0.08em!important;text-transform:uppercase!important;}
[aria-selected="true"][data-baseweb="tab"]{color:var(--accent)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

API = st.session_state.get("api_url", "http://localhost:8000")

# ── helpers ───────────────────────────────────────────────────────────────────

SEV_COLOR = {"none": "#00e5a0", "minor": "#00d4ff", "major": "#ffb800", "critical": "#ff4560"}
TAG_COLOR = {"Monitor": "#00d4ff", "Diagnosis": "#9b59ff", "Remediation": "#ffb800", "Reporting": "#00e5a0"}


def section(title: str, sub: str = "") -> None:
    st.markdown(
        f"""<div style="margin:24px 0 14px;padding-bottom:10px;border-bottom:1px solid #1f2330;">
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8eaf0;">{title}</span>
        {"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:#555c72;margin-left:12px;'>"+sub+"</span>" if sub else ""}
        </div>""",
        unsafe_allow_html=True,
    )


def api_get(path: str, **kwargs):
    return requests.get(f"{API}{path}", timeout=6, **kwargs).json()


def api_post(path: str, **kwargs):
    return requests.post(f"{API}{path}", timeout=10, **kwargs).json()


def model_server_call(tool: str, params: dict | None = None):
    model_url = st.session_state.get("model_url", "http://localhost:8080")
    r = requests.post(
        f"{model_url}/mcp/call",
        json={"tool": tool, "params": params or {}},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def fetch_last_run():
    try:
        runs = api_get("/runs")
        done = [r for r in runs if r.get("status") in ("completed", "failed", "rejected")]
        return done[0] if done else None
    except Exception:
        return None


# ── PAGE HEADER ───────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:24px 0 8px;">
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8eaf0;">Drift Lab</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#555c72;margin-top:4px;letter-spacing:0.08em;">
        DATASET CREATION · SCENARIO ACTIVATION · TRANSACTION GENERATOR · LIVE OBSERVATION · AGENT OUTPUT
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET LIBRARY
# ═════════════════════════════════════════════════════════════════════════════
section("Dataset Library", "one-time setup — create all scenario CSVs from creditcard.csv")

# fetch datasets from API
try:
    datasets: list[dict] = api_get("/datasets")
    datasets_ok = True
except Exception:
    datasets = []
    datasets_ok = False

if not datasets_ok or len(datasets) == 0:
    st.markdown("""
    <div style="padding:10px 16px;background:rgba(255,184,0,0.08);border:1px solid rgba(255,184,0,0.3);
                border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ffb800;
                margin-bottom:12px;">
        ○ No datasets found in data/datasets/. Run "Create Datasets" once to generate all scenarios.
    </div>""", unsafe_allow_html=True)
else:
    active_name = next((d["name"] for d in datasets if d.get("active")), None)
    if active_name:
        st.markdown(f"""
        <div style="padding:8px 16px;background:rgba(0,229,160,0.05);border:1px solid rgba(0,229,160,0.2);
                    border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#00e5a0;
                    margin-bottom:12px;">
            ● Active dataset: <strong>{active_name}</strong>
        </div>""", unsafe_allow_html=True)

SEV_BG = {
    "none":     "rgba(0,229,160,0.06)",
    "minor":    "rgba(0,212,255,0.06)",
    "major":    "rgba(255,184,0,0.06)",
    "critical": "rgba(255,70,96,0.06)",
}

for ds in datasets:
    name         = ds.get("name", "—")
    dtype        = ds.get("drift_type", "none")
    description  = ds.get("description", "")
    rows         = ds.get("rows", "—")
    fraud_rate   = ds.get("fraud_rate")
    exp_sev      = ds.get("expected_severity", "none")
    exp_action   = ds.get("expected_action", "—")
    exp_strategy = ds.get("expected_strategy", "—")
    is_active    = ds.get("active", False)

    sev_color = SEV_COLOR.get(exp_sev, "#555c72")
    bg_color  = SEV_BG.get(exp_sev, "rgba(255,255,255,0.02)")
    border    = sev_color if is_active else "#1f2330"

    col_info, col_btn = st.columns([5, 1])
    with col_info:
        fr_str    = f"{fraud_rate:.4f}" if fraud_rate is not None else "—"
        rows_str  = f"{rows:,}" if isinstance(rows, int) else str(rows)
        active_tag = (
            f'<span style="background:rgba(0,229,160,0.2);color:#00e5a0;'
            f'padding:2px 8px;border-radius:3px;font-size:0.65rem;margin-left:8px;">ACTIVE</span>'
            if is_active else ""
        )
        st.markdown(
            f'<div style="padding:12px 16px;background:{bg_color};border:1px solid {border};'
            f'border-radius:6px;margin-bottom:6px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="font-size:0.85rem;font-weight:600;color:#e8eaf0;">{name}{active_tag}</div>'
            f'<div style="display:flex;gap:10px;font-size:0.7rem;">'
            f'<span style="color:#555c72;">rows: <span style="color:#8b91a8;">{rows_str}</span></span>'
            f'<span style="color:#555c72;">fraud_rate: <span style="color:#8b91a8;">{fr_str}</span></span>'
            f'<span style="color:{sev_color};">severity: {exp_sev}</span>'
            f'<span style="color:#555c72;">action: <span style="color:#00d4ff;">{exp_action}</span></span>'
            f'</div></div>'
            f'<div style="font-size:0.73rem;color:#555c72;margin-top:4px;">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if not is_active:
            if st.button("Activate", key=f"activate_{name}", use_container_width=True):
                try:
                    api_post(f"/generator/stop")  # stop generator before switching
                    # write active_dataset.json via the datasets endpoint pattern
                    # (we POST to generator/start with just the dataset name to write the file,
                    #  but don't auto-start — user controls that in section 2)
                    requests.post(
                        f"{API}/generator/start",
                        json={"dataset": name, "rate": 2.0, "seed_n": 500},
                        timeout=10,
                    )
                    # immediately stop again — we just wanted the file written
                    api_post("/generator/stop")
                    st.success(f"Activated {name}")
                    time.sleep(0.3)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.markdown(
                '<div style="padding:6px 0;text-align:center;font-size:0.72rem;color:#00e5a0;">✓ active</div>',
                unsafe_allow_html=True,
            )

# Create datasets button
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
c1, c2, _ = st.columns([2, 2, 4])
with c1:
    if st.button("⚙  Create All Datasets", use_container_width=True):
        try:
            st.write(f"Attempting to connect to: `{API}/datasets/create`")
            r = requests.post(f"{API}/datasets/create", timeout=10)
            r.raise_for_status()
            st.info("Dataset generation started in background — refresh in ~30 s")
        except Exception as e:
            st.error(f"Failed: {e}")
with c2:
    if st.button("↺  Refresh List", use_container_width=True):
        st.rerun()

st.caption(
    "Datasets live in `data/datasets/`. Each CSV has a matching `.json` with scenario metadata. "
    "Needs `data/creditcard.csv` (Kaggle creditcardfraud dataset)."
)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRANSACTION GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
section("Transaction Generator", "replays rows from the active dataset against /predict")

# fetch generator status
try:
    gen_status = api_get("/generator/status")
    gen_alive  = gen_status.get("running", False)
    gen_dataset = gen_status.get("dataset")
    gen_pid     = gen_status.get("pid")
except Exception:
    gen_alive   = False
    gen_dataset = None
    gen_pid     = None

if not gen_alive:
    st.markdown("""
    <div style="padding:10px 16px;background:rgba(255,70,96,0.08);border:1px solid rgba(255,70,96,0.3);
                border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ff4560;
                margin-bottom:12px;">
        ○ Not running — metrics are stale training-time fallbacks. Activate a dataset and start the generator.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="padding:10px 16px;background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.3);
                border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#00e5a0;
                margin-bottom:12px;">
        ● Running — replaying <strong>{gen_dataset}</strong> against /predict
        &nbsp;·&nbsp; <span style="color:#555c72;">pid {gen_pid}</span>
        &nbsp;·&nbsp; <span style="color:#555c72;">hot-swap poll: 30 s</span>
    </div>""", unsafe_allow_html=True)

g1, g2, g3, g4 = st.columns(4)
with g1:
    # default to active dataset if available
    active_name = next((d["name"] for d in datasets if d.get("active")), "baseline")
    dataset_names = [d["name"] for d in datasets] if datasets else ["baseline"]
    default_idx   = dataset_names.index(active_name) if active_name in dataset_names else 0
    sel_dataset   = st.selectbox("Dataset", dataset_names, index=default_idx, key="gen_dataset_sel")
with g2:
    rate = st.selectbox("Rate (req/s)", [1, 2, 5, 10, 20], index=1, key="gen_rate")
with g3:
    err_sel  = st.selectbox("Error inject", ["0% (none)", "5%", "10%", "20%"], index=0, key="gen_err")
    err_frac = float(err_sel.replace("%", "").split()[0]) / 100
with g4:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if not gen_alive:
        if st.button("▶  Start Generator", type="primary", use_container_width=True):
            try:
                r = requests.post(
                    f"{API}/generator/start",
                    json={"dataset": sel_dataset, "rate": rate, "error_rate": err_frac, "seed_n": 500},
                    timeout=10,
                )
                r.raise_for_status()
                st.success(f"Started — {rate} req/s on {sel_dataset}")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
    else:
        if st.button("■  Stop Generator", use_container_width=True):
            try:
                api_post("/generator/stop")
                st.warning("Stopped")
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

st.caption(
    "Generator polls `data/active_dataset.json` every 30 s — activating a new dataset above "
    "will be picked up automatically without a restart."
)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LIVE METRICS
# ═════════════════════════════════════════════════════════════════════════════
section("Live Metrics")

try:
    m           = model_server_call("get_current_metrics")
    sample_size = m.get("sample_size", 0)
    metrics_ok  = True
except Exception as e:
    st.error(f"Model server unreachable: {e}")
    metrics_ok  = False
    m, sample_size = {}, 0

if metrics_ok:
    if sample_size < 10:
        st.markdown(f"""
        <div style="padding:10px 16px;background:rgba(255,184,0,0.08);
                    border:1px solid rgba(255,184,0,0.35);border-radius:6px;
                    font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ffb800;
                    margin-bottom:12px;">
            ⚠ STALE — {sample_size} predictions in history (need ≥ 10).
            Values are training-time fallbacks.
            {"Start the generator above." if not gen_alive else "Wait for predictions to accumulate."}
        </div>""", unsafe_allow_html=True)
    else:
        active_label = gen_dataset or "—"
        st.markdown(f"""
        <div style="padding:8px 16px;background:rgba(0,229,160,0.05);
                    border:1px solid rgba(0,229,160,0.2);border-radius:6px;
                    font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#00e5a0;
                    margin-bottom:12px;">
            ● Live — {sample_size} real predictions &nbsp;·&nbsp; dataset: {active_label}
        </div>""", unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    stale = " *" if sample_size < 10 else ""
    for col, (lbl, val, color, fmt) in zip(
        [mc1, mc2, mc3, mc4],
        [
            ("Drift Score (PSI)" + stale, m.get("drift_score", 0),  "#ff4560", "{:.4f}"),
            ("Accuracy Proxy" + stale,    m.get("accuracy", 0),     "#00d4ff", "{:.4f}"),
            ("Latency p95" + stale,       m.get("latency_ms", 0),   "#00e5a0", "{:.1f}ms"),
            ("Error Rate" + stale,        m.get("error_rate", 0),   "#ffb800", "{:.4f}"),
        ]
    ):
        disp = f"{val:.1f}ms" if "ms" in fmt else fmt.format(val)
        with col:
            st.markdown(
                f'<div style="background:#111318;border:1px solid {color}33;border-radius:8px;'
                f'padding:16px 12px;text-align:center;">'
                f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">{lbl}</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{color};margin-top:6px;'
                f'font-family:\'JetBrains Mono\',monospace;">{disp}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="margin-top:8px;font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#555c72;">'
        f'recall: {m.get("recall",0):.4f} &nbsp;·&nbsp; roc_auc: {m.get("roc_auc",0):.4f}'
        f' &nbsp;·&nbsp; fraud_rate: {m.get("fraud_rate",0):.4f}'
        f' &nbsp;·&nbsp; n={sample_size}</div>',
        unsafe_allow_html=True,
    )

# ── confidence histogram ──────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
section("Prediction Confidence Distribution", "from real /predict calls")

try:
    r2    = model_server_call("get_prediction_history", {"n": 200})
    preds = r2.get("predictions", [])
except Exception:
    preds = []

if len(preds) < 5:
    st.markdown(
        f'<div style="color:#555c72;font-size:0.8rem;padding:16px 0;">'
        f'{"Start the generator above." if not gen_alive else f"{len(preds)} predictions so far — accumulating."}'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    import numpy as np
    df_p  = pd.DataFrame(preds)
    probs = df_p["fraud_prob"].values
    hist, edges = np.histogram(probs, bins=20, range=(0, 1))
    hdf = pd.DataFrame({"bin": [f"{e:.2f}" for e in edges[:-1]], "count": hist}).set_index("bin")

    # get active dataset metadata for hints
    active_ds_meta = next((d for d in datasets if d.get("active")), {})
    drift_type     = active_ds_meta.get("drift_type", "none")

    hc1, hc2 = st.columns([3, 1])
    with hc1:
        chart_color = "#ff4560" if drift_type != "none" else "#00d4ff"
        st.bar_chart(hdf, color=chart_color, height=180)
        hints = {
            "none":         "Healthy — bimodal: confident fraud + confident legit",
            "data_drift":   "Data drift — uncertainty rises, probabilities cluster near 0.5",
            "concept_drift": "Concept drift — may look bimodal but recall is collapsing",
            "mixed":        "Mixed — watch for distribution shift AND recall collapse simultaneously",
        }
        st.caption(hints.get(drift_type, ""))
    with hc2:
        fraud_n  = int(df_p["prediction"].sum())
        avg_conf = float(df_p["fraud_prob"].apply(lambda p: max(p, 1 - p)).mean())
        avg_lat  = float(df_p["latency_ms"].mean())
        st.markdown(
            f'<div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:14px;">'
            f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:10px;">STATS</div>'
            f'<div style="font-size:0.75rem;font-family:\'JetBrains Mono\',monospace;display:flex;flex-direction:column;gap:7px;">'
            f'<div><span style="color:#555c72;">predictions</span><span style="color:#e8eaf0;float:right;">{len(df_p)}</span></div>'
            f'<div><span style="color:#555c72;">fraud flagged</span><span style="color:#ff4560;float:right;">{fraud_n}</span></div>'
            f'<div><span style="color:#555c72;">avg confidence</span><span style="color:#00d4ff;float:right;">{avg_conf:.3f}</span></div>'
            f'<div><span style="color:#555c72;">avg latency</span><span style="color:#00e5a0;float:right;">{avg_lat:.1f}ms</span></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPECTED OUTCOME (from dataset metadata)
# ═════════════════════════════════════════════════════════════════════════════
active_ds_meta = next((d for d in datasets if d.get("active")), {})
if active_ds_meta:
    section("Expected Outcome", "from active dataset metadata — not hardcoded")

    exp_sev      = active_ds_meta.get("expected_severity", "—")
    exp_action   = active_ds_meta.get("expected_action", "—")
    exp_strategy = active_ds_meta.get("expected_strategy", "—")
    exp_drift_ds = active_ds_meta.get("drift_dataset", "—")
    sev_c        = SEV_COLOR.get(exp_sev, "#555c72")

    e1, e2, e3, e4 = st.columns(4)
    for col, (lbl, val, color) in zip([e1, e2, e3, e4], [
        ("Dataset",          active_ds_meta.get("name", "—"), "#00d4ff"),
        ("Expected Severity", exp_sev.upper(),                sev_c),
        ("Expected Action",  exp_action,                      "#00d4ff"),
        ("Retrain Dataset",  exp_drift_ds,                    "#9b59ff"),
    ]):
        with col:
            st.markdown(
                f'<div style="background:#111318;border:1px solid #1f2330;border-radius:6px;padding:12px 14px;">'
                f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">{lbl}</div>'
                f'<div style="font-size:0.85rem;font-weight:600;color:{color};margin-top:4px;'
                f'font-family:\'JetBrains Mono\',monospace;">{val}</div></div>',
                unsafe_allow_html=True,
            )
    if exp_strategy and exp_strategy != "—":
        st.markdown(
            f'<div style="margin-top:8px;font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#555c72;">'
            f'expected_strategy: <span style="color:#ffb800;">{exp_strategy}</span></div>',
            unsafe_allow_html=True,
        )

    # Run Monitor button
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("▶  Run Monitor Agent", use_container_width=False):
        if not gen_alive:
            st.warning("No predictions flowing — monitor may return severity=none. Start the generator first.")
        try:
            r = requests.post(
                f"{API}/runs",
                json={"model_id": "fraud-classifier-v1", "environment": "production"},
                timeout=6,
            )
            r.raise_for_status()
            tid = r.json()["thread_id"]
            st.session_state["active_thread_id"] = tid
            st.info(f"Started — {tid[:16]}… · check Overview")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LAST AGENT RUN OUTPUT
# ═════════════════════════════════════════════════════════════════════════════
section("Last Agent Run Output", "real results — not hardcoded")

run = fetch_last_run()

if not run:
    st.markdown(
        '<div style="color:#555c72;font-size:0.8rem;padding:8px 0;">'
        'No completed runs. Click ▶ Run Monitor Agent above.</div>',
        unsafe_allow_html=True,
    )
else:
    sev     = run.get("severity", "—")
    action  = run.get("recommended_action", "—")
    status  = run.get("status", "—")
    rem_sta = run.get("remediation_status", "—")
    diag    = run.get("diagnosis", "—")
    rem_det = run.get("remediation_detail", "—")
    inc_id  = run.get("incident_id", "—") or "—"

    sev_c  = SEV_COLOR.get(sev, "#555c72")
    stat_c = "#00e5a0" if status == "completed" else "#ff4560"
    rem_c  = "#00e5a0" if rem_sta == "success" else "#ffb800" if rem_sta else "#555c72"

    r1, r2, r3, r4 = st.columns(4)
    for col, (lbl, val, color) in zip([r1, r2, r3, r4], [
        ("Status",      status,               stat_c),
        ("Severity",    (sev or "—").upper(), sev_c),
        ("Action",      action,               "#00d4ff"),
        ("Remediation", rem_sta or "—",       rem_c),
    ]):
        with col:
            st.markdown(
                f'<div style="background:#111318;border:1px solid #1f2330;border-radius:6px;padding:12px 14px;">'
                f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">{lbl}</div>'
                f'<div style="font-size:0.85rem;font-weight:600;color:{color};margin-top:4px;'
                f'font-family:\'JetBrains Mono\',monospace;">{val}</div></div>',
                unsafe_allow_html=True,
            )

    if diag and diag != "—":
        st.markdown(
            f'<div style="margin-top:10px;padding:12px 16px;background:#111318;'
            f'border:1px solid #1f2330;border-radius:6px;">'
            f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">DIAGNOSIS</div>'
            f'<div style="font-size:0.82rem;color:#8b91a8;margin-top:6px;line-height:1.6;">{diag}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    dj         = run.get("diagnosis_json") or {}
    evidence   = dj.get("evidence", [])
    reasoning  = dj.get("reasoning", "")
    confidence = dj.get("confidence")
    root_cat   = dj.get("root_cause_category", "")

    if evidence or reasoning:
        ec1, ec2 = st.columns(2)
        with ec1:
            if evidence:
                ev_html  = "".join(
                    f'<div style="font-size:0.76rem;color:#8b91a8;padding:3px 0;line-height:1.6;">· {e}</div>'
                    for e in evidence
                )
                conf_tag = f"<span style='float:right;color:#9b59ff;'>{confidence}</span>" if confidence else ""
                cat_tag  = f"<span style='color:#00d4ff;margin-left:8px;font-size:0.65rem;'>{root_cat}</span>" if root_cat else ""
                st.markdown(
                    f'<div style="margin-top:10px;padding:12px 16px;background:#111318;'
                    f'border:1px solid #1f2330;border-radius:6px;">'
                    f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:8px;">'
                    f'EVIDENCE {conf_tag}{cat_tag}</div>{ev_html}</div>',
                    unsafe_allow_html=True,
                )
        with ec2:
            if reasoning:
                st.markdown(
                    f'<div style="margin-top:10px;padding:12px 16px;background:#111318;'
                    f'border:1px solid #1f2330;border-radius:6px;">'
                    f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:8px;">REASONING</div>'
                    f'<div style="font-size:0.78rem;color:#8b91a8;line-height:1.7;">{reasoning}</div></div>',
                    unsafe_allow_html=True,
                )

    p = run.get("retrain_prescription")
    if p:
        chips = [
            ("strategy",      p.get("data_strategy", "—"),      "#00d4ff"),
            ("window",        f"{p.get('window_days','—')}d",    "#e8eaf0"),
            ("optimize",      p.get("optimize_for", "—"),        "#9b59ff"),
            ("target_recall", str(p.get("target_recall", "—")), "#ffb800"),
            ("deploy",        p.get("deployment_strategy", "—"), "#00e5a0"),
        ]
        if p.get("drift_dataset"):
            chips.append(("drift_dataset", p["drift_dataset"], "#ff4560"))
        if p.get("drifted_features"):
            chips.append(("drifted", ", ".join(p["drifted_features"]), "#ff4560"))

        chips_html = "".join(
            f'<span style="background:#1a1d26;color:{c};padding:4px 10px;'
            f'border-radius:3px;font-size:0.75rem;">{k}: {v}</span>'
            for k, v, c in chips
        )
        st.markdown(
            f'<div style="margin-top:10px;padding:12px 16px;background:#111318;'
            f'border:1px solid rgba(0,212,255,0.2);border-radius:6px;">'
            f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:8px;">PRESCRIPTION</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{chips_html}</div></div>',
            unsafe_allow_html=True,
        )

    if rem_det and rem_det != "—":
        st.markdown(
            f'<div style="margin-top:10px;padding:12px 16px;background:#111318;'
            f'border:1px solid #1f2330;border-radius:6px;">'
            f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">REMEDIATION DETAIL</div>'
            f'<div style="font-size:0.78rem;color:#8b91a8;margin-top:6px;">{rem_det}</div></div>',
            unsafe_allow_html=True,
        )

    msgs = run.get("messages", []) or []
    if msgs:
        with st.expander("Agent message trace", expanded=False):
            for msg in msgs:
                tag     = msg.split("]")[0].lstrip("[") if "]" in msg else "Agent"
                content = msg.split("]", 1)[1].strip()  if "]" in msg else msg
                tc      = TAG_COLOR.get(tag, "#555c72")
                st.markdown(
                    f'<div style="display:flex;gap:12px;padding:6px 0;border-bottom:1px solid #13161e;'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:0.73rem;">'
                    f'<div style="min-width:100px;color:{tc};font-weight:600;">[{tag}]</div>'
                    f'<div style="color:#8b91a8;">{content}</div></div>',
                    unsafe_allow_html=True,
                )

    similar  = run.get("similar_incidents", []) or []
    runbooks = run.get("relevant_runbooks", []) or []
    notifs   = run.get("notifications_sent", []) or []
    st.markdown(
        f'<div style="margin-top:8px;font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#555c72;">'
        f'RAG: {len(similar)} incidents · {len(runbooks)} runbooks'
        f'{"  ·  top sim: "+str(round(1-similar[0].get("distance",1),3)) if similar else ""}'
        f'{"  ·  notifications: "+str(notifs) if notifs else ""}'
        f'{"  ·  "+inc_id[:28]+"…" if inc_id and inc_id!="—" else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── auto refresh ──────────────────────────────────────────────────────────────
st.divider()
rf1, rf2 = st.columns([1, 5])
with rf1:
    if st.button("↺  Refresh", use_container_width=True):
        st.rerun()
with rf2:
    if gen_alive:
        if st.toggle("Auto-refresh every 5s", value=False, key="auto_rf"):
            time.sleep(5)
            st.rerun()