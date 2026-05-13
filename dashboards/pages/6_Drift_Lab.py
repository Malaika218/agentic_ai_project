"""
dashboard/pages/6_Drift_Lab.py

Drift Lab — fixes all four problems:
  1. Transaction generator control panel — start/stop subprocess inside the page
  2. Single drift state — only inject_drift, no simulate_drift confusion
  3. Explicit stale metrics warning when sample_size < 10
  4. Actual agent output shown — not hardcoded expected behaviour strings
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

import os

st.set_page_config(page_title="Drift Lab · MLOps",
                   page_icon="⬡", layout="wide")

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
[data-baseweb="multi-select"]{background:var(--surface2)!important;border-color:var(--border)!important;}
[data-baseweb="tag"]{background:rgba(0,212,255,0.15)!important;color:var(--accent)!important;}
.stTextInput input{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-family:var(--mono)!important;}
[data-testid="stRadio"] label{font-family:var(--mono)!important;font-size:0.82rem!important;color:#8b91a8!important;}
[data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;}
[data-baseweb="tab"]{background:transparent!important;color:#555c72!important;font-family:var(--mono)!important;font-size:0.75rem!important;letter-spacing:0.08em!important;text-transform:uppercase!important;}
[aria-selected="true"][data-baseweb="tab"]{color:var(--accent)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

API = st.session_state.get("api_url",   "http://localhost:8000")
MODEL_SERVER = st.session_state.get("model_url", "http://localhost:8080")
FEATURE_NAMES = [f"V{i}" for i in range(
    1, 29)] + ["Amount_scaled", "Time_scaled"]

# ── session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("gen_proc",          None),
    ("gen_running",       False),
    ("drift_features_sel", ["Amount_scaled", "V14", "V17"]),
    ("drift_mag_sel",     2.5),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── helpers ───────────────────────────────────────────────────────────────────


def section(title, sub=""):
    st.markdown(
        f"""<div style="margin:24px 0 14px;padding-bottom:10px;border-bottom:1px solid #1f2330;">
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8eaf0;">{title}</span>
        {"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:#555c72;margin-left:12px;'>"+sub+"</span>" if sub else ""}
        </div>""",
        unsafe_allow_html=True,
    )


def check_gen():
    proc = st.session_state["gen_proc"]
    if proc is None:
        return False
    if proc.poll() is not None:
        st.session_state["gen_running"] = False
        st.session_state["gen_proc"] = None
        return False
    return True


def api_post(tool, params=None):
    r = requests.post(
        f"{MODEL_SERVER}/mcp/call",
        json={"tool": tool, "params": params or {}},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def fetch_last_run():
    try:
        runs = requests.get(f"{API}/runs", timeout=4).json()
        done = [r for r in runs if r.get("status") in (
            "completed", "failed", "rejected")]
        return done[0] if done else None
    except Exception:
        return None


SEV_COLOR = {"none": "#00e5a0", "minor": "#00d4ff",
             "major": "#ffb800", "critical": "#ff4560"}
TAG_COLOR = {"Monitor": "#00d4ff", "Diagnosis": "#9b59ff",
             "Remediation": "#ffb800", "Reporting": "#00e5a0"}

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px;">
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8eaf0;">Drift Lab</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#555c72;margin-top:4px;letter-spacing:0.08em;">
        TRANSACTION GENERATOR · DRIFT INJECTION · LIVE OBSERVATION · AGENT OUTPUT
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRANSACTION GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
section("Transaction Generator",
        "required — drift has no observable effect without predictions flowing")

gen_alive = check_gen()

if not gen_alive:
    st.markdown("""
    <div style="padding:10px 16px;background:rgba(255,70,96,0.08);border:1px solid rgba(255,70,96,0.3);
                border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ff4560;
                margin-bottom:12px;">
        ○ Not running — metrics are stale training-time fallbacks.
        Start the generator before injecting drift.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="padding:10px 16px;background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.3);
                border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#00e5a0;
                margin-bottom:12px;">
        ● Running — Kaggle rows flowing to /predict
    </div>""", unsafe_allow_html=True)

g1, g2, g3, g4 = st.columns(4)
with g1:
    rate = st.selectbox(
        "Rate (req/s)", [1, 2, 5, 10, 20], index=1, key="gen_rate")
with g2:
    err_sel = st.selectbox(
        "Error inject", ["0% (none)", "5%", "10%", "20%"], index=0, key="gen_err")
    err_frac = float(err_sel.replace("%", "").split()[0]) / 100
with g3:
    seed_n = st.selectbox(
        "Pool size", [100, 200, 500, 1000], index=2, key="gen_seed")
with g4:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if not gen_alive:
        if st.button("▶  Start Generator", type="primary", use_container_width=True):
            script_path = Path(__file__).parent.parent.parent / \
                "mlops_agents" / "scripts" / "transaction_generator.py"
            if not script_path.exists():
                st.error(f"Generator script not found: {script_path}")
            else:
                # adjust to repo root
                project_root = Path(__file__).parent.parent.parent
                cmd = [
                    sys.executable, str(script_path),
                    "--rate",       str(rate),
                    "--error-rate", str(err_frac),
                    "--seed-n",     str(seed_n),
                    "--quiet",
                ]
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(script_path.parent),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        env={**os.environ, "PYTHONPATH": str(project_root)},
                    )
                    time.sleep(1)
                    if proc.poll() is not None:
                        err = proc.stderr.read().strip() if proc.stderr else ""
                        st.error(f"Generator exited: {proc.returncode}. {err}")
                    else:
                        st.session_state["gen_proc"] = proc
                        st.session_state["gen_running"] = True
                        st.success(f"Started — {rate} req/s")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")
    else:
        if st.button("■  Stop Generator", use_container_width=True):
            proc = st.session_state["gen_proc"]
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass
            st.session_state["gen_proc"] = None
            st.session_state["gen_running"] = False
            st.warning("Stopped")
            st.rerun()

st.caption(
    "Runs `scripts/transaction_generator.py` as a subprocess. Needs `./data/creditcard.csv`.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DRIFT INJECTION
# ═════════════════════════════════════════════════════════════════════════════
section("Drift Injection")

# current state
try:
    ds = api_post("get_drift_status")
    active = ds.get("active", False)
    dtype = ds.get("drift_type", "none")
except Exception:
    ds, active, dtype = {}, False, "none"

if active:
    st.markdown(f"""
    <div style="margin-bottom:12px;padding:10px 16px;background:rgba(255,184,0,0.08);
                border:1px solid rgba(255,184,0,0.4);border-radius:6px;
                font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ffb800;
                display:flex;justify-content:space-between;">
        <span>⚠ ACTIVE: <strong>{dtype.replace('_', ' ').upper()}</strong>
              {"  ·  "+ds.get('description', '') if ds.get('description') else ""}</span>
        <span style="color:#555c72;font-size:0.7rem;">
            {str(ds.get('features', [])) if ds.get('features') else ""}
            {" swap:"+str(ds.get('swap_features', [])) if ds.get('swap_features') else ""}
            {" mag:"+str(ds.get('magnitude', 0)) if ds.get('magnitude') else ""}
        </span>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="margin-bottom:12px;padding:8px 16px;background:rgba(0,229,160,0.05);
                border:1px solid rgba(0,229,160,0.2);border-radius:6px;
                font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#00e5a0;">
        ✓ No drift active
    </div>""", unsafe_allow_html=True)

drift_type_sel = st.radio(
    "Type", ["data_drift", "concept_drift", "mixed"],
    horizontal=True,
    format_func=lambda x: x.replace("_", " ").title(),
    key="drift_type_radio",
    label_visibility="collapsed",
)
config = {"type": drift_type_sel}

if drift_type_sel in ("data_drift", "mixed"):
    dc1, dc2 = st.columns([3, 1])
    with dc1:
        sel_feats = st.multiselect(
            "Features to bias", FEATURE_NAMES,
            default=st.session_state["drift_features_sel"],
            key="drift_feats_widget",
            label_visibility="collapsed",
        )
        st.session_state["drift_features_sel"] = sel_feats
    with dc2:
        mag = st.slider("Magnitude", 0.1, 5.0,
                        value=float(st.session_state["drift_mag_sel"]),
                        step=0.1, key="drift_mag_widget",
                        label_visibility="collapsed")
        st.session_state["drift_mag_sel"] = mag
        st.caption(f"mag: {mag}")
    config["features"] = sel_feats
    config["magnitude"] = mag

    pc1, pc2, pc3 = st.columns(3)
    for col, (lbl, feats, m) in zip([pc1, pc2, pc3], [
        ("High-value shift",      ["Amount_scaled"],         3.0),
        ("Geographic shift",      ["V14", "V17", "V12"],       2.0),
        ("Schema change",         ["V1", "V2", "V3", "V4"],     1.5),
    ]):
        with col:
            if st.button(lbl, key=f"pre_{lbl}", use_container_width=True):
                st.session_state["drift_features_sel"] = feats
                st.session_state["drift_mag_sel"] = m
                st.rerun()

if drift_type_sel in ("concept_drift", "mixed"):
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns([2, 0.3, 2])
    with cc1:
        fa = st.selectbox("Swap FROM", FEATURE_NAMES,
                          index=FEATURE_NAMES.index("V14"), key="sw_a")
    with cc2:
        st.markdown("<div style='text-align:center;padding-top:28px;color:#555c72;'>↔</div>",
                    unsafe_allow_html=True)
    with cc3:
        fb = st.selectbox("Swap TO", FEATURE_NAMES,
                          index=FEATURE_NAMES.index("V4"), key="sw_b")
    config["swap_features"] = [fa, fb]
    st.markdown(
        f'<div style="margin-top:6px;padding:8px 14px;background:#0d0f14;border:1px solid #1f2330;'
        f'border-radius:4px;font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#8b91a8;">'
        f'Model learned <span style="color:#ff4560;">{fa}</span> high = fraud. '
        f'After swap fraud now correlates with <span style="color:#ff4560;">{fb}</span> — recall collapses.</div>',
        unsafe_allow_html=True,
    )

config["description"] = st.text_input(
    "Label", placeholder="e.g. Q4 merchant shift", key="drift_desc",
    label_visibility="collapsed",
)

ab1, ab2, ab3, _ = st.columns([1.5, 1.5, 1.5, 3])
with ab1:
    if st.button("⚡  Inject", type="primary", use_container_width=True):
        if not gen_alive:
            st.error("Start the transaction generator first.")
        else:
            try:
                r = requests.post(f"{API}/drift/inject",
                                  json=config, timeout=6)
                r.raise_for_status()
                st.success(r.json().get("message", "Injected"))
                time.sleep(0.4)
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
with ab2:
    if st.button("↺  Reset", use_container_width=True):
        try:
            requests.post(f"{API}/drift/reset", timeout=6)
            st.success("Drift cleared")
            time.sleep(0.4)
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")
with ab3:
    if st.button("▶  Run Monitor", use_container_width=True):
        if not gen_alive:
            st.warning(
                "No predictions flowing — monitor may return severity=none regardless.")
        try:
            r = requests.post(
                f"{API}/runs",
                json={"model_id": "fraud-classifier-v1",
                      "environment": "production"},
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
# SECTION 3 — LIVE METRICS with explicit stale warning
# ═════════════════════════════════════════════════════════════════════════════
section("Live Metrics")

try:
    m = api_post("get_current_metrics")
    sample_size = m.get("sample_size", 0)
    metrics_ok = True
except Exception as e:
    st.error(f"Model server unreachable: {e}")
    metrics_ok = False
    m, sample_size = {}, 0

if metrics_ok:
    if sample_size < 10:
        st.markdown(f"""
        <div style="padding:10px 16px;background:rgba(255,184,0,0.08);
                    border:1px solid rgba(255,184,0,0.35);border-radius:6px;
                    font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ffb800;
                    margin-bottom:12px;">
            ⚠ STALE — {sample_size} predictions in history (need ≥ 10).
            Values are training-time fallbacks, not real serving metrics.
            {"Start the generator above." if not gen_alive else "Wait for predictions to accumulate."}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="padding:8px 16px;background:rgba(0,229,160,0.05);
                    border:1px solid rgba(0,229,160,0.2);border-radius:6px;
                    font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#00e5a0;
                    margin-bottom:12px;">
            ● Live — computed from {sample_size} real predictions
        </div>""", unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    stale = " *" if sample_size < 10 else ""
    for col, (lbl, val, color, fmt) in zip(
        [mc1, mc2, mc3, mc4],
        [
            ("Drift Score"+stale,
             m.get("drift_score", 0),  "#ff4560", "{:.4f}"),
            ("Accuracy Proxy"+stale,
             m.get("accuracy", 0),     "#00d4ff", "{:.4f}"),
            ("Latency p95"+stale,    m.get("latency_ms", 0),
             "#00e5a0", "{:.1f}ms"),
            ("Error Rate"+stale,     m.get("error_rate", 0),
             "#ffb800", "{:.4f}"),
        ]
    ):
        disp = fmt.format(val) if "ms" not in fmt else f"{val:.1f}ms"
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
        f'recall: {m.get("recall", 0):.4f} &nbsp;·&nbsp; roc_auc: {m.get("roc_auc", 0):.4f}'
        f' &nbsp;·&nbsp; fraud_rate: {m.get("fraud_rate", 0):.4f}'
        f' &nbsp;·&nbsp; drift_type: <span style="color:#ffb800;">{m.get("drift_type", "none")}</span>'
        f' &nbsp;·&nbsp; n={sample_size}</div>',
        unsafe_allow_html=True,
    )

# ── confidence histogram ──────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
section("Prediction Confidence Distribution", "from real /predict calls")

try:
    r2 = api_post("get_prediction_history", {"n": 200})
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
    df_p = pd.DataFrame(preds)
    probs = df_p["fraud_prob"].values
    hist, edges = np.histogram(probs, bins=20, range=(0, 1))
    hdf = pd.DataFrame(
        {"bin": [f"{e:.2f}" for e in edges[:-1]], "count": hist}).set_index("bin")

    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.bar_chart(hdf, color="#ff4560" if active else "#00d4ff", height=180)
        hints = {
            "none":         "Healthy — bimodal: confident fraud + confident legit",
            "data_drift":   "Data drift — uncertainty rises, probabilities cluster near 0.5",
            "concept_drift": "Concept drift — may look bimodal but recall is collapsing",
            "mixed":        "Mixed — watch for distribution shift AND recall collapse simultaneously",
        }
        st.caption(hints.get(dtype if active else "none", ""))
    with hc2:
        fraud_n = int(df_p["prediction"].sum())
        avg_conf = float(df_p["fraud_prob"].apply(
            lambda p: max(p, 1-p)).mean())
        avg_lat = float(df_p["latency_ms"].mean())
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
# SECTION 4 — ACTUAL AGENT OUTPUT
# ═════════════════════════════════════════════════════════════════════════════
section("Last Agent Run Output", "real results — not hardcoded")

run = fetch_last_run()

if not run:
    st.markdown(
        '<div style="color:#555c72;font-size:0.8rem;padding:8px 0;">'
        'No completed runs. Click ▶ Run Monitor above.</div>',
        unsafe_allow_html=True,
    )
else:
    sev = run.get("severity", "—")
    action = run.get("recommended_action", "—")
    status = run.get("status", "—")
    rem_sta = run.get("remediation_status", "—")
    diag = run.get("diagnosis", "—")
    rem_det = run.get("remediation_detail", "—")
    inc_id = run.get("incident_id", "—") or "—"

    sev_c = SEV_COLOR.get(sev, "#555c72")
    stat_c = "#00e5a0" if status == "completed" else "#ff4560"
    rem_c = "#00e5a0" if rem_sta == "success" else "#ffb800" if rem_sta else "#555c72"

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

    # diagnosis
    if diag and diag != "—":
        st.markdown(
            f'<div style="margin-top:10px;padding:12px 16px;background:#111318;'
            f'border:1px solid #1f2330;border-radius:6px;">'
            f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">DIAGNOSIS</div>'
            f'<div style="font-size:0.82rem;color:#8b91a8;margin-top:6px;line-height:1.6;">{diag}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # evidence + reasoning
    dj = run.get("diagnosis_json") or {}
    evidence = dj.get("evidence", [])
    reasoning = dj.get("reasoning", "")
    confidence = dj.get("confidence")
    root_cat = dj.get("root_cause_category", "")

    if evidence or reasoning:
        ec1, ec2 = st.columns(2)
        with ec1:
            if evidence:
                ev_html = "".join(
                    f'<div style="font-size:0.76rem;color:#8b91a8;padding:3px 0;line-height:1.6;">· {e}</div>'
                    for e in evidence
                )
                conf_tag = f"<span style='float:right;color:#9b59ff;'>{confidence}</span>" if confidence else ""
                cat_tag = f"<span style='color:#00d4ff;margin-left:8px;font-size:0.65rem;'>{root_cat}</span>" if root_cat else ""
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

    # prescription chips
    p = run.get("retrain_prescription")
    if p:
        chips = [
            ("strategy",       p.get("data_strategy", "—"),          "#00d4ff"),
            (f"window",
             f"{p.get('window_days', '—')}d",       "#e8eaf0"),
            ("optimize",       p.get("optimize_for", "—"),            "#9b59ff"),
            ("target_recall",  str(p.get("target_recall", "—")),      "#ffb800"),
            ("deploy",         p.get("deployment_strategy", "—"),     "#00e5a0"),
        ]
        if p.get("drifted_features"):
            chips.append(("drifted", ", ".join(
                p["drifted_features"]), "#ff4560"))

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

    # remediation detail
    if rem_det and rem_det != "—":
        st.markdown(
            f'<div style="margin-top:10px;padding:12px 16px;background:#111318;'
            f'border:1px solid #1f2330;border-radius:6px;">'
            f'<div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">REMEDIATION DETAIL</div>'
            f'<div style="font-size:0.78rem;color:#8b91a8;margin-top:6px;">{rem_det}</div></div>',
            unsafe_allow_html=True,
        )

    # agent trace
    msgs = run.get("messages", []) or []
    if msgs:
        with st.expander("Agent message trace", expanded=False):
            for msg in msgs:
                tag = msg.split("]")[0].lstrip("[") if "]" in msg else "Agent"
                content = msg.split("]", 1)[1].strip() if "]" in msg else msg
                tc = TAG_COLOR.get(tag, "#555c72")
                st.markdown(
                    f'<div style="display:flex;gap:12px;padding:6px 0;border-bottom:1px solid #13161e;'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:0.73rem;">'
                    f'<div style="min-width:100px;color:{tc};font-weight:600;">[{tag}]</div>'
                    f'<div style="color:#8b91a8;">{content}</div></div>',
                    unsafe_allow_html=True,
                )

    # footer meta
    similar = run.get("similar_incidents", []) or []
    runbooks = run.get("relevant_runbooks", []) or []
    notifs = run.get("notifications_sent", []) or []
    st.markdown(
        f'<div style="margin-top:8px;font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#555c72;">'
        f'RAG: {len(similar)} incidents · {len(runbooks)} runbooks'
        f'{"  ·  top sim: "+str(round(1-similar[0].get("distance", 1), 3)) if similar else ""}'
        f'{"  ·  notifications: "+str(notifs) if notifs else ""}'
        f'{"  ·  "+inc_id[:28]+"…" if inc_id and inc_id != "—" else ""}'
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
    if gen_alive or active:
        if st.toggle("Auto-refresh every 5s", value=False, key="auto_rf"):
            time.sleep(5)
            st.rerun()
