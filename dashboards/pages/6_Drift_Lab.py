"""
dashboard/pages/6_Drift_Lab.py

Drift Lab — inject data drift, concept drift, or mixed drift.
Observe live model response. Trigger monitor cycles.
"""

import time
import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Drift Lab · MLOps",
                   page_icon="⬡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0b0e;--surface:#111318;--surface2:#181b22;--border:#1f2330;
--text:#e8eaf0;--text2:#8b91a8;--text3:#555c72;--accent:#00d4ff;--green:#00e5a0;--amber:#ffb800;--red:#ff4560;
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
[data-testid="stRadio"] label{font-family:var(--mono)!important;font-size:0.82rem!important;color:var(--text2)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

API = st.session_state.get("api_url",   "http://localhost:8000")
MODEL_SERVER = st.session_state.get("model_url", "http://localhost:8080")

FEATURE_NAMES = [f"V{i}" for i in range(
    1, 29)] + ["Amount_scaled", "Time_scaled"]


def section(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:24px 0 14px 0;padding-bottom:10px;border-bottom:1px solid #1f2330;">
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8eaf0;">{title}</span>
        {"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:#555c72;margin-left:12px;'>"+subtitle+"</span>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def _rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# ── fetch current drift status ────────────────────────────────────────────────
drift_status = {}
try:
    r = requests.post(
        f"{MODEL_SERVER}/mcp/call",
        json={"tool": "get_drift_status", "params": {}},
        timeout=5,
    )
    drift_status = r.json()
except Exception:
    pass

# ── page header + active drift banner ────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8eaf0;">Drift Lab</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#555c72;margin-top:4px;letter-spacing:0.08em;">
        DRIFT INJECTION · LIVE OBSERVATION · PIPELINE TESTING
    </div>
</div>
""", unsafe_allow_html=True)

active = drift_status.get("active", False)
dtype = drift_status.get("drift_type", "none")

if active:
    dtype_label = dtype.replace("_", " ").upper()
    desc = drift_status.get("description", "")
    st.markdown(f"""
    <div style="margin:8px 0 4px;padding:12px 16px;background:rgba(255,184,0,0.08);
                border:1px solid rgba(255,184,0,0.4);border-radius:6px;
                font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ffb800;
                display:flex;align-items:center;justify-content:space-between;">
        <span>⚠ ACTIVE DRIFT: <strong>{dtype_label}</strong>{"  ·  "+desc if desc else ""}</span>
        <span style="color:#555c72;font-size:0.7rem;">
            features: {drift_status.get('features', [])} &nbsp; magnitude: {drift_status.get('magnitude', 0)}
            &nbsp; swap: {drift_status.get('swap_features', [])}
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="margin:8px 0 4px;padding:10px 16px;background:rgba(0,229,160,0.05);
                border:1px solid rgba(0,229,160,0.2);border-radius:6px;
                font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#00e5a0;">
        ✓ No active drift — model operating normally
    </div>
    """, unsafe_allow_html=True)

# ── drift type explainer ──────────────────────────────────────────────────────
section("Drift Type")

ec1, ec2, ec3 = st.columns(3)
for col, (dtype_key, title, desc, color) in zip(
    [ec1, ec2, ec3],
    [
        ("data_drift",    "Data Drift",
         "Input feature distributions shift. Model sees unfamiliar inputs. Accuracy degrades gradually. Uncertainty rises.", "#00d4ff"),
        ("concept_drift", "Concept Drift",
         "Label relationships change. Fraud now looks like legit transactions. Model confidently wrong. Recall collapses.", "#ff4560"),
        ("mixed",         "Mixed",
         "Both simultaneously. Severe degradation across all metrics. Most realistic production failure mode.", "#ffb800"),
    ]
):
    with col:
        st.markdown(f"""
        <div style="background:#111318;border:1px solid {color}33;border-radius:8px;padding:16px;">
            <div style="font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;color:{color};">{title}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8b91a8;
                        margin-top:8px;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── drift configuration ───────────────────────────────────────────────────────
section("Configure Injection")

drift_type_sel = st.radio(
    "Drift type",
    ["data_drift", "concept_drift", "mixed"],
    horizontal=True,
    format_func=lambda x: x.replace("_", " ").title(),
    key="drift_type_radio",
    label_visibility="collapsed",
)

config = {"type": drift_type_sel}

# data drift config
if drift_type_sel in ("data_drift", "mixed"):
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#555c72;
                margin:12px 0 6px;letter-spacing:0.08em;">DATA DRIFT CONFIG</div>
    """, unsafe_allow_html=True)

    dc1, dc2 = st.columns([3, 1])
    with dc1:
        selected_features = st.multiselect(
            "Features to bias",
            FEATURE_NAMES,
            default=["Amount_scaled", "V14", "V17"],
            key="drift_features",
            label_visibility="collapsed",
        )
    with dc2:
        magnitude = st.slider(
            "Magnitude",
            min_value=0.1, max_value=5.0,
            value=2.5, step=0.1,
            key="drift_magnitude",
            label_visibility="collapsed",
        )
        st.markdown(f"""
        <div style="text-align:center;font-size:0.72rem;color:#555c72;margin-top:-8px;">
            magnitude: <span style="color:#00d4ff;">{magnitude}</span>
        </div>
        """, unsafe_allow_html=True)

    config["features"] = selected_features
    config["magnitude"] = magnitude

    # preset scenarios
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#555c72;
                margin:8px 0 4px;">PRESET SCENARIOS</div>
    """, unsafe_allow_html=True)
    ps1, ps2, ps3 = st.columns(3)
    presets = [
        ("High-value merchant shift", ["Amount_scaled"], 3.0),
        ("Geographic shift",          ["V14", "V17", "V12"], 2.0),
        ("Feature pipeline change",   ["V1", "V2", "V3", "V4"], 1.5),
    ]
    for col, (name, feats, mag) in zip([ps1, ps2, ps3], presets):
        with col:
            if st.button(name, key=f"preset_{name}", use_container_width=True):
                st.session_state["drift_features"] = feats
                st.session_state["drift_magnitude"] = mag
                st.rerun()

# concept drift config
if drift_type_sel in ("concept_drift", "mixed"):
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#555c72;
                margin:12px 0 6px;letter-spacing:0.08em;">CONCEPT DRIFT CONFIG</div>
    """, unsafe_allow_html=True)

    cc1, cc2, cc3 = st.columns([2, 0.3, 2])
    with cc1:
        feat_a = st.selectbox(
            "Fraud signal feature (swap FROM)",
            FEATURE_NAMES,
            index=FEATURE_NAMES.index("V14"),
            key="swap_feat_a",
            label_visibility="collapsed",
        )
        st.markdown(f'<div style="font-size:0.7rem;color:#555c72;">FROM: {feat_a} (learned fraud signal)</div>',
                    unsafe_allow_html=True)
    with cc2:
        st.markdown("<div style='text-align:center;font-size:1.2rem;color:#555c72;padding-top:4px;'>↔</div>",
                    unsafe_allow_html=True)
    with cc3:
        feat_b = st.selectbox(
            "Target feature (swap TO)",
            FEATURE_NAMES,
            index=FEATURE_NAMES.index("V4"),
            key="swap_feat_b",
            label_visibility="collapsed",
        )
        st.markdown(f'<div style="font-size:0.7rem;color:#555c72;">TO: {feat_b} (new fraud signal)</div>',
                    unsafe_allow_html=True)

    config["swap_features"] = [feat_a, feat_b]
    st.markdown(f"""
    <div style="margin-top:8px;padding:10px 14px;background:#0d0f14;border:1px solid #1f2330;
                border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8b91a8;line-height:1.6;">
        Effect: model learned that high <strong style="color:#ff4560;">{feat_a}</strong> = fraud.
        After swap, fraud now correlates with <strong style="color:#ff4560;">{feat_b}</strong>
        — a pattern the model has never seen. Recall will collapse.
    </div>
    """, unsafe_allow_html=True)

# description
config["description"] = st.text_input(
    "Label (logged with incident)",
    placeholder="e.g. high-value merchant category shift Q4",
    key="drift_description",
    label_visibility="collapsed",
)

# ── action buttons ────────────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

btn1, btn2, btn3, btn4 = st.columns([1.5, 1.5, 1.5, 3])

with btn1:
    if st.button("⚡  Inject Drift", type="primary", use_container_width=True):
        try:
            r = requests.post(f"{API}/drift/inject", json=config, timeout=8)
            r.raise_for_status()
            resp = r.json()
            st.success(resp.get("message", "Drift injected"))
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Injection failed: {e}")

with btn2:
    if st.button("↺  Reset Baseline", use_container_width=True):
        try:
            r = requests.post(f"{API}/drift/reset", timeout=8)
            r.raise_for_status()
            st.success("Drift cleared — model operating normally")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

with btn3:
    if st.button("▶  Trigger Monitor", use_container_width=True):
        try:
            r = requests.post(
                f"{API}/runs",
                json={"model_id": "fraud-classifier-v1",
                      "environment": "production"},
                timeout=8,
            )
            r.raise_for_status()
            tid = r.json()["thread_id"]
            st.session_state["active_thread_id"] = tid
            st.info(
                f"Monitor cycle started — thread {tid[:16]}… · Check Overview page")
        except Exception as e:
            st.error(f"Failed: {e}")

# ── live observation ──────────────────────────────────────────────────────────
section("Live Model Response", "updates on refresh")

try:
    mr = requests.post(
        f"{MODEL_SERVER}/mcp/call",
        json={"tool": "get_current_metrics", "params": {}},
        timeout=6,
    )
    mr.raise_for_status()
    m = mr.json()

    lc1, lc2, lc3, lc4 = st.columns(4)
    cards = [
        ("Drift Score",    m.get("drift_score", 0),   "#ff4560",   "f{:.4f}"),
        ("Accuracy Proxy", m.get("accuracy", 0),       "#00d4ff",   "f{:.4f}"),
        ("Latency p95",    m.get("latency_ms", 0),
         "#00e5a0",   "f{:.1f}ms"),
        ("Error Rate",     m.get("error_rate", 0),     "#ffb800",   "f{:.4f}"),
    ]
    for col, (label, val, color, fmt) in zip([lc1, lc2, lc3, lc4], cards):
        display = f"{val:.4f}" if "ms" not in fmt else f"{val:.1f}ms"
        with col:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid {color}33;border-radius:8px;
                        padding:16px 12px;text-align:center;">
                <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;text-transform:uppercase;">{label}</div>
                <div style="font-size:1.4rem;font-weight:700;color:{color};margin-top:6px;
                            font-family:'JetBrains Mono',monospace;">{display}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:10px;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#555c72;">
        sample_size: {m.get('sample_size', 0)} predictions &nbsp;·&nbsp;
        fraud_rate: {m.get('fraud_rate', 0):.4f} &nbsp;·&nbsp;
        drift_type: {m.get('drift_type', 'none')}
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.warning(f"Could not fetch live metrics: {e}")

# ── prediction confidence histogram ──────────────────────────────────────────
section("Prediction Confidence Distribution", "last 200 predictions")

try:
    hr = requests.post(
        f"{MODEL_SERVER}/mcp/call",
        json={"tool": "get_prediction_history", "params": {"n": 200}},
        timeout=6,
    )
    hr.raise_for_status()
    preds = hr.json().get("predictions", [])

    if len(preds) < 5:
        st.markdown('<div style="color:#555c72;font-size:0.8rem;">Not enough predictions. Send requests to /predict.</div>',
                    unsafe_allow_html=True)
    else:
        df_p = pd.DataFrame(preds)
        probs = df_p["fraud_prob"].values
        hist, edges = np.histogram(probs, bins=20, range=(0, 1))
        bin_labels = [f"{e:.2f}" for e in edges[:-1]]
        hist_df = pd.DataFrame(
            {"bin": bin_labels, "count": hist}).set_index("bin")

        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            color = "#ff4560" if active else "#00d4ff"
            st.bar_chart(hist_df, color=color, height=180)
            hint = ""
            if dtype == "concept_drift":
                hint = "Concept drift: expect model to be confidently wrong — probabilities cluster near extremes but recall is low"
            elif dtype == "data_drift":
                hint = "Data drift: expect uncertainty — probabilities cluster near 0.5"
            elif dtype == "mixed":
                hint = "Mixed drift: unpredictable distribution — watch for collapse in recall"
            else:
                hint = "Healthy: bimodal distribution (confident fraud AND confident legit predictions)"
            st.markdown(f'<div style="font-size:0.7rem;color:#555c72;margin-top:-8px;">{hint}</div>',
                        unsafe_allow_html=True)

        with hcol2:
            avg_conf = float(df_p["fraud_prob"].apply(
                lambda p: max(p, 1-p)).mean())
            fraud_n = int(df_p["prediction"].sum())
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:14px;">
                <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:10px;">STATS</div>
                <div style="font-size:0.75rem;display:flex;flex-direction:column;gap:7px;">
                    <div><span style="color:#555c72;">predictions</span><span style="color:#e8eaf0;float:right;">{len(df_p)}</span></div>
                    <div><span style="color:#555c72;">fraud flagged</span><span style="color:#ff4560;float:right;">{fraud_n}</span></div>
                    <div><span style="color:#555c72;">avg confidence</span><span style="color:#00d4ff;float:right;">{avg_conf:.3f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.warning(f"Prediction history unavailable: {e}")

# ── what to expect ────────────────────────────────────────────────────────────
section("Expected Agent Behaviour")

expect_map = {
    "data_drift": [
        ("Monitor Agent",     "drift_score rises → major or critical severity"),
        ("Diagnosis Agent",   "root_cause_category=data_drift, recommends retrain"),
        ("Prescription",      "data_strategy=recent_window, drift_period_weight=2.0"),
        ("Remediation Agent", "dispatches retrain.yml with prescription"),
    ],
    "concept_drift": [
        ("Monitor Agent",     "accuracy/recall collapses → critical severity"),
        ("Diagnosis Agent",   "root_cause_category=concept_drift, high confidence"),
        ("Prescription",      "data_strategy=drift_period_only, optimize_for=recall"),
        ("Remediation Agent", "dispatches retrain.yml, deployment_strategy=canary"),
    ],
    "mixed": [
        ("Monitor Agent",     "all metrics degrade → critical"),
        ("Diagnosis Agent",   "may struggle to distinguish — watch confidence score"),
        ("Prescription",      "drift_period_only + high drift_period_weight"),
        ("Remediation Agent", "dispatches retrain with canary rollout"),
    ],
    "none": [
        ("Monitor Agent",     "all metrics within thresholds → severity=none"),
        ("Diagnosis Agent",   "not invoked"),
        ("Remediation Agent", "not invoked"),
        ("Reporting Agent",   "not invoked"),
    ],
}

rows = expect_map.get(drift_type_sel if active else "none", [])
for agent, behaviour in rows:
    st.markdown(f"""
    <div style="display:flex;gap:16px;padding:8px 12px;border-bottom:1px solid #13161e;
                font-family:'JetBrains Mono',monospace;font-size:0.76rem;">
        <div style="min-width:160px;color:#555c72;">{agent}</div>
        <div style="color:#8b91a8;">{behaviour}</div>
    </div>
    """, unsafe_allow_html=True)

# ── auto refresh ──────────────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
rc1, rc2 = st.columns([1, 5])
with rc1:
    if st.button("↺  Refresh", use_container_width=True):
        st.rerun()
with rc2:
    if active:
        auto = st.toggle("Auto-refresh every 5s", value=False,
                         key="drift_auto_refresh")
        if auto:
            time.sleep(5)
            st.rerun()
