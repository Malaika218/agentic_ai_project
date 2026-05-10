"""
dashboard/pages/1_Overview.py

Overview page — model health, trigger run, live pipeline status.
"""

import time
import streamlit as st
import requests

st.set_page_config(page_title="Overview · MLOps", page_icon="⬡", layout="wide")

# re-apply global styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0b0e;--surface:#111318;--surface2:#181b22;--border:#1f2330;--border2:#2a2f3d;
--text:#e8eaf0;--text2:#8b91a8;--text3:#555c72;--accent:#00d4ff;--accent2:#0099cc;
--green:#00e5a0;--amber:#ffb800;--red:#ff4560;--purple:#9b59ff;
--mono:'JetBrains Mono',monospace;--display:'Syne',sans-serif;}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="metric-container"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:16px!important;}
[data-testid="metric-container"] label{color:var(--text2)!important;font-size:0.7rem!important;letter-spacing:0.12em!important;text-transform:uppercase!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--accent)!important;font-size:1.6rem!important;font-weight:600!important;}
.stButton>button{background:transparent!important;border:1px solid var(--border2)!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.8rem!important;border-radius:4px!important;transition:all 0.15s!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
.stButton>button[kind="primary"]{background:var(--accent)!important;border-color:var(--accent)!important;color:#000!important;font-weight:600!important;}
.stSelectbox [data-baseweb="select"]>div{background:var(--surface2)!important;border-color:var(--border)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

API = st.session_state.get("api_url", "http://localhost:8000")
MODEL_SERVER = st.session_state.get("model_url", "http://localhost:8080")

# ── helpers ───────────────────────────────────────────────────────────────────

SEV_COLOR = {
    "none":     "#00e5a0",
    "minor":    "#00d4ff",
    "major":    "#ffb800",
    "critical": "#ff4560",
    None:       "#555c72",
}

STATUS_COLOR = {
    "completed":        "#00e5a0",
    "running":          "#00d4ff",
    "queued":           "#8b91a8",
    "awaiting_approval":"#ffb800",
    "failed":           "#ff4560",
    "rejected":         "#555c72",
}

def sev_badge(sev):
    c = SEV_COLOR.get(sev, "#555c72")
    label = (sev or "—").upper()
    return f'<span style="background:rgba({_hex_to_rgb(c)},0.15);color:{c};font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;font-weight:600;letter-spacing:0.1em;padding:3px 10px;border-radius:3px;border:1px solid rgba({_hex_to_rgb(c)},0.3);">{label}</span>'

def status_badge(status):
    c = STATUS_COLOR.get(status, "#555c72")
    return f'<span style="color:{c};font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;font-weight:500;">● {(status or "—").upper()}</span>'

def _hex_to_rgb(h):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"{r},{g},{b}"

def section(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:28px 0 16px 0;padding-bottom:10px;border-bottom:1px solid #1f2330;">
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                     color:#e8eaf0;letter-spacing:0.02em;">{title}</span>
        {"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:#555c72;margin-left:12px;'>"+subtitle+"</span>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

# ── page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                color:#e8eaf0;letter-spacing:-0.01em;">Overview</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                color:#555c72;margin-top:4px;letter-spacing:0.08em;">
        PIPELINE CONTROL · MODEL HEALTH · LIVE STATUS
    </div>
</div>
""", unsafe_allow_html=True)

# ── health strip ──────────────────────────────────────────────────────────────
try:
    h = requests.get(f"{API}/health", timeout=4).json()
    services = h.get("services", {})
    overall  = h.get("status") == "ok"
except Exception:
    services = {}
    overall  = False

health_cols = st.columns(len(services) + 1 if services else 5)
with health_cols[0]:
    color = "#00e5a0" if overall else "#ff4560"
    st.markdown(f"""
    <div style="background:#111318;border:1px solid {'#00e5a0' if overall else '#ff4560'};
                border-radius:6px;padding:12px 16px;text-align:center;">
        <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;">SYSTEM</div>
        <div style="font-size:0.9rem;font-weight:600;color:{color};margin-top:4px;">
            {'OPERATIONAL' if overall else 'DEGRADED'}
        </div>
    </div>
    """, unsafe_allow_html=True)

for i, (svc, ok) in enumerate(services.items()):
    with health_cols[i + 1]:
        c = "#00e5a0" if ok else "#ff4560"
        st.markdown(f"""
        <div style="background:#111318;border:1px solid #1f2330;border-radius:6px;
                    padding:12px 16px;text-align:center;">
            <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;">{svc.upper()}</div>
            <div style="font-size:0.85rem;font-weight:500;color:{c};margin-top:4px;">
                {'● OK' if ok else '○ DOWN'}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── live model metrics ────────────────────────────────────────────────────────
section("Model Metrics", "live from serving layer")

try:
    m = requests.post(
        f"{MODEL_SERVER}/mcp/call",
        json={"tool": "get_current_metrics", "params": {}},
        timeout=6,
    ).json()
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Accuracy",    f"{m.get('accuracy',0):.3f}",    delta=None)
    mc2.metric("Drift Score", f"{m.get('drift_score',0):.3f}", delta=None)
    mc3.metric("Latency p95", f"{m.get('latency_ms',0):.0f}ms")
    mc4.metric("Error Rate",  f"{m.get('error_rate',0):.4f}")
    mc5.metric("Recall",      f"{m.get('recall',0):.3f}")
    mc6.metric("ROC-AUC",     f"{m.get('roc_auc',0):.3f}")

    drift_active = m.get("drift_active", False)
    if drift_active:
        st.markdown(f"""
        <div style="margin-top:10px;padding:10px 16px;background:rgba(255,184,0,0.08);
                    border:1px solid rgba(255,184,0,0.3);border-radius:6px;
                    font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ffb800;">
            ⚠ Active drift injection: <strong>{m.get('drift_type','').replace('_',' ').upper()}</strong>
            — go to Drift Lab to reset
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Could not reach model server: {e}")

# ── trigger run ───────────────────────────────────────────────────────────────
section("Trigger Monitor Cycle")

with st.container():
    st.markdown("""
    <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:20px 24px;">
    """, unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns([3, 2, 1])
    with tc1:
        model_id = st.text_input(
            "Model ID",
            value="fraud-classifier-v1",
            key="trigger_model_id",
            label_visibility="visible",
        )
    with tc2:
        environment = st.selectbox(
            "Environment",
            ["production", "staging", "canary"],
            key="trigger_env",
        )
    with tc3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        trigger = st.button("▶  Run", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

if trigger:
    try:
        r = requests.post(
            f"{API}/runs",
            json={"model_id": model_id, "environment": environment},
            timeout=10,
        )
        r.raise_for_status()
        tid = r.json()["thread_id"]
        st.session_state["active_thread_id"] = tid
        st.success(f"Pipeline started — thread {tid[:16]}…")
    except Exception as e:
        st.error(f"Failed to start pipeline: {e}")

# ── live run status ───────────────────────────────────────────────────────────
section("Active Run")

tid = st.session_state.get("active_thread_id")

if not tid:
    st.markdown("""
    <div style="color:#555c72;font-size:0.8rem;padding:16px 0;">
        No active run. Trigger a cycle above or select a run from the run history below.
    </div>
    """, unsafe_allow_html=True)
else:
    try:
        run = requests.get(f"{API}/runs/{tid}", timeout=5).json()
        status  = run.get("status", "unknown")
        sev     = run.get("severity")
        agent   = run.get("current_agent", "—")
        inc_id  = run.get("incident_id", "—")

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1f2330;border-radius:6px;padding:14px 16px;">
                <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;">STATUS</div>
                <div style="margin-top:6px;">{status_badge(status)}</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1f2330;border-radius:6px;padding:14px 16px;">
                <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;">SEVERITY</div>
                <div style="margin-top:6px;">{sev_badge(sev)}</div>
            </div>""", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1f2330;border-radius:6px;padding:14px 16px;">
                <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;">CURRENT AGENT</div>
                <div style="font-size:0.85rem;font-weight:500;color:#e8eaf0;margin-top:6px;">{agent}</div>
            </div>""", unsafe_allow_html=True)
        with rc4:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1f2330;border-radius:6px;padding:14px 16px;">
                <div style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;">INCIDENT ID</div>
                <div style="font-size:0.75rem;color:#8b91a8;margin-top:6px;word-break:break-all;">
                    {inc_id[:24]+"…" if inc_id and inc_id != "—" and len(inc_id) > 24 else inc_id}
                </div>
            </div>""", unsafe_allow_html=True)

        # agent message trace
        if run.get("messages"):
            section("Agent Trace")
            for msg in run["messages"]:
                # messages are prefixed [Monitor], [Diagnosis], [Remediation], [Reporting]
                agent_tag = msg.split("]")[0].lstrip("[") if "]" in msg else "Agent"
                content   = msg.split("]", 1)[1].strip() if "]" in msg else msg
                tag_color = {
                    "Monitor":     "#00d4ff",
                    "Diagnosis":   "#9b59ff",
                    "Remediation": "#ffb800",
                    "Reporting":   "#00e5a0",
                }.get(agent_tag, "#555c72")

                st.markdown(f"""
                <div style="display:flex;gap:12px;padding:8px 12px;border-bottom:1px solid #13161e;
                            font-family:'JetBrains Mono',monospace;font-size:0.76rem;">
                    <div style="min-width:100px;color:{tag_color};font-weight:600;">[{agent_tag}]</div>
                    <div style="color:#8b91a8;line-height:1.6;">{content}</div>
                </div>
                """, unsafe_allow_html=True)

        # remediation detail
        if run.get("remediation_detail"):
            st.markdown(f"""
            <div style="margin-top:12px;padding:12px 16px;background:#111318;
                        border:1px solid #1f2330;border-radius:6px;">
                <span style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;">REMEDIATION DETAIL</span>
                <div style="font-size:0.82rem;color:#8b91a8;margin-top:6px;line-height:1.6;">
                    {run['remediation_detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # retrain prescription
        if run.get("retrain_prescription"):
            p = run["retrain_prescription"]
            section("Retrain Prescription")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.markdown(f"""
                <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:16px;">
                    <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:10px;">DATA</div>
                    <table style="width:100%;font-size:0.76rem;border-collapse:collapse;">
                        <tr><td style="color:#555c72;padding:4px 0;">strategy</td>
                            <td style="color:#00d4ff;text-align:right;">{p.get('data_strategy','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">window_days</td>
                            <td style="color:#e8eaf0;text-align:right;">{p.get('window_days','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">drift_weight</td>
                            <td style="color:#e8eaf0;text-align:right;">{p.get('drift_period_weight','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">exclude_before</td>
                            <td style="color:#e8eaf0;text-align:right;">{p.get('exclude_before','—') or 'none'}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            with pc2:
                st.markdown(f"""
                <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:16px;">
                    <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:10px;">MODEL</div>
                    <table style="width:100%;font-size:0.76rem;border-collapse:collapse;">
                        <tr><td style="color:#555c72;padding:4px 0;">optimize_for</td>
                            <td style="color:#9b59ff;text-align:right;">{p.get('optimize_for','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">target_recall</td>
                            <td style="color:#e8eaf0;text-align:right;">{p.get('target_recall','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">target_roc_auc</td>
                            <td style="color:#e8eaf0;text-align:right;">{p.get('target_roc_auc','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">drifted_features</td>
                            <td style="color:#ff4560;text-align:right;font-size:0.68rem;">
                                {', '.join(p.get('drifted_features',[]) or ['—'])}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            with pc3:
                st.markdown(f"""
                <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;padding:16px;">
                    <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:10px;">DEPLOYMENT</div>
                    <table style="width:100%;font-size:0.76rem;border-collapse:collapse;">
                        <tr><td style="color:#555c72;padding:4px 0;">strategy</td>
                            <td style="color:#00e5a0;text-align:right;">{p.get('deployment_strategy','—')}</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">canary_pct</td>
                            <td style="color:#e8eaf0;text-align:right;">{p.get('canary_traffic_pct','—')}%</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">shadow_hours</td>
                            <td style="color:#e8eaf0;text-align:right;">{p.get('shadow_period_hours','—')}h</td></tr>
                        <tr><td style="color:#555c72;padding:4px 0;">refit_preproc</td>
                            <td style="color:#e8eaf0;text-align:right;">{p.get('refit_preprocessors','—')}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

        if run.get("diagnosis"):
            st.markdown(f"""
            <div style="margin-top:12px;padding:12px 16px;background:#111318;
                        border:1px solid #1f2330;border-radius:6px;">
                <span style="font-size:0.65rem;color:#555c72;letter-spacing:0.12em;">DIAGNOSIS</span>
                <div style="font-size:0.82rem;color:#8b91a8;margin-top:6px;line-height:1.6;">
                    {run['diagnosis']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if status == "awaiting_approval":
            st.warning("⏸  Pipeline paused — human approval required. Go to **Approvals** page.")

        if status == "running":
            time.sleep(3)
            st.rerun()

    except Exception as e:
        st.error(f"Could not fetch run status: {e}")

# ── run history ───────────────────────────────────────────────────────────────
section("Run History", "last 20 pipeline cycles")

try:
    all_runs = requests.get(f"{API}/runs", timeout=5).json()

    if not all_runs:
        st.markdown('<div style="color:#555c72;font-size:0.8rem;padding:8px 0;">No runs yet.</div>',
                    unsafe_allow_html=True)
    else:
        # header
        st.markdown("""
        <div style="display:grid;grid-template-columns:160px 120px 80px 100px 100px 1fr;
                    gap:8px;padding:8px 12px;
                    font-size:0.65rem;color:#555c72;letter-spacing:0.12em;
                    border-bottom:1px solid #1f2330;margin-bottom:4px;">
            <div>THREAD</div><div>STARTED</div><div>ENV</div>
            <div>STATUS</div><div>SEVERITY</div><div>ACTION</div>
        </div>
        """, unsafe_allow_html=True)

        for run in all_runs[:20]:
            tid_short  = run["thread_id"][:16] + "…"
            started    = (run.get("started_at") or run.get("created_at", ""))[:16].replace("T", " ")
            env        = run.get("environment", "—")
            status     = run.get("status", "—")
            sev        = run.get("severity") or "—"
            action     = run.get("recommended_action") or "—"
            sev_c      = SEV_COLOR.get(sev, "#555c72")
            stat_c     = STATUS_COLOR.get(status, "#555c72")

            is_active = run["thread_id"] == st.session_state.get("active_thread_id")
            bg = "rgba(0,212,255,0.04)" if is_active else "transparent"

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:160px 120px 80px 100px 100px 1fr;
                        gap:8px;padding:9px 12px;
                        border-bottom:1px solid #1a1d26;background:{bg};
                        font-size:0.78rem;">
                <div style="color:#8b91a8;font-size:0.73rem;">{tid_short}</div>
                <div style="color:#555c72;">{started}</div>
                <div style="color:#555c72;">{env}</div>
                <div style="color:{stat_c};">● {status}</div>
                <div style="color:{sev_c};">{sev.upper()}</div>
                <div style="color:#8b91a8;">{action}</div>
            </div>
            """, unsafe_allow_html=True)

        # click to set active
        run_ids = [r["thread_id"] for r in all_runs[:20]]
        selected = st.selectbox(
            "Select run to track",
            options=["—"] + run_ids,
            format_func=lambda x: x[:24] + "…" if x != "—" and len(x) > 24 else x,
            key="run_selector",
        )
        if selected != "—":
            st.session_state["active_thread_id"] = selected
            st.rerun()

except Exception as e:
    st.error(f"Could not fetch run history: {e}")
