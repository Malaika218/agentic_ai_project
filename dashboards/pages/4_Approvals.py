"""
dashboard/pages/4_Approvals.py

Human-in-the-loop approvals — list paused major-severity runs, approve or reject.
Auto-refreshes every 5s when there are pending approvals.
"""

import time
import streamlit as st
import requests

st.set_page_config(page_title="Approvals · MLOps", page_icon="⬡", layout="wide")

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
.stButton>button[kind="primary"]{background:var(--green)!important;border-color:var(--green)!important;color:#000!important;font-weight:600!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

API = st.session_state.get("api_url", "http://localhost:8000")

def section(title, subtitle=""):
    st.markdown(f"""
    <div style="margin:24px 0 14px 0;padding-bottom:10px;border-bottom:1px solid #1f2330;">
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#e8eaf0;">{title}</span>
        {"<span style='font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;color:#555c72;margin-left:12px;'>"+subtitle+"</span>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

# ── page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8eaf0;">Approvals</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#555c72;margin-top:4px;letter-spacing:0.08em;">
        HUMAN-IN-THE-LOOP · MAJOR SEVERITY REMEDIATION GATE
    </div>
</div>
""", unsafe_allow_html=True)

# ── fetch all runs ────────────────────────────────────────────────────────────
try:
    all_runs = requests.get(f"{API}/runs", timeout=5).json()
except Exception as e:
    st.error(f"Could not reach API: {e}")
    st.stop()

pending  = [r for r in all_runs if r.get("status") == "awaiting_approval"]
resolved = [r for r in all_runs if r.get("status") in ("completed","rejected") and r.get("human_approved") is not None]

# ── pending approvals ─────────────────────────────────────────────────────────
section("Pending Approval", f"{len(pending)} waiting")

if not pending:
    st.markdown("""
    <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;
                padding:32px;text-align:center;color:#555c72;font-size:0.82rem;">
        <div style="font-size:1.5rem;margin-bottom:8px;">✓</div>
        No runs awaiting approval.<br>
        <span style="font-size:0.72rem;">Major-severity incidents will appear here automatically.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    # pulse indicator
    st.markdown(f"""
    <div style="margin-bottom:16px;padding:10px 16px;background:rgba(255,184,0,0.08);
                border:1px solid rgba(255,184,0,0.4);border-radius:6px;
                font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#ffb800;">
        ⏸ {len(pending)} run{"s" if len(pending)>1 else ""} paused awaiting your decision
    </div>
    """, unsafe_allow_html=True)

    for run in pending:
        tid        = run["thread_id"]
        model_id   = run.get("model_id","—")
        env        = run.get("environment","—")
        diagnosis  = run.get("diagnosis","No diagnosis available")
        action     = run.get("recommended_action","—")
        sev        = run.get("severity","major")
        started    = (run.get("started_at","")[:16]).replace("T"," ")

        st.markdown(f"""
        <div style="background:#111318;border:1px solid rgba(255,184,0,0.3);
                    border-radius:8px;padding:24px;margin-bottom:16px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
                <div>
                    <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#ffb800;">
                        ⏸ AWAITING APPROVAL
                    </span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                 color:#555c72;margin-left:12px;">{tid[:24]}…</span>
                </div>
                <div style="font-size:0.72rem;color:#555c72;">{started}</div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;">
                <div style="background:#0d0f14;border:1px solid #1f2330;border-radius:6px;padding:12px;">
                    <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">MODEL</div>
                    <div style="font-size:0.85rem;color:#e8eaf0;margin-top:4px;">{model_id}</div>
                </div>
                <div style="background:#0d0f14;border:1px solid #1f2330;border-radius:6px;padding:12px;">
                    <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">ENVIRONMENT</div>
                    <div style="font-size:0.85rem;color:#e8eaf0;margin-top:4px;">{env}</div>
                </div>
                <div style="background:#0d0f14;border:1px solid #1f2330;border-radius:6px;padding:12px;">
                    <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;">PROPOSED ACTION</div>
                    <div style="font-size:0.85rem;color:#00d4ff;margin-top:4px;font-weight:600;">{action}</div>
                </div>
            </div>

            <div style="background:#0d0f14;border:1px solid #1f2330;border-radius:6px;
                        padding:14px 16px;margin-bottom:16px;">
                <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:8px;">ROOT CAUSE DIAGNOSIS</div>
                <div style="font-size:0.82rem;color:#8b91a8;line-height:1.7;">{diagnosis}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # evidence list
        diag_json = run.get("diagnosis_json", {}) or {}
        evidence  = diag_json.get("evidence", [])
        reasoning = diag_json.get("reasoning", "")
        confidence = diag_json.get("confidence")
        root_cat   = diag_json.get("root_cause_category", "")

        if evidence:
            st.markdown(f"""
            <div style="background:#0d0f14;border:1px solid #1f2330;border-radius:6px;
                        padding:14px 16px;margin-bottom:12px;">
                <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:8px;">
                    EVIDENCE
                    {"<span style='float:right;color:#9b59ff;'>confidence: "+str(confidence)+"</span>" if confidence else ""}
                    {"<span style='margin-left:8px;color:#00d4ff;font-size:0.65rem;'>"+root_cat+"</span>" if root_cat else ""}
                </div>
                {"".join(f'<div style="font-size:0.78rem;color:#8b91a8;padding:3px 0;line-height:1.6;">· {e}</div>' for e in evidence)}
            </div>
            """, unsafe_allow_html=True)

        if reasoning:
            st.markdown(f"""
            <div style="background:#0d0f14;border:1px solid #1f2330;border-radius:6px;
                        padding:14px 16px;margin-bottom:12px;">
                <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:8px;">REASONING CHAIN</div>
                <div style="font-size:0.82rem;color:#8b91a8;line-height:1.7;">{reasoning}</div>
            </div>
            """, unsafe_allow_html=True)

        # prescription summary
        prescription = run.get("retrain_prescription")
        if prescription:
            st.markdown(f"""
            <div style="background:#0d0f14;border:1px solid rgba(0,212,255,0.2);border-radius:6px;
                        padding:14px 16px;margin-bottom:12px;">
                <div style="font-size:0.62rem;color:#555c72;letter-spacing:0.12em;margin-bottom:8px;">PROPOSED PRESCRIPTION</div>
                <div style="display:flex;flex-wrap:wrap;gap:8px;font-size:0.75rem;">
                    <span style="background:#1a1d26;color:#00d4ff;padding:4px 10px;border-radius:3px;">
                        strategy: {prescription.get('data_strategy','—')}
                    </span>
                    <span style="background:#1a1d26;color:#e8eaf0;padding:4px 10px;border-radius:3px;">
                        window: {prescription.get('window_days','—')}d
                    </span>
                    <span style="background:#1a1d26;color:#9b59ff;padding:4px 10px;border-radius:3px;">
                        optimize: {prescription.get('optimize_for','—')}
                    </span>
                    <span style="background:#1a1d26;color:#00e5a0;padding:4px 10px;border-radius:3px;">
                        deploy: {prescription.get('deployment_strategy','—')}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # RAG context used
        similar = run.get("similar_incidents", []) or []
        runbooks = run.get("relevant_runbooks", []) or []
        if similar or runbooks:
            st.markdown(f"""
            <div style="font-size:0.72rem;color:#555c72;padding:8px 0;">
                RAG context used: {len(similar)} similar incidents · {len(runbooks)} runbooks
                {"· top match: "+str(round(1-similar[0].get('distance',1),3))+" similarity" if similar else ""}
            </div>
            """, unsafe_allow_html=True)

        # agent trace
        messages = run.get("messages", []) or []
        if messages:
            with st.expander("Agent message trace", expanded=False):
                for msg in messages:
                    agent_tag = msg.split("]")[0].lstrip("[") if "]" in msg else "Agent"
                    content   = msg.split("]", 1)[1].strip() if "]" in msg else msg
                    tag_color = {
                        "Monitor": "#00d4ff", "Diagnosis": "#9b59ff",
                        "Remediation": "#ffb800", "Reporting": "#00e5a0",
                    }.get(agent_tag, "#555c72")
                    st.markdown(f"""
                    <div style="display:flex;gap:12px;padding:6px 0;border-bottom:1px solid #13161e;
                                font-family:'JetBrains Mono',monospace;font-size:0.73rem;">
                        <div style="min-width:90px;color:{tag_color};">[{agent_tag}]</div>
                        <div style="color:#8b91a8;">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # approve / reject buttons
        btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 4])
        with btn_c1:
            if st.button(
                "✓  Approve",
                key=f"approve_{tid}",
                type="primary",
                use_container_width=True,
            ):
                try:
                    r = requests.post(f"{API}/runs/{tid}/approve", timeout=8)
                    r.raise_for_status()
                    st.success(f"Approved — remediation resuming for thread {tid[:16]}…")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Approval failed: {e}")

        with btn_c2:
            if st.button(
                "✗  Reject",
                key=f"reject_{tid}",
                use_container_width=True,
            ):
                try:
                    r = requests.post(f"{API}/runs/{tid}/reject", timeout=8)
                    r.raise_for_status()
                    st.warning(f"Rejected — pipeline ended for thread {tid[:16]}…")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Rejection failed: {e}")

# ── resolved approvals history ────────────────────────────────────────────────
section("Resolved", f"{len(resolved)} past decisions")

if not resolved:
    st.markdown('<div style="color:#555c72;font-size:0.8rem;padding:8px 0;">No resolved approvals yet.</div>',
                unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="display:grid;grid-template-columns:180px 120px 100px 120px 1fr;
                gap:8px;padding:8px 12px;
                font-size:0.63rem;color:#555c72;letter-spacing:0.12em;
                border-bottom:1px solid #1f2330;margin-bottom:2px;">
        <div>THREAD</div><div>MODEL</div><div>ACTION</div><div>DECISION</div><div>COMPLETED</div>
    </div>
    """, unsafe_allow_html=True)

    for run in resolved[:20]:
        tid_s    = run["thread_id"][:18] + "…"
        model    = run.get("model_id","—")
        action   = run.get("recommended_action","—")
        approved = run.get("human_approved")
        dec_txt  = "APPROVED" if approved else "REJECTED"
        dec_c    = "#00e5a0" if approved else "#ff4560"
        completed = (run.get("completed_at","")[:16]).replace("T"," ")

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:180px 120px 100px 120px 1fr;
                    gap:8px;padding:9px 12px;border-bottom:1px solid #13161e;font-size:0.76rem;">
            <div style="color:#555c72;font-size:0.72rem;">{tid_s}</div>
            <div style="color:#8b91a8;">{model}</div>
            <div style="color:#e8eaf0;">{action}</div>
            <div style="color:{dec_c};font-weight:600;">{'● ' + dec_txt}</div>
            <div style="color:#555c72;">{completed}</div>
        </div>
        """, unsafe_allow_html=True)

# ── auto-refresh when pending ─────────────────────────────────────────────────
if pending:
    st.markdown("""
    <div style="position:fixed;bottom:16px;right:20px;font-family:'JetBrains Mono',monospace;
                font-size:0.65rem;color:#ffb800;background:#0a0b0e;padding:4px 10px;
                border:1px solid rgba(255,184,0,0.3);border-radius:3px;">
        ⏸ polling every 5s
    </div>
    """, unsafe_allow_html=True)
    time.sleep(5)
    st.rerun()
