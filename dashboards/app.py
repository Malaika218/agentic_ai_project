"""
dashboard/app.py — MLOps Agent Dashboard entry point.

Run with:
    streamlit run dashboard/app.py --server.port 8501
"""

import streamlit as st
from utils.session import init_session, sync_api_url, sync_model_url

init_session()

st.set_page_config(
    page_title="MLOps Agent Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
    --bg:          #0a0b0e;
    --surface:     #111318;
    --surface2:    #181b22;
    --border:      #1f2330;
    --border2:     #2a2f3d;
    --text:        #e8eaf0;
    --text2:       #8b91a8;
    --text3:       #555c72;
    --accent:      #00d4ff;
    --accent2:     #0099cc;
    --green:       #00e5a0;
    --amber:       #ffb800;
    --red:         #ff4560;
    --purple:      #9b59ff;
    --mono:        'JetBrains Mono', monospace;
    --display:     'Syne', sans-serif;
}

/* base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    font-family: var(--mono) !important;
    color: var(--text) !important;
}

/* metrics */
[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: var(--text2) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: var(--mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    font-family: var(--mono) !important;
}

/* buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 4px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0, 212, 255, 0.05) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #000 !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent2) !important;
    color: #000 !important;
}

/* inputs */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
}
.stTextInput input:focus, .stSelectbox select:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* selectbox */
[data-baseweb="select"] {
    background: var(--surface2) !important;
}
[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}

/* expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text2) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* divider */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* tabs */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text3) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* sidebar nav links */
[data-testid="stSidebarNav"] a {
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    color: var(--text2) !important;
    border-radius: 4px !important;
    padding: 6px 12px !important;
}
[data-testid="stSidebarNav"] a:hover {
    color: var(--accent) !important;
    background: rgba(0,212,255,0.05) !important;
}
[data-testid="stSidebarNav"] [aria-selected="true"] a {
    color: var(--accent) !important;
    background: rgba(0,212,255,0.08) !important;
}

/* alerts */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
}

/* spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* slider */
[data-baseweb="slider"] [data-testid="stSlider"] { }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
}

/* multiselect */
[data-baseweb="multi-select"] {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
}
[data-baseweb="tag"] {
    background: rgba(0,212,255,0.15) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}

/* radio */
[data-testid="stRadio"] label {
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    color: var(--text2) !important;
}
</style>
""", unsafe_allow_html=True)

# ── sidebar header ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 24px 0; border-bottom: 1px solid #1f2330; margin-bottom: 16px;">
        <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 800;
                    color: #00d4ff; letter-spacing: 0.05em;">⬡ MLOPS AGENT</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                    color: #555c72; letter-spacing: 0.15em; margin-top: 4px;">
            MONITORING SYSTEM v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    

    api_url = st.text_input(
        "API endpoint",
        key="_api_url",
        label_visibility="visible",
        on_change=sync_api_url,
    )

    model_url = st.text_input(
        "Model server",
        key="_model_url",
        label_visibility="visible",
        on_change=sync_model_url,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # quick health indicator in sidebar
    if st.button("↺  check health", use_container_width=True):
        try:
            import requests
            r = requests.get(f"{api_url}/health", timeout=60)
            s = r.json().get("services", {})
            for svc, ok in s.items():
                icon = "●" if ok else "○"
                color = "#00e5a0" if ok else "#ff4560"
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;'
                    f'color:{color};padding:2px 0;">{icon} {svc}</div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.markdown("""
    <div style="position:fixed;bottom:20px;left:0;width:260px;padding:0 16px;
                font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#555c72;">
        fraud-classifier-v1 · production
    </div>
    """, unsafe_allow_html=True)

# ── landing ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 48px 0 32px 0;">
    <div style="font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;
                color:#e8eaf0;line-height:1.1;letter-spacing:-0.02em;">
        Multi-Agent<br>
        <span style="color:#00d4ff;">MLOps Monitor</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                color:#8b91a8;margin-top:16px;max-width:520px;line-height:1.7;">
        Autonomous monitoring, diagnosis, and remediation for deployed ML models.
        Powered by LangGraph · Ollama · ChromaDB
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
pages = [
    ("01", "Overview",   "pages/1_Overview.py"),
    ("02", "Metrics",    "pages/2_Metrics.py"),
    ("03", "Incidents",  "pages/3_Incidents.py"),
    ("04", "Approvals",  "pages/4_Approvals.py"),
    ("05", "Runbooks",   "pages/5_Runbooks.py"),
    ("06", "Drift Lab",  "pages/6_Drift_Lab.py"),
]

for col, (num, name, _) in zip([col1,col2,col3,col4,col5,col6], pages):
    with col:
        st.markdown(f"""
        <div style="background:#111318;border:1px solid #1f2330;border-radius:8px;
                    padding:20px 16px;text-align:center;cursor:pointer;
                    transition:border-color 0.15s;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                        color:#555c72;letter-spacing:0.15em;">{num}</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:600;
                        color:#e8eaf0;margin-top:6px;">{name}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
st.info("↖ Use the sidebar to navigate between pages.")
