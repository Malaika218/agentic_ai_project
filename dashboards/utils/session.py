# utils/session.py

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

def init_session():
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = os.getenv("MLOPS_API_URL")

    if "model_url" not in st.session_state:
        st.session_state["model_url"] = os.getenv("FRAUD_MODEL_MCP_URL")

    # 2. Sync permanent values BACK into the temporary sidebar inputs 
    # (Fixes text fields clearing out when you navigate back to app.py)
    st.session_state["_api_url"] = st.session_state["api_url"]
    st.session_state["_model_url"] = st.session_state["model_url"]


def sync_api_url():
    st.session_state["api_url"] = st.session_state["_api_url"]

def sync_model_url():
    st.session_state["model_url"] = st.session_state["_model_url"]
