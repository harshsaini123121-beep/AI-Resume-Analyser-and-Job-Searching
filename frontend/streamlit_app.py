"""
ResumeLab — AI Resume Analyzer & Job Matching Platform
Modular Streamlit Entrypoint utilizing st.navigation, st.Page, and ResumeLab Design System.
"""
from datetime import datetime
import html
from pathlib import Path
import sys

import streamlit as st

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Typed API Client (Zero direct business logic or service imports)
from frontend.api_client import api_client

# Re-exported chart helpers for backward compatibility
from frontend.components.charts import (
    render_ats_category_chart,
    render_skill_radar_chart,
    render_match_signals_chart
)

__all__ = [
    "render_ats_category_chart",
    "render_skill_radar_chart",
    "render_match_signals_chart"
]

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeLab — AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Design System Injection (System Font Stack, Design Tokens, WCAG AA Contrast)
# -----------------------------------------------------------------------------
CSS_FILE = Path(__file__).parent / "styles.css"
if CSS_FILE.exists():
    css_content = CSS_FILE.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "ats_result" not in st.session_state:
    st.session_state.ats_result = None
if "active_filename" not in st.session_state:
    st.session_state.active_filename = None
if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None
if "is_sample" not in st.session_state:
    st.session_state.is_sample = False
if "analysis_timestamp" not in st.session_state:
    st.session_state.analysis_timestamp = None
if "page_count" not in st.session_state:
    st.session_state.page_count = None
if "active_job_match" not in st.session_state:
    st.session_state.active_job_match = None
if "active_job_evidence" not in st.session_state:
    st.session_state.active_job_evidence = None

# -----------------------------------------------------------------------------
# 4. Backend API Liveness Verification
# -----------------------------------------------------------------------------
if not api_client.is_healthy():
    st.error("⚠️ **Service Unavailable — Backend Disconnected**")
    st.markdown("""
    <div style="background: var(--color-danger-bg); border: 1px solid var(--color-danger-border); border-radius: 12px; padding: 1.5rem; margin-top: 1rem; margin-bottom: 2rem;">
        <div style="font-size: 1.25rem; font-weight: 800; color: var(--color-danger-text); margin-bottom: 0.5rem;">FastAPI Backend Is Offline</div>
        <div style="font-size: 0.95rem; color: var(--color-danger-text); line-height: 1.6;">
            The user interface communicates strictly through the backend REST API (<code>http://127.0.0.1:8000</code>).
            <br><br>
            <strong>How to start the API server:</strong>
            <ol style="margin-top: 0.5rem; margin-bottom: 0.5rem; padding-left: 1.25rem;">
                <li>Open a terminal in the project directory.</li>
                <li>Run: <code>.\\venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000</code></li>
            </ol>
            Once the server is running, refresh this page.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# 5. Top Fixed Header Bar (Pinned Recruiter Profile + Notification Bell)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="top-navbar">
    <div class="user-profile-badge" title="Recruiter: Harsh">
        <div class="bell-icon-container" title="Notifications">
            🔔
            <span class="bell-badge-dot"></span>
        </div>
        <div class="user-avatar" title="Harsh">H</div>
        <div class="user-details">
            <div class="user-name">Harsh</div>
            <div class="user-role">Recruiter</div>
        </div>
        <span class="user-profile-caret">▼</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Sidebar Context & Status (Maximum 5 Navigation Items, No Pro Plan Upsell)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0; margin-bottom: 1.25rem; padding: 0.1rem 0.2rem;">
        <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: #ffffff; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; font-weight: 800; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);">
            📄
        </div>
        <div>
            <div style="font-size: 1.25rem; font-weight: 800; color: var(--color-text); line-height: 1.1;">ResumeLab</div>
            <div style="font-size: 0.72rem; color: var(--color-muted); font-weight: 500;">AI Resume & Job Matching</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Candidate Status Pill in Sidebar
    if st.session_state.resume_data:
        cand_name = st.session_state.resume_data.get("candidate_name") or "Applicant"
        score_val = st.session_state.ats_result.get("total_score", 0) if st.session_state.ats_result else 0
        st.markdown(f"""
        <div style="background-color: var(--color-surface); border: 1px solid var(--color-border); border-left: 4px solid var(--color-primary); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px;">
            <div style="font-size: var(--font-xs); font-weight: 700; color: var(--color-muted);">ACTIVE RESUME</div>
            <div style="font-size: var(--font-sm); font-weight: 700; color: var(--color-text);">{html.escape(cand_name)}</div>
            <div style="font-size: var(--font-xs); color: var(--color-primary); font-weight: 600; margin-top: 2px;">ATS Score: {score_val}/100</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🗑️ Clear Active Resume", key="sidebar_clear_resume_btn", type="secondary", use_container_width=True):
            st.session_state.resume_data = None
            st.session_state.ats_result = None
            st.session_state.active_filename = None
            st.session_state.processed_file_hash = None
            st.session_state.active_job_match = None
            st.session_state.active_job_evidence = None
            st.rerun()

    # System Status Widget
    st.markdown("""
    <div style="background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 12px; padding: 0.85rem; margin-top: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
            <span style="font-size: 0.74rem; font-weight: 700; color: var(--color-muted); letter-spacing: 0.03em;">SYSTEM STATUS</span>
            <span style="display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; font-weight: 700; color: var(--color-success);">
                <span class="status-indicator-dot"></span> ONLINE
            </span>
        </div>
        <div style="font-size: 0.72rem; color: var(--color-muted); line-height: 1.5;">
            <div>• <b>API:</b> 127.0.0.1:8000</div>
            <div>• <b>Model:</b> all-MiniLM-L6-v2</div>
            <div>• <b>Dataset:</b> 721 Live Roles</div>
            <div>• <b>Engine:</b> Hybrid Dual-Signal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. Navigation Routing with st.navigation and st.Page (Exactly 5 Items)
# -----------------------------------------------------------------------------
pages = [
    st.Page("pages/1_home.py", title="Upload Resume", icon="📄", default=True),
    st.Page("pages/2_results.py", title="Analysis Results", icon="📊"),
    st.Page("pages/3_job_match.py", title="Job Match", icon="💼"),
    st.Page("pages/4_job_recommendations.py", title="Job Recommendations", icon="🎯"),
    st.Page("pages/5_batch_and_history.py", title="Batch & History", icon="🥞"),
]

nav = st.navigation(pages)
nav.run()
