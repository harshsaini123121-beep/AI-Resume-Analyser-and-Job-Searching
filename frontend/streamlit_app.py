"""
ResumeLab — AI Resume Analyzer & Job Matching Platform
Production-grade SaaS UI with 100% genuine local ML inference:
- Robust file validation (PDF, DOCX, TXT) and SHA-256 duplicate processing prevention
- Clean initial empty state (no auto-loaded mock data)
- Clear distinction for demo/sample resumes
- Complete state reset workflow via New Analysis
- Strictly dynamic metrics (genuine ATS score, page count, word count, skills, experience)
- Consistent custom JD skill extraction & Explainable AI (XAI) evidence mapping
- Batch screening with descriptive per-file error handling, size validation & hash caching
- Guarded dataset indexing preventing empty dataframe IndexError
- Transparent active resume identification when replacement uploads fail
"""

from datetime import datetime
import hashlib
import html
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Typed API Client (UI contains zero direct business logic or service imports)
from frontend.api_client import api_client, APIConnectionError, APIResponseError

# -----------------------------------------------------------------------------
# Streamlit Page Config & ResumeLab Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeLab — AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

RESUMELAB_CSS = """
<style>
    /* Google Font: Plus Jakarta Sans */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Overall clean light SaaS background */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a;
    }

    /* Top Fixed Header styling matching Image 1 black background */
    header[data-testid="stHeader"], .stAppHeader {
        background-color: #0b0f19 !important;
        border-bottom: 1px solid #1e293b !important;
        z-index: 999990 !important;
    }

    /* Hide Streamlit Deploy button & any toolbar clutter so it never overlaps the recruiter profile */
    [data-testid="stDeployButton"],
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    .stDeployButton,
    header [data-testid="stToolbarActions"],
    header button[title="Deploy this app"],
    header [data-testid="stToolbar"] [data-testid="stToolbarActions"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }

    /* Recruiter Profile pinned inside the black header bar */
    .top-navbar {
        position: fixed !important;
        top: 0 !important;
        right: 3.5rem !important;
        height: 3.75rem !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding: 0 !important;
        margin: 0 !important;
        z-index: 1000000 !important;
        pointer-events: auto !important;
    }
    .user-profile-badge {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        background: #0f172a !important; /* solid sleek dark background so nothing behind can show */
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
    }
    .user-profile-badge:hover {
        background: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    .bell-icon-container {
        position: relative;
        font-size: 1.2rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #f1f5f9;
        margin-right: 0.35rem;
        transition: transform 0.15s ease;
    }
    .bell-icon-container:hover {
        transform: scale(1.1);
    }
    @keyframes bell-pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.75; }
    }
    .bell-badge-dot {
        position: absolute;
        top: 1px;
        right: 1px;
        width: 8px;
        height: 8px;
        background-color: #ef4444;
        border-radius: 50%;
        border: 1.5px solid #0b0f19;
        box-shadow: 0 0 6px rgba(239, 68, 68, 0.85);
        animation: bell-pulse 2s infinite ease-in-out;
    }
    .status-indicator-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 5px rgba(16, 185, 129, 0.8);
    }
    .user-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1.5px solid #38bdf8;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
    }
    .user-details {
        line-height: 1.2;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .user-name {
        font-size: 0.88rem;
        font-weight: 700;
        color: #f8fafc !important; /* Visible crisp white text against black */
        letter-spacing: -0.01em;
    }
    .user-role {
        font-size: 0.72rem;
        color: #94a3b8 !important; /* Visible soft silver slate against black */
        font-weight: 500;
    }
    .user-profile-caret {
        font-size: 0.65rem;
        color: #94a3b8;
        margin-left: 0.15rem;
        transition: color 0.15s ease;
    }
    .user-profile-badge:hover .user-profile-caret {
        color: #f8fafc;
    }

    /* Minimize vacant space below the black header */
    .block-container {
        padding-top: 3.85rem !important; /* snug directly below 3.75rem header */
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Prevent empty wrapper margin */
    div[data-testid="stMarkdownContainer"]:has(.top-navbar),
    .element-container:has(.top-navbar),
    div[data-testid="element-container"]:has(.top-navbar) {
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        line-height: 0 !important;
    }

    /* Page Header - compact spacing right below black header */
    .page-main-header {
        margin-top: 0 !important;
        margin-bottom: 1.25rem !important;
        padding-top: 0.15rem !important;
    }
    .page-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-top: 0 !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.2 !important;
    }
    .page-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        font-weight: 400;
        line-height: 1.4 !important;
    }

    /* Card styling */
    .resumelab-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .resumelab-card:hover {
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.08);
    }

    /* Upload Box (Dashed Container) */
    .upload-dashed-container {
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
        background-color: #f8fafc;
        padding: 2.2rem 1.5rem 1.25rem 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
        margin-bottom: 1.25rem;
    }
    .upload-dashed-container:hover {
        border-color: #3b82f6;
        background-color: #f0f7ff;
    }
    .upload-icon-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #eff6ff;
        color: #2563eb;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        margin-bottom: 0.85rem;
    }
    .upload-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .upload-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .upload-note {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.75rem;
    }

    /* Sample Resume Box */
    .sample-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 0.6rem;
    }
    .sample-info {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .sample-icon-bg {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        background-color: #fee2e2;
        color: #dc2626;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }

    /* Result Card Header */
    .result-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .file-meta-badge {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .file-icon-box {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background-color: #eff6ff;
        color: #2563eb;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }
    .status-pill-green {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
    }
    .status-pill-amber {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
    }

    /* Verdict Pill */
    .verdict-pill {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
    }
    .verdict-green { background-color: #dcfce7; color: #166534; }
    .verdict-blue { background-color: #dbeafe; color: #1e40af; }
    .verdict-amber { background-color: #fef3c7; color: #92400e; }
    .verdict-red { background-color: #fee2e2; color: #991b1b; }

    /* Quick Insights Mini-Cards */
    .insights-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.85rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .insight-card {
        background: #f8fafc;
        border: 1px solid #f1f5f9;
        border-radius: 12px;
        padding: 0.9rem 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .insight-icon-square {
        width: 42px;
        height: 42px;
        min-width: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    .icon-blue { background-color: #eff6ff; color: #2563eb; border: 1px solid #dbeafe; }
    .icon-green { background-color: #ecfdf5; color: #10b981; border: 1px solid #d1fae5; }
    .icon-purple { background-color: #f5f3ff; color: #7c3aed; border: 1px solid #ede9fe; }
    .insight-main {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .insight-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .insight-sub {
        font-size: 0.74rem;
        color: #94a3b8;
        line-height: 1.2;
    }

    /* Badge chips */
    .badge-chip {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #f1f5f9;
        color: #334155;
    }
    .badge-matched {
        background-color: #ecfdf5 !important;
        color: #065f46 !important;
        border: 1px solid #a7f3d0 !important;
    }
    .badge-missing {
        background-color: #fef2f2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca !important;
    }
    .badge-preferred {
        background-color: #f5f3ff !important;
        color: #6d28d9 !important;
        border: 1px solid #ddd6fe !important;
    }
    .badge-info {
        background-color: #eff6ff !important;
        color: #1e40af !important;
        border: 1px solid #bfdbfe !important;
    }

    /* Streamlit Sidebar Customization - Minimize vacant space above sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] .block-container,
    div[data-testid="stSidebar"] div.st-emotion-cache-16txtl3,
    div[data-testid="stSidebar"] div.st-emotion-cache-10oheav,
    div[data-testid="stSidebar"] div.st-emotion-cache-1cypcdb {
        padding-top: 0.6rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1.5rem !important;
    }
    div[data-testid="stSidebarHeader"] {
        height: auto !important;
        min-height: 0 !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0 !important;
        margin-bottom: -0.5rem !important;
    }
    [data-testid="stSidebarNav"] {
        display: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 4px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background-color: transparent;
        border-radius: 10px;
        padding: 0.6rem 0.85rem;
        font-weight: 600;
        color: #475569;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #f1f5f9;
        color: #0f172a;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background-color: #eff6ff !important;
        color: #2563eb !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        padding: 0.65rem 1.25rem !important;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #2563eb !important;
    }

    /* Buttons Styling */
    button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.15s ease !important;
    }
    button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.25) !important;
    }
    button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        border: 1.5px solid #2563eb !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.15s ease !important;
    }
    button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
        background-color: #eff6ff !important;
        border-color: #1d4ed8 !important;
    }

    /* Detailed category breakdown bars */
    .cat-bar-container {
        margin-bottom: 0.75rem;
    }
    .cat-header-flex {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .cat-progress-bg {
        width: 100%;
        height: 8px;
        background-color: #e2e8f0;
        border-radius: 9999px;
        overflow: hidden;
    }
    .cat-progress-fill {
        height: 100%;
        border-radius: 9999px;
    }
</style>
"""
st.markdown(RESUMELAB_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Backend API Liveness Verification (Service Unavailable Protection)
# -----------------------------------------------------------------------------
if not api_client.is_healthy():
    st.error("⚠️ **Service Unavailable — Backend Disconnected**")
    st.markdown("""
    <div style="background: #fef2f2; border: 1px solid #fca5a5; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; margin-bottom: 2rem;">
        <div style="font-size: 1.25rem; font-weight: 800; color: #991b1b; margin-bottom: 0.5rem;">FastAPI Backend Is Offline</div>
        <div style="font-size: 0.95rem; color: #7f1d1d; line-height: 1.6;">
            The ResumeLab user interface operates strictly through the backend REST API (<code>http://127.0.0.1:8000</code>).
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


def get_jobs_dataframe() -> pd.DataFrame:
    """Retrieves curated job positions from the backend API."""
    try:
        jobs = api_client.list_jobs(limit=100)
        if jobs:
            return pd.DataFrame(jobs)
    except Exception:
        pass
    return pd.DataFrame(columns=[
        "job_id", "title", "company", "location", "salary",
        "experience_required", "description", "required_skills"
    ])



# -----------------------------------------------------------------------------
# Circular SVG Donut Gauge Generator
# -----------------------------------------------------------------------------
def render_donut_svg(score: float, max_score: float = 100.0) -> str:
    """Generates the crisp circular donut gauge matching ResumeLab UI."""
    score = max(0.0, min(float(score), max_score))
    radius = 42
    circ = 2 * 3.14159265 * radius  # ~263.89
    offset = circ * (1.0 - (score / max_score))

    if score >= 80:
        bar_color = "#10b981"  # Emerald green
        verdict = "Good Match"
        verdict_class = "verdict-green"
    elif score >= 60:
        bar_color = "#2563eb"  # Royal blue
        verdict = "Competitive"
        verdict_class = "verdict-blue"
    elif score >= 40:
        bar_color = "#f59e0b"  # Amber
        verdict = "Needs Work"
        verdict_class = "verdict-amber"
    else:
        bar_color = "#ef4444"  # Red
        verdict = "Low Alignment"
        verdict_class = "verdict-red"

    svg_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 0.5rem 0;">
        <svg width="150" height="150" viewBox="0 0 110 110">
            <!-- Track -->
            <circle cx="55" cy="55" r="{radius}" fill="none" stroke="#f1f5f9" stroke-width="9" />
            <!-- Progress Arc -->
            <circle cx="55" cy="55" r="{radius}" fill="none" stroke="{bar_color}" stroke-width="9"
                    stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
                    stroke-linecap="round" transform="rotate(-90 55 55)" />
            <!-- Center Score -->
            <text x="55" y="52" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif" font-size="26" font-weight="800" fill="#0f172a">{int(round(score))}</text>
            <!-- /100 -->
            <text x="55" y="68" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif" font-size="11" font-weight="600" fill="#94a3b8">/100</text>
        </svg>
        <div class="verdict-pill {verdict_class}" style="margin-top: 0.5rem;">{verdict}</div>
    </div>
    """
    return svg_html


def render_ats_category_chart(category_breakdown: dict) -> Optional[go.Figure]:
    """Renders a polished, interactive horizontal bar chart for the 8 ATS categories."""
    if not category_breakdown:
        return None

    categories = []
    scores = []
    max_scores = []
    percentages = []
    colors = []
    hover_texts = []

    for cat_name, info in category_breakdown.items():
        score = info.get("score", 0)
        max_s = info.get("max_score", 10)
        pct = int((score / max_s) * 100) if max_s > 0 else 0
        details = info.get("details", "")

        categories.append(cat_name)
        scores.append(score)
        max_scores.append(max_s)
        percentages.append(pct)

        if pct >= 80:
            colors.append("#10b981")  # Emerald
        elif pct >= 60:
            colors.append("#2563eb")  # Royal blue
        elif pct >= 40:
            colors.append("#f59e0b")  # Amber
        else:
            colors.append("#ef4444")  # Red

        hover_texts.append(f"<b>{cat_name}</b><br>Score: {score}/{max_s} pts ({pct}%)<br>Details: {details}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=percentages,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.6)', width=1)
        ),
        text=[f"{p}% ({s}/{m})" for p, s, m in zip(percentages, scores, max_scores)],
        textposition='inside',
        textfont=dict(family='Plus Jakarta Sans', size=11, color='#ffffff'),
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor='#f1f5f9',
            tickfont=dict(family='Plus Jakarta Sans', size=11, color='#64748b'),
            ticksuffix='%'
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(family='Plus Jakarta Sans', size=12, color='#0f172a')
        ),
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


def render_skill_radar_chart(categories: dict, candidate_skills: list) -> Optional[go.Figure]:
    """Renders a sleek Radar / Spider chart showing candidate skill density across domains."""
    if not categories or not candidate_skills:
        return None

    candidate_skills_lower = set(str(s).lower() for s in candidate_skills)
    domain_names = []
    coverage_pcts = []

    for domain, domain_skills in categories.items():
        if isinstance(domain_skills, list) and len(domain_skills) > 0:
            matched = sum(1 for s in domain_skills if str(s).lower() in candidate_skills_lower)
            if matched > 0:
                pct = min(100.0, round((matched / min(8, len(domain_skills))) * 100, 1))
                domain_names.append(domain.replace('_', ' ').title())
                coverage_pcts.append(pct)

    if len(domain_names) < 3:
        return None

    radar_labels = domain_names + [domain_names[0]]
    radar_vals = coverage_pcts + [coverage_pcts[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_vals,
        theta=radar_labels,
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.20)',
        line=dict(color='#2563eb', width=2),
        marker=dict(size=6, color='#1d4ed8')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#e2e8f0'),
            angularaxis=dict(tickfont=dict(family='Plus Jakarta Sans', size=11, color='#334155'))
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=25, b=25),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def render_match_signals_chart(skill_score: float, semantic_score: float, exp_score: float) -> go.Figure:
    """Renders grouped bar visualizer showing individual signal contributions."""
    signals = ["Skill Overlap (50%)", "Semantic Vector (40%)", "Tenure Alignment (10%)"]
    vals = [round(skill_score * 100, 1), round(semantic_score * 100, 1), round(exp_score * 100, 1)]
    colors = ["#10b981", "#2563eb", "#8b5cf6"]

    fig = go.Figure(go.Bar(
        x=signals,
        y=vals,
        marker=dict(color=colors),
        text=[f"{v}%" for v in vals],
        textposition='auto',
        textfont=dict(family='Plus Jakarta Sans', size=12, color='#ffffff')
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=15, b=10),
        height=230,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor='#f1f5f9',
            ticksuffix='%'
        ),
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


# -----------------------------------------------------------------------------
# Session State Initialization (Starts in Clean Empty State)
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
if "show_detailed_analysis" not in st.session_state:
    st.session_state.show_detailed_analysis = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "rejected_replacement_notice" not in st.session_state:
    st.session_state.rejected_replacement_notice = None


def reset_analysis_state():
    """Resets all active resume analysis session state to return to clean empty state."""
    st.session_state.resume_data = None
    st.session_state.ats_result = None
    st.session_state.active_filename = None
    st.session_state.processed_file_hash = None
    st.session_state.is_sample = False
    st.session_state.analysis_timestamp = None
    st.session_state.page_count = None
    st.session_state.show_detailed_analysis = False
    st.session_state.active_job_match = None
    st.session_state.active_job_evidence = None
    st.session_state.rejected_replacement_notice = None
    # Increment uploader key so file_uploader widget is cleanly emptied
    st.session_state.uploader_key += 1


# -----------------------------------------------------------------------------
# Left Sidebar (ResumeLab Navigation)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0; margin-bottom: 1.25rem; padding: 0.1rem 0.2rem;">
        <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: #ffffff; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; font-weight: 800; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);">
            📄
        </div>
        <div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; line-height: 1.1;">ResumeLab</div>
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 500;">AI Resume & Job Matching</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_page = st.radio(
        "Navigation",
        [
            "📄  Analyze Resume",
            "💼  Job Matching",
            "🥞  Batch Screening",
            "🪄  Resume Optimizer",
            "🎯  Job Recommendations",
            "📊  Reports"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.85rem; margin-top: 0.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
            <span style="font-size: 0.74rem; font-weight: 700; color: #475569; letter-spacing: 0.03em;">SYSTEM STATUS</span>
            <span style="display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; font-weight: 700; color: #10b981;">
                <span class="status-indicator-dot"></span> ONLINE
            </span>
        </div>
        <div style="font-size: 0.72rem; color: #64748b; line-height: 1.5;">
            <div>• <b>API:</b> 127.0.0.1:8000</div>
            <div>• <b>Model:</b> all-MiniLM-L6-v2</div>
            <div>• <b>Dataset:</b> 721 Live Roles</div>
            <div>• <b>Engine:</b> Hybrid Dual-Signal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# Global Top Header Bar (Notification Bell + Recruiter Profile pinned to black header)
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
# PAGE 1: ANALYZE RESUME (Matches Exact Layout in Screenshot)
# -----------------------------------------------------------------------------
if "Analyze Resume" in nav_page:
    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Analyze Your Resume</h1>
        <p class="page-subtitle">Get AI-powered insights, ATS score, and best job matches.</p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1.0, 1.15], gap="large")

    # ---------------- LEFT COLUMN: UPLOAD & SAMPLE AREA ----------------
    with left_col:
        st.markdown("""
        <div class="resumelab-card">
            <div class="upload-dashed-container">
                <div class="upload-icon-circle">📄</div>
                <div class="upload-title">Upload your resume</div>
                <div class="upload-subtitle">Drag and drop a PDF, DOCX, or TXT file here or click to browse</div>
        """, unsafe_allow_html=True)

        # File Uploader with dynamic key to allow clean reset
        uploaded_file = st.file_uploader(
            "Choose File",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            key=f"resume_uploader_{st.session_state.uploader_key}"
        )
        st.markdown("""
                <div class="upload-note">Supported formats: PDF, DOCX, TXT (Max 5 MB)</div>
            </div>
        """, unsafe_allow_html=True)

        # Process upload safely using SHA-256 hash to prevent duplicate runs on rerun
        upload_rejection_reason = None

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            max_size_bytes = 5 * 1024 * 1024

            if len(raw_bytes) == 0:
                upload_rejection_reason = "File is empty (0 bytes)."
            elif len(raw_bytes) > max_size_bytes:
                upload_rejection_reason = f"File size ({len(raw_bytes) / 1024 / 1024:.1f} MB) exceeds maximum allowed limit of 5 MB."
            else:
                current_hash = hashlib.sha256(raw_bytes).hexdigest()
                # Only analyze if this is a genuinely new file content
                if st.session_state.get("processed_file_hash") != current_hash:
                    with st.spinner("Analyzing resume structure, extracting skills, and calculating ATS score..."):
                        try:
                            analysis = api_client.analyse_resume(raw_bytes, uploaded_file.name)
                            data = analysis["extracted_data"]
                            ats = analysis["ats_score"]
                            p_count = analysis.get("page_count", 1)

                            # Successful upload! Update state
                            st.session_state.resume_data = data
                            st.session_state.ats_result = ats
                            st.session_state.page_count = p_count
                            st.session_state.active_filename = uploaded_file.name
                            st.session_state.processed_file_hash = current_hash
                            st.session_state.is_sample = False
                            st.session_state.analysis_timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
                            st.session_state.show_detailed_analysis = False
                            st.session_state.active_job_match = None
                            st.session_state.active_job_evidence = None
                            st.session_state.rejected_replacement_notice = None
                        except APIResponseError as are:
                            upload_rejection_reason = f"Document Processing Error: {are.detail}"
                        except APIConnectionError as ace:
                            upload_rejection_reason = f"Backend Connection Error: {str(ace)}"
                        except Exception as ex:
                            upload_rejection_reason = f"Analysis failed: {str(ex)}"

            # Handle upload rejection without corrupting or misrepresenting retained state
            if upload_rejection_reason:
                st.error(f"❌ Upload Rejected: {upload_rejection_reason}")
                if st.session_state.resume_data is not None:
                    st.session_state.rejected_replacement_notice = {
                        "rejected_name": uploaded_file.name,
                        "reason": upload_rejection_reason,
                        "active_name": st.session_state.active_filename
                    }
                    st.warning(
                        f"⚠️ **Notice**: Retaining previous active analysis for **{st.session_state.active_filename}**. "
                        f"The new file **{uploaded_file.name}** was rejected."
                    )
        else:
            st.session_state.rejected_replacement_notice = None

        # Try a sample section
        sample_col1, sample_col2 = st.columns([2, 1])
        with sample_col1:
            st.markdown('<div style="font-weight: 700; font-size: 1rem; color: #0f172a; margin-top: 0.5rem;">Try a sample</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 0.82rem; color: #64748b; margin-bottom: 0.8rem;">Explore with a sample resume to see how it works.</div>', unsafe_allow_html=True)
        with sample_col2:
            st.markdown("<div style='text-align: right; margin-top: 0.5rem;'>", unsafe_allow_html=True)
            if st.button("Use Sample →", key="use_sample_top_link", type="secondary", width="stretch"):
                sample_p = ROOT_DIR / "data" / "sample_resumes" / "software_engineer_resume.txt"
                if sample_p.exists():
                    raw_text = sample_p.read_text(encoding="utf-8")
                    s_bytes = raw_text.encode("utf-8")
                    s_hash = hashlib.sha256(s_bytes).hexdigest()
                    try:
                        analysis = api_client.analyse_resume(s_bytes, "software_engineer_resume.txt")
                        st.session_state.resume_data = analysis["extracted_data"]
                        st.session_state.ats_result = analysis["ats_score"]
                        st.session_state.page_count = analysis.get("page_count", 1)
                        st.session_state.active_filename = "software_engineer_resume.txt"
                        st.session_state.processed_file_hash = s_hash
                        st.session_state.is_sample = True
                        st.session_state.analysis_timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
                        st.session_state.show_detailed_analysis = False
                        st.session_state.active_job_match = None
                        st.session_state.active_job_evidence = None
                        st.session_state.rejected_replacement_notice = None
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Sample analysis failed: {ex}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Sample Resume Card
        s_card_col, s_btn_col = st.columns([3, 1])
        with s_card_col:
            st.markdown("""
            <div class="sample-box">
                <div class="sample-info">
                    <div class="sample-icon-bg">📄</div>
                    <div>
                        <div style="font-size: 0.92rem; font-weight: 700; color: #0f172a;">Software Engineer Resume</div>
                        <div style="font-size: 0.78rem; color: #64748b;">Sample Resume • TXT Demonstration</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with s_btn_col:
            st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
            if st.button("Analyze", key="analyze_sample_btn", type="secondary", width="stretch"):
                sample_p = ROOT_DIR / "data" / "sample_resumes" / "software_engineer_resume.txt"
                if sample_p.exists():
                    raw_text = sample_p.read_text(encoding="utf-8")
                    s_bytes = raw_text.encode("utf-8")
                    s_hash = hashlib.sha256(s_bytes).hexdigest()
                    try:
                        analysis = api_client.analyse_resume(s_bytes, "software_engineer_resume.txt")
                        st.session_state.resume_data = analysis["extracted_data"]
                        st.session_state.ats_result = analysis["ats_score"]
                        st.session_state.page_count = analysis.get("page_count", 1)
                        st.session_state.active_filename = "software_engineer_resume.txt"
                        st.session_state.processed_file_hash = s_hash
                        st.session_state.is_sample = True
                        st.session_state.analysis_timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
                        st.session_state.show_detailed_analysis = False
                        st.session_state.active_job_match = None
                        st.session_state.active_job_evidence = None
                        st.session_state.rejected_replacement_notice = None
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Sample analysis failed: {ex}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RIGHT COLUMN: RESULTS OR CLEAN EMPTY STATE ----------------
    with right_col:
        # Case A: Clean Initial Empty State when no resume has been uploaded or chosen
        if st.session_state.resume_data is None:
            st.markdown("""
            <div class="resumelab-card" style="text-align: center; padding: 3rem 2rem;">
                <div style="width: 64px; height: 64px; border-radius: 50%; background-color: #f1f5f9; color: #64748b; display: inline-flex; align-items: center; justify-content: center; font-size: 2rem; margin-bottom: 1.25rem;">
                    📊
                </div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;">
                    Ready for Analysis
                </div>
                <p style="font-size: 0.95rem; color: #64748b; line-height: 1.5; max-width: 480px; margin: 0 auto 1.75rem auto;">
                    Upload your resume (PDF, DOCX, or TXT) on the left to get instant ATS compatibility scores, skill extraction, and real-world job matching. Or explore using a sample demonstration.
                </p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; text-align: left; max-width: 500px; margin: 0 auto 1.75rem auto;">
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.85rem;">
                        <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">🎯 100-Pt ATS Rubric</div>
                        <div style="font-size: 0.76rem; color: #64748b;">Evaluates 8 quantitative scanner dimensions.</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.85rem;">
                        <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">⚡ 721 Live Tech Jobs</div>
                        <div style="font-size: 0.76rem; color: #64748b;">Dual-signal hybrid semantic matching.</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.85rem;">
                        <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">🔍 Explainable AI (XAI)</div>
                        <div style="font-size: 0.76rem; color: #64748b;">Maps candidate sentences to JD requirements.</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.85rem;">
                        <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">🪄 Google X-Y-Z Auditor</div>
                        <div style="font-size: 0.76rem; color: #64748b;">Rewrites bullet points to hiring standards.</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Case B: Results View (Current Active Resume Analysis)
        else:
            # Check if there is an active rejection notice from a failed replacement upload
            rej_notice = st.session_state.get("rejected_replacement_notice")
            if rej_notice:
                st.warning(
                    f"⚠️ **Active Analysis**: Currently displaying results for **'{rej_notice['active_name']}'** (retained from previous analysis). "
                    f"The replacement file **'{rej_notice['rejected_name']}'** could not be processed ({rej_notice['reason']})."
                )
                col_clear_ret, _ = st.columns([1, 2])
                with col_clear_ret:
                    if st.button("🗑️ Discard Retained Results", key="discard_retained_btn", type="secondary"):
                        reset_analysis_state()
                        st.rerun()

            data = st.session_state.resume_data
            ats = st.session_state.ats_result
            status_badge = (
                '<span class="status-pill-amber">Sample Resume (Demo)</span>'
                if st.session_state.is_sample
                else '<span class="status-pill-green">Analysis Completed</span>'
            )
            timestamp_str = html.escape(st.session_state.analysis_timestamp or datetime.now().strftime("%d %b %Y, %I:%M %p"))
            filename_str = html.escape(st.session_state.active_filename or "resume.pdf")

            # Render Outer Card
            st.markdown(f"""
            <div class="resumelab-card">
                <div class="result-header-bar">
                    <div class="file-meta-badge">
                        <div class="file-icon-box">📄</div>
                        <div>
                            <div style="font-weight: 700; font-size: 1.05rem; color: #0f172a; display: flex; align-items: center; gap: 0.65rem;">
                                {filename_str}
                                {status_badge}
                            </div>
                            <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.15rem;">
                                Analysed on {timestamp_str}
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Action buttons right under header (Download PDF & New Analysis)
            h_btn_col1, h_btn_col2 = st.columns([2, 1])
            with h_btn_col1:
                try:
                    pdf_bytes = api_client.generate_pdf_report(
                        candidate_name=data.get("candidate_name") or "Applicant",
                        email=data.get("email"),
                        phone=data.get("phone"),
                        ats_score=ats.get("total_score", 0),
                        ats_grade=ats.get("grade") or "C",
                        category_breakdown=ats.get("category_breakdown", {}),
                        strengths=ats.get("strengths", []),
                        issues=ats.get("issues", []),
                        suggestions=ats.get("suggestions", []),
                        skills=data.get("skills", [])
                    )
                    st.download_button(
                        label="📥 Download ATS Audit PDF",
                        data=pdf_bytes,
                        file_name=f"ATS_Audit_{data.get('candidate_name', 'Applicant')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as pex:
                    st.caption(f"PDF generator offline: {pex}")
            with h_btn_col2:
                if st.button("🔄 New Analysis", key="reset_analysis_btn", type="secondary", width="stretch"):
                    reset_analysis_state()
                    st.rerun()


            # Tabs: Overview, Skills, Job Match, Suggestions
            tab_overview, tab_skills, tab_match, tab_sug = st.tabs([
                "Overview",
                "Skills",
                "Job Match",
                "Suggestions"
            ])

            # ---------------- TAB 1: OVERVIEW ----------------
            with tab_overview:
                score_val = ats.get("total_score", 0)

                # Gauge Row
                g_col1, g_col2 = st.columns([1, 1.6])
                with g_col1:
                    st.markdown(render_donut_svg(score_val), unsafe_allow_html=True)

                with g_col2:
                    # Dynamic verdict description based on score band
                    if score_val >= 80:
                        desc_text = "Your resume is well-optimized for ATS systems with strong keyword density and clean structure. A few refinements can maximize recruiter impact."
                    elif score_val >= 60:
                        desc_text = "Your resume demonstrates a solid foundation. Addressing identified gaps and adding quantifiable metrics will significantly improve ranking."
                    elif score_val >= 40:
                        desc_text = "Your resume needs targeted improvements in skill coverage, standard section formatting, and active measurable accomplishments."
                    else:
                        desc_text = "Your resume requires significant revision. Essential contact details, standard section headers, or core technical skills were not detected."

                    st.markdown(f"""
                    <div style="padding-top: 1.25rem;">
                        <div style="font-size: 1.35rem; font-weight: 800; color: #0f172a; margin-bottom: 0.4rem;">ATS Compatibility</div>
                        <p style="font-size: 0.92rem; color: #64748b; line-height: 1.5; margin-bottom: 0;">
                            {desc_text}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Quick Insights Section (Strictly Dynamic Derived Metrics)
                st.markdown('<div style="font-size: 1rem; font-weight: 800; color: #0f172a; margin-top: 1.25rem; margin-bottom: 0.4rem;">Quick Insights</div>', unsafe_allow_html=True)

                # Page count metric: from parser if supported, otherwise "Not available"
                pg_count = st.session_state.get("page_count")
                if pg_count is not None:
                    pages_display = f"{pg_count}"
                    pages_sub = "Ideal length" if 1 <= pg_count <= 2 else "Document pages"
                else:
                    pages_display = "N/A"
                    pages_sub = "Not available (TXT/DOCX)"

                # Skills count metric: from normalized detected skills
                skills_count = len(data.get("skills", []))

                # Experience metric: from tenure heuristics or "Not detected"
                exp_years = data.get("experience_years")
                if exp_years is not None and exp_years > 0.0:
                    exp_display = f"{exp_years:.0f} Years" if exp_years.is_integer() else f"{exp_years:.1f} Years"
                    exp_sub = "Good range" if exp_years >= 2.0 else "Entry level"
                else:
                    exp_display = "Not detected"
                    exp_sub = "Add date ranges"

                st.markdown(f"""
                <div class="insights-grid">
                    <div class="insight-card">
                        <div class="insight-icon-square icon-blue">📄</div>
                        <div>
                            <div class="insight-main">{pages_display}</div>
                            <div class="insight-label">Pages</div>
                            <div class="insight-sub">{pages_sub}</div>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-icon-square icon-green">📋</div>
                        <div>
                            <div class="insight-main">{skills_count}</div>
                            <div class="insight-label">Key Skills</div>
                            <div class="insight-sub">Detected</div>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-icon-square icon-purple">💼</div>
                        <div>
                            <div class="insight-main">{exp_display}</div>
                            <div class="insight-label">Experience</div>
                            <div class="insight-sub">{exp_sub}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Bottom Action Buttons
                btn_col1, btn_col2 = st.columns([1, 1], gap="medium")
                with btn_col1:
                    if st.button("View Detailed Analysis →", key="view_details_btn", type="secondary", width="stretch"):
                        st.session_state.show_detailed_analysis = not st.session_state.show_detailed_analysis
                        st.rerun()

                with btn_col2:
                    try:
                        pdf_bytes = api_client.generate_pdf_report(
                            candidate_name=data.get("candidate_name", "Applicant"),
                            email=data.get("email"),
                            phone=data.get("phone"),
                            ats_score=ats.get("total_score", 0),
                            ats_grade=ats.get("grade", "N/A"),
                            category_breakdown=ats.get("category_breakdown", {}),
                            strengths=ats.get("strengths", []),
                            issues=ats.get("issues", []),
                            suggestions=ats.get("suggestions", []),
                            skills=data.get("skills", [])
                        )
                        safe_dl_name = filename_str.rsplit(".", 1)[0]
                        st.download_button(
                            label="📥 Download Report",
                            data=pdf_bytes,
                            file_name=f"{safe_dl_name}_Audit_Report.pdf",
                            mime="application/pdf",
                            type="primary",
                            width="stretch"
                        )
                    except Exception as pex:
                        st.caption(f"PDF download unavailable: {pex}")

                # Expandable Detailed Analysis Drawer
                if st.session_state.show_detailed_analysis:
                    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
                    st.markdown("#### 📊 ATS Scoring Rubric Breakdown (8 Dimensions)")

                    cats = ats.get("category_breakdown", {})
                    cat_fig = render_ats_category_chart(cats)
                    if cat_fig:
                        st.plotly_chart(cat_fig, use_container_width=True, config={"displayModeBar": False})

                    d_c1, d_c2 = st.columns(2)
                    with d_c1:
                        st.markdown("##### ✅ Detected Strengths")
                        for s in ats.get("strengths", []):
                            st.markdown(f"- **{s}**")
                        if not ats.get("strengths"):
                            st.info("No standout strengths detected yet.")
                    with d_c2:
                        st.markdown("##### ⚠️ Actionable Fixes")
                        for iss in ats.get("issues", []):
                            st.markdown(f"- {iss}")
                        if not ats.get("issues"):
                            st.success("No critical issues found!")

            # ---------------- TAB 2: SKILLS ----------------
            with tab_skills:
                skills_list = data.get("skills", [])
                st.markdown(f"#### Identified Technical Stack ({len(skills_list)} Detected)")

                if skills_list:
                    chips_html = "".join([f'<span class="badge-chip badge-matched">✓ {html.escape(str(s))}</span>' for s in skills_list])
                    st.markdown(f"<div style='margin-bottom: 1.25rem;'>{chips_html}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No standard technical skills were detected. Consider adding a clear 'Skills' section.")

                st.markdown("#### 🎯 Domain Coverage & Density")
                try:
                    skill_db = api_client.get_skills_taxonomy()
                    categories = skill_db.get("categories", {})
                except Exception:
                    categories = {}

                # Render Plotly Skill Radar
                radar_fig = render_skill_radar_chart(categories, skills_list)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

                resume_skills_lower = set(s.lower() for s in skills_list)
                found_any_domain = False
                for domain, domain_skills in categories.items():
                    if isinstance(domain_skills, list):
                        found = [s for s in domain_skills if s.lower() in resume_skills_lower]
                        if found:
                            found_any_domain = True
                            st.markdown(f"**{domain.title()}:** {', '.join(found)}")

                if not found_any_domain:
                    st.info("Upload a resume with programming, cloud, or data skills to inspect domain coverage.")

            # ---------------- TAB 3: JOB MATCH ----------------
            with tab_match:
                st.markdown("#### Target Role Alignment & Explainable AI (XAI)")
                jobs_df = get_jobs_dataframe()

                if jobs_df is None or jobs_df.empty:
                    st.warning("No job positions are currently available from the backend API.")
                else:
                    job_options = [f"{r['job_id']}: {r['title']} @ {r['company']}" for _, r in jobs_df.head(25).iterrows()]
                    if not job_options:
                        st.warning("No selectable jobs found in the dataset.")
                    else:
                        selected_role = st.selectbox(
                            "Select Target Position to Compare",
                            job_options
                        )
                        matching_rows = jobs_df[jobs_df["job_id"] == selected_role.split(":")[0].strip()]
                        if matching_rows.empty:
                            st.error("Selected position could not be resolved from dataset.")
                        else:
                            target_job = matching_rows.iloc[0]
                            t_skills = [s.strip().lower() for s in str(target_job["required_skills"]).replace(";", ",").split(",") if s.strip()]
                            candidate_tenure = float(data.get("experience_years") or 0.0)
                            required_tenure = float(target_job.get("experience_required", 0.0) or 0.0)

                            try:
                                match_res = api_client.match_resume_with_job(
                                    resume_text=data.get("raw_text", ""),
                                    resume_skills=data.get("skills", []),
                                    candidate_exp_years=candidate_tenure,
                                    jd_text=str(target_job["description"]),
                                    jd_skills=t_skills,
                                    required_exp_years=required_tenure
                                )

                                m1, m2, m3, m4 = st.columns(4)
                                with m1:
                                    st.metric("Total Alignment", f"{match_res['match_percentage']}%")
                                with m2:
                                    st.metric("Skill Match", f"{int(match_res['skill_score'] * 100)}%")
                                with m3:
                                    st.metric("Semantic Fit", f"{int(match_res['semantic_score'] * 100)}%")
                                with m4:
                                    st.metric("Tenure Match", f"{int(match_res['experience_score'] * 100)}%")

                                # Grouped signal contribution chart
                                sig_fig = render_match_signals_chart(
                                    match_res['skill_score'],
                                    match_res['semantic_score'],
                                    match_res['experience_score']
                                )
                                st.plotly_chart(sig_fig, use_container_width=True, config={"displayModeBar": False})

                                # Skill Overlap Breakdown
                                matched_sk = match_res.get("matched_skills", [])
                                missing_sk = match_res.get("missing_skills", [])
                                if matched_sk or missing_sk:
                                    st.markdown("##### 🎯 Skill Overlap Breakdown")
                                    if matched_sk:
                                        m_chips = "".join([f'<span class="badge-chip badge-matched">✓ {html.escape(str(s))}</span>' for s in matched_sk])
                                        st.markdown(f"**Matched Required Skills ({len(matched_sk)}):**<br>{m_chips}", unsafe_allow_html=True)
                                    if missing_sk:
                                        ms_chips = "".join([f'<span class="badge-chip badge-missing">✗ {html.escape(str(s))}</span>' for s in missing_sk])
                                        st.markdown(f"**Missing Required Skills ({len(missing_sk)}):**<br>{ms_chips}", unsafe_allow_html=True)

                                st.markdown("##### 🔍 Sentence-Level Evidence Mapping")
                                evidence = api_client.get_xai_evidence(
                                    resume_text=data.get("raw_text", ""),
                                    jd_text=str(target_job["description"]),
                                    top_n_requirements=3
                                )
                                for ev in evidence:
                                    is_strong = ev.get("status") == "Strong Evidence"
                                    border_c = "#10b981" if is_strong else "#3b82f6"
                                    status_badge = '<span class="status-pill-green">Strong Match</span>' if is_strong else '<span class="status-pill-amber">Moderate Inferred</span>'
                                    escaped_req = html.escape(str(ev.get('job_requirement', '')))
                                    escaped_ev = html.escape(str(ev.get('resume_evidence', '')))
                                    st.markdown(f"""
                                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {border_c}; border-radius: 10px; padding: 0.95rem; margin-bottom: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                                            <span style="font-size: 0.76rem; font-weight: 700; color: #64748b; letter-spacing: 0.02em;">REQUISITION REQUIREMENT</span>
                                            {status_badge}
                                        </div>
                                        <div style="font-size: 0.9rem; color: #0f172a; font-weight: 600; margin-bottom: 0.5rem; line-height: 1.35;">"{escaped_req}"</div>
                                        <div style="font-size: 0.76rem; font-weight: 700; color: #2563eb; letter-spacing: 0.02em; margin-bottom: 0.15rem;">CANDIDATE RESUME PROOF</div>
                                        <div style="font-size: 0.88rem; color: #334155; line-height: 1.4; background: #f8fafc; padding: 0.45rem 0.65rem; border-radius: 6px;">"{escaped_ev}"</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            except Exception as ex:
                                st.error(f"Failed to calculate match via API: {ex}")

            # ---------------- TAB 4: SUGGESTIONS ----------------
            with tab_sug:
                st.markdown("#### Actionable Resume Enhancements")
                sug_list = ats.get("suggestions", [])
                if sug_list:
                    for sug in sug_list:
                        st.markdown(f"- 💡 {sug}")
                else:
                    st.success("No major suggestions needed; resume formatting is strong!")

                st.markdown("<br>#### 🪄 Google X-Y-Z Bullet Point Optimization", unsafe_allow_html=True)
                st.info("Google formula: 'Accomplished [X] as measured by [Y], by doing [Z]'")
                sample_bullet = "Developed and maintained web applications using Python and React, improving throughput by 35%."
                try:
                    opt_res = api_client.optimize_bullet(sample_bullet)
                    st.markdown(f"**Current Bullet:** *{sample_bullet}*")
                    st.markdown(f"**Quality Score:** {opt_res['score']}/100 ({opt_res['grade']})")
                    st.success(f"**Suggested High-Impact Rewrite:**\n\n{opt_res['improved_rewrite']}")
                except Exception as ex:
                    st.warning(f"Bullet optimization service unavailable: {ex}")

            st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: JOB MATCHING (Stand-alone deep explorer)
# -----------------------------------------------------------------------------
elif "Job Matching" in nav_page:
    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Job Matching & Explainable AI (XAI)</h1>
        <p class="page-subtitle">Evaluate semantic fit, inspect keyword skill gaps, and view sentence-level evidence mapping.</p>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.resume_data
    if not data:
        st.warning("Please upload a resume or select a sample on the 'Analyze Resume' page first.")
        st.stop()

    jobs_df = get_jobs_dataframe()
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Target Position Selection")
        job_item = None
        if jobs_df is not None and not jobs_df.empty:
            job_list = [f"{r['job_id']}: {r['title']} @ {r['company']}" for _, r in jobs_df.head(50).iterrows()]
            if job_list:
                sel_job = st.selectbox("Select Harvested Requisition", job_list)
                matching_jobs = jobs_df[jobs_df["job_id"] == sel_job.split(":")[0].strip()]
                if not matching_jobs.empty:
                    job_item = matching_jobs.iloc[0]
            else:
                st.info("No job listings available in dataset.")
        else:
            st.info("Jobs dataset is empty. You can still paste any custom job description on the right.")

    with col2:
        st.subheader("2. Or Paste Custom Job Description")
        custom_desc = st.text_area("Custom Requisition Text", height=100, placeholder="Paste JD requirements here...")

    # Ensure consistent text, skills, and required experience
    if custom_desc.strip():
        target_text = custom_desc.strip()
        # Dynamically extract skills and experience from custom description via API
        try:
            ext_meta = api_client.extract_skills_and_exp(target_text)
            target_req_skills = ext_meta.get("skills", [])
            required_exp_years = float(ext_meta.get("experience_years", 0.0) or 0.0)
        except Exception:
            target_req_skills = []
            required_exp_years = 0.0

        job_title_label = "Custom Job Requisition"
        st.info(
            f"📋 **Custom JD Active**: Extracted {len(target_req_skills)} required skills "
            f"({', '.join(target_req_skills[:6]) if target_req_skills else 'General competencies'}) "
            f"and {required_exp_years:.1f} yrs required experience."
        )
    elif job_item is not None:
        target_text = str(job_item["description"])
        target_req_skills = [s.strip().lower() for s in str(job_item["required_skills"]).replace(";", ",").split(",") if s.strip()]
        required_exp_years = float(job_item.get("experience_required", 0.0) or 0.0)
        job_title_label = f"{job_item['title']} @ {job_item['company']}"
    else:
        st.warning("Please either select a job position or paste a custom job description above.")
        st.stop()

    if st.button("🚀 Calculate Match & Evidence", type="primary", width="stretch"):
        with st.spinner("Executing dual-signal semantic matcher and sentence evidence analysis..."):
            candidate_tenure = float(data.get("experience_years") or 0.0)
            try:
                res = api_client.match_resume_with_job(
                    resume_text=data.get("raw_text", ""),
                    resume_skills=data.get("skills", []),
                    candidate_exp_years=candidate_tenure,
                    jd_text=target_text,
                    jd_skills=target_req_skills if target_req_skills else None,
                    required_exp_years=required_exp_years
                )
                evidence_items = api_client.get_xai_evidence(
                    resume_text=data.get("raw_text", ""),
                    jd_text=target_text,
                    top_n_requirements=5
                )

                st.session_state.active_job_match = res
                st.session_state.active_job_evidence = evidence_items
                st.session_state.active_job_target_title = job_title_label
            except Exception as ex:
                st.error(f"Matching service error: {ex}")

    if "active_job_match" in st.session_state and st.session_state.active_job_match:
        res = st.session_state.active_job_match
        ev_list = st.session_state.get("active_job_evidence", [])
        tgt_label = st.session_state.get("active_job_target_title", "Target Position")

        st.markdown(f"#### Match Results for **{tgt_label}**")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Overall Match", f"{res['match_percentage']}%")
        with m2:
            st.metric("Skill Overlap", f"{int(res['skill_score'] * 100)}%")
        with m3:
            st.metric("Semantic Fit", f"{int(res['semantic_score'] * 100)}%")
        with m4:
            st.metric("Tenure Match", f"{int(res['experience_score'] * 100)}%")

        # Signal distribution chart
        sig_fig = render_match_signals_chart(res['skill_score'], res['semantic_score'], res['experience_score'])
        st.plotly_chart(sig_fig, use_container_width=True, config={"displayModeBar": False})

        # Skill Overlap Chips
        matched_sk = res.get("matched_skills", [])
        missing_sk = res.get("missing_skills", [])
        if matched_sk or missing_sk:
            sk_c1, sk_c2 = st.columns(2)
            with sk_c1:
                st.markdown(f"**Matched Requisition Skills ({len(matched_sk)}):**")
                if matched_sk:
                    m_chips = "".join([f'<span class="badge-chip badge-matched">✓ {html.escape(str(s))}</span>' for s in matched_sk])
                    st.markdown(f"<div>{m_chips}</div>", unsafe_allow_html=True)
                else:
                    st.info("No required skills directly overlapped.")
            with sk_c2:
                st.markdown(f"**Missing Required Skills ({len(missing_sk)}):**")
                if missing_sk:
                    ms_chips = "".join([f'<span class="badge-chip badge-missing">✗ {html.escape(str(s))}</span>' for s in missing_sk])
                    st.markdown(f"<div>{ms_chips}</div>", unsafe_allow_html=True)
                else:
                    st.success("All required skills covered!")

        st.markdown("<br>#### 🔍 Explainable AI (XAI) Sentence-Level Proof", unsafe_allow_html=True)
        for ev in ev_list:
            is_strong = ev.get("status") == "Strong Evidence"
            border = "#10b981" if is_strong else "#3b82f6"
            status_chip = '<span class="status-pill-green">Strong Evidence</span>' if is_strong else '<span class="status-pill-amber">Moderate Inferred</span>'
            escaped_req = html.escape(str(ev.get('job_requirement', '')))
            escaped_ev = html.escape(str(ev.get('resume_evidence', '')))
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {border}; border-radius: 12px; padding: 1.15rem; margin-bottom: 0.85rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                    <span style="font-size: 0.76rem; font-weight: 700; color: #64748b; letter-spacing: 0.03em;">TARGET REQUISITION REQUIREMENT</span>
                    {status_chip}
                </div>
                <div style="font-size: 0.95rem; color: #0f172a; font-weight: 600; margin-bottom: 0.6rem; line-height: 1.4;">"{escaped_req}"</div>
                <div style="font-size: 0.76rem; font-weight: 700; color: #2563eb; letter-spacing: 0.03em; margin-bottom: 0.2rem;">CANDIDATE RESUME EVIDENCE</div>
                <div style="font-size: 0.9rem; color: #334155; line-height: 1.45; background: #f8fafc; padding: 0.55rem 0.75rem; border-radius: 8px;">"{escaped_ev}"</div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 3: BATCH SCREENING (Recruiter Mode)
# -----------------------------------------------------------------------------
elif "Batch Screening" in nav_page:
    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Recruiter Batch Screening</h1>
        <p class="page-subtitle">Screen and rank multiple candidates simultaneously against any target requisition.</p>
    </div>
    """, unsafe_allow_html=True)

    jobs_df = get_jobs_dataframe()
    if jobs_df is None or jobs_df.empty:
        st.warning("No job positions available from the backend API.")
        st.stop()

    job_choices = [f"{r['job_id']}: {r['title']} @ {r['company']}" for _, r in jobs_df.head(30).iterrows()]
    if not job_choices:
        st.warning("No selectable jobs available.")
        st.stop()

    target_pos = st.selectbox("Target Requisition", job_choices)
    matching_jobs = jobs_df[jobs_df["job_id"] == target_pos.split(":")[0].strip()]
    if matching_jobs.empty:
        st.error("Selected job requisition could not be resolved from dataset.")
        st.stop()
    req_row = matching_jobs.iloc[0]

    b_files = st.file_uploader(
        "Upload Multiple Resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="batch_resumes_uploader"
    )

    c_b1, c_b2 = st.columns([1, 1])
    with c_b1:
        if st.button("⚡ Screen Pre-Packaged Candidate Resumes", type="secondary", width="stretch"):
            s_dir = ROOT_DIR / "data" / "sample_resumes"
            loaded_list = []
            for p in s_dir.glob("*.txt"):
                try:
                    loaded_list.append((p.name, p.read_text(encoding="utf-8")))
                except Exception as ex:
                    st.warning(f"Could not load sample file '{p.name}': {ex}")
            st.session_state.batch_resumes = loaded_list

    candidates_to_eval = []
    file_hashes = []

    if b_files:
        for f in b_files:
            file_bytes = f.getvalue()
            # 1. Size validation
            if len(file_bytes) == 0:
                st.warning(f"⚠️ **{f.name}**: File is empty (0 bytes). Skipped.")
                continue
            if len(file_bytes) > 5 * 1024 * 1024:
                st.warning(f"⚠️ **{f.name}**: File size ({len(file_bytes)/1024/1024:.1f} MB) exceeds 5 MB limit. Skipped.")
                continue

            # 2. Extract document safely via backend API upload
            try:
                upload_res = api_client.upload_resume(file_bytes, f.name)
                ext_text = upload_res["extracted_data"]["raw_text"]
                f_hash = hashlib.sha256(file_bytes).hexdigest()
                candidates_to_eval.append((f.name, ext_text, f_hash))
                file_hashes.append(f_hash)
            except APIResponseError as are:
                st.warning(f"⚠️ **{f.name}**: Parsing Error - {are.detail}. Skipped.")
            except Exception as ex:
                st.warning(f"⚠️ **{f.name}**: Extraction failed - {str(ex)}. Skipped.")

    elif "batch_resumes" in st.session_state and st.session_state.batch_resumes:
        for fname, text in st.session_state.batch_resumes:
            f_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            candidates_to_eval.append((fname, text, f_hash))
            file_hashes.append(f_hash)

    if candidates_to_eval:
        st.markdown(f"Screening **{len(candidates_to_eval)} candidate(s)** for **{req_row['title']}**...")

        # Cache key based on target job ID and sorted candidate hashes to avoid recalculating on rerun
        cache_signature = f"{req_row['job_id']}_" + hashlib.sha256(",".join(sorted(file_hashes)).encode("utf-8")).hexdigest()

        if st.session_state.get("last_batch_cache_key") == cache_signature and "last_batch_rows" in st.session_state:
            rows = st.session_state.last_batch_rows
        else:
            t_skills = [s.strip().lower() for s in str(req_row["required_skills"]).replace(";", ",").split(",") if s.strip()]
            req_exp = float(req_row.get("experience_required", 0.0) or 0.0)

            candidates_payload = [{"filename": fname, "text": text} for fname, text, _ in candidates_to_eval]
            batch_response = None

            # Attempt real-time asynchronous background screening with live progress
            try:
                async_job = api_client.batch_screen_async(
                    job_id=str(req_row["job_id"]),
                    jd_title=str(req_row["title"]),
                    jd_text=str(req_row["description"]),
                    required_skills=t_skills,
                    experience_required=req_exp,
                    candidates=candidates_payload
                )
                batch_task_id = async_job.get("job_id")

                progress_container = st.empty()
                progress_bar = progress_container.progress(5, text="Submitting candidate batch to background queue...")

                for _ in range(120):  # poll up to 60s
                    time.sleep(0.3)
                    poll_res = api_client.get_batch_progress(batch_task_id)
                    pct = int(poll_res.get("progress_pct", 0))
                    cur_file = poll_res.get("current_file") or "Processing candidates..."
                    progress_bar.progress(min(100, max(5, pct)), text=f"Screening: {cur_file} ({pct}%)")

                    if poll_res.get("status") == "completed":
                        batch_response = poll_res
                        progress_bar.progress(100, text="✨ Screening complete!")
                        time.sleep(0.4)
                        progress_container.empty()
                        break
                    elif poll_res.get("status") == "failed":
                        progress_container.empty()
                        raise RuntimeError(poll_res.get("error") or "Background batch screening failed")
            except Exception:
                # Fallback to synchronous batch screening
                try:
                    with st.spinner("Executing candidate batch screening..."):
                        batch_response = api_client.batch_screen(
                            job_id=str(req_row["job_id"]),
                            jd_title=str(req_row["title"]),
                            jd_text=str(req_row["description"]),
                            required_skills=t_skills,
                            experience_required=req_exp,
                            candidates=candidates_payload
                        )
                except Exception as ex:
                    st.error(f"Batch screening failed via API: {ex}")
                    batch_response = None

            if batch_response:
                results_list = batch_response.get("results", [])
                ranked_results = sorted(results_list, key=lambda x: x.get("match_percentage", 0), reverse=True)

                medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
                rows = [
                    {
                        "Rank": medals[i] if i < len(medals) else f"#{i+1}",
                        "Candidate": res.get("candidate_name", "Applicant"),
                        "Filename": res.get("filename", ""),
                        "Match Score": f"{res.get('match_percentage', 0)}%",
                        "ATS Score": f"{res.get('ats_score', 0)}/100",
                        "Experience": f"{res.get('experience_years', 0.0):.1f} yrs" if res.get('experience_years') and res.get('experience_years') > 0 else "Not detected",
                        "Skills Matched": res.get("skills_matched_count", 0),
                        "Missing Skills": ", ".join(res.get("missing_skills", [])[:3]) if res.get("missing_skills") else "None"
                    }
                    for i, res in enumerate(ranked_results)
                ]
            else:
                rows = []

            st.session_state.last_batch_cache_key = cache_signature
            st.session_state.last_batch_rows = rows

        if rows:
            st.markdown("#### 🏆 Candidate Leaderboard")
            df_batch = pd.DataFrame(rows)
            st.dataframe(df_batch, width="stretch")

            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            safe_title = "".join(c for c in req_row.get('title', 'Requisition') if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            st.download_button(
                label="📥 Export Leaderboard (CSV)",
                data=csv_data,
                file_name=f"Screening_Leaderboard_{safe_title}.csv",
                mime="text/csv",
                key="export_batch_leaderboard_btn"
            )
        else:
            st.info("No candidates were evaluated in this batch.")


# -----------------------------------------------------------------------------
# PAGE 4: RESUME OPTIMIZER (Google X-Y-Z)
# -----------------------------------------------------------------------------
elif "Resume Optimizer" in nav_page:
    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Resume Optimizer (Google X-Y-Z)</h1>
        <p class="page-subtitle">Audit and rewrite individual resume bullet points following Google's hiring standard.</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Samples for testing
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    preset_bullet = None
    with preset_col1:
        if st.button("📝 Weak / Passive Example", width="stretch"):
            st.session_state.opt_bullet_text = "Helped the team with data tasks and cleaned spreadsheets."
    with preset_col2:
        if st.button("⚖️ Moderate / Vague Example", width="stretch"):
            st.session_state.opt_bullet_text = "Responsible for building machine learning models for user churn prediction."
    with preset_col3:
        if st.button("🚀 Strong Google X-Y-Z", width="stretch"):
            st.session_state.opt_bullet_text = "Reduced customer churn by 18% (saving $340K annually) by engineering real-time gradient boosting models deployed on AWS."

    initial_bullet = st.session_state.get(
        "opt_bullet_text",
        "Responsible for building data pipelines and helping the analytics team clean database tables."
    )

    bullet_input = st.text_area(
        "Candidate Bullet Point",
        value=initial_bullet,
        height=100,
        key="bullet_input_area"
    )

    if st.button("⚡ Audit & Optimize Bullet Point", type="primary", width="stretch"):
        try:
            opt = api_client.optimize_bullet(bullet_input)
            
            # Metric & Grade
            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                st.metric("Impact Score", f"{opt['score']}/100", delta=f"Grade: {opt.get('grade', 'C')}")
            with m_col2:
                # Rubric Checks
                has_action = opt.get("has_action_verb", False)
                has_metric = opt.get("has_quantifiable_metric", False)
                has_context = opt.get("has_context_or_tech", False)
                
                c_action = '<span class="status-pill-green">✓ Strong Action Verb</span>' if has_action else '<span class="status-pill-amber">⚠️ Weak Verb</span>'
                c_metric = '<span class="status-pill-green">✓ Quantifiable Metric</span>' if has_metric else '<span class="status-pill-amber">⚠️ No Metric</span>'
                c_context = '<span class="status-pill-green">✓ Clear Scope / Tech</span>' if has_context else '<span class="status-pill-amber">⚠️ Vague Context</span>'
                
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.85rem; margin-top: 4px;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #64748b; margin-bottom: 0.35rem;">GOOGLE X-Y-Z RUBRIC CHECKS</div>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        {c_action} {c_metric} {c_context}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Diagnostic Feedback")
            for fb in opt.get("feedback", []):
                st.markdown(f"- {fb}")
                
            st.markdown("#### ✨ Suggested High-Impact Rewrite (Google X-Y-Z):")
            rewrite = opt.get("improved_rewrite", "")
            st.success(rewrite)
            st.code(rewrite, language="markdown")
        except Exception as ex:
            st.error(f"Failed to optimize bullet point: {ex}")


# -----------------------------------------------------------------------------
# PAGE 5: JOB RECOMMENDATIONS
# -----------------------------------------------------------------------------
elif "Job Recommendations" in nav_page:
    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Smart Job Recommendations</h1>
        <p class="page-subtitle">Top matching positions ranked across 721 real-world roles.</p>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.resume_data
    if not data:
        st.warning("Please upload a resume on the 'Analyze Resume' page first.")
        st.stop()

    # Filters row
    f_c1, f_c2 = st.columns([2, 1])
    with f_c1:
        search_filter = st.text_input("🔍 Search roles or keywords", placeholder="e.g. Python, Machine Learning, Data Engineer")
    with f_c2:
        top_k = st.selectbox("Max Recommendations", [5, 10, 15, 20], index=1)

    candidate_tenure = float(data.get("experience_years") or 0.0)
    with st.spinner("Retrieving top matching positions via Two-Stage Recommender..."):
        try:
            rec_res = api_client.recommend_jobs(
                resume_text=data.get("raw_text", ""),
                resume_skills=data.get("skills", []),
                candidate_exp_years=candidate_tenure,
                top_n=top_k,
                search_query=search_filter.strip() if search_filter.strip() else None
            )
            recs = rec_res.get("recommendations", [])
        except Exception as ex:
            st.error(f"Failed to fetch recommendations: {ex}")
            recs = []

    if recs:
        st.markdown(f"Displaying **{len(recs)} top matched positions** ranked by semantic fit & skill coverage:")
        for idx, job in enumerate(recs):
            match_pct = job.get("match_percentage", 0)
            if match_pct >= 80:
                m_pill = f'<span class="verdict-pill verdict-green">{match_pct}% Match</span>'
                card_border = "#10b981"
            elif match_pct >= 60:
                m_pill = f'<span class="verdict-pill verdict-blue">{match_pct}% Match</span>'
                card_border = "#2563eb"
            else:
                m_pill = f'<span class="verdict-pill verdict-amber">{match_pct}% Match</span>'
                card_border = "#f59e0b"

            req_sk = [s.strip() for s in str(job.get("required_skills", "")).replace(";", ",").split(",") if s.strip()]
            candidate_sk_set = set(str(s).lower() for s in data.get("skills", []))
            
            matched_here = [s for s in req_sk if s.lower() in candidate_sk_set]
            missing_here = [s for s in req_sk if s.lower() not in candidate_sk_set]

            matched_chips = "".join([f'<span class="badge-chip badge-matched">✓ {html.escape(s)}</span>' for s in matched_here[:5]])
            missing_chips = "".join([f'<span class="badge-chip badge-missing">✗ {html.escape(s)}</span>' for s in missing_here[:3]])

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid {card_border}; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">{html.escape(str(job.get('title')))}</div>
                        <div style="font-size: 0.88rem; color: #64748b; font-weight: 600;">{html.escape(str(job.get('company')))}</div>
                    </div>
                    <div>{m_pill}</div>
                </div>
                <div style="display: flex; gap: 0.85rem; font-size: 0.8rem; color: #475569; margin-bottom: 0.75rem; flex-wrap: wrap;">
                    <span>📍 <b>{html.escape(str(job.get('location', 'Remote')))}</b></span>
                    <span>💰 <b>{html.escape(str(job.get('salary', 'Competitive')))}</b></span>
                    <span>⏳ <b>{job.get('experience_required', 0)} yrs experience</b></span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    {matched_chips} {missing_chips}
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander(f"Inspect Description for {job.get('title')}"):
                st.markdown(job.get("description", "No description available."))
    else:
        st.info("No matching job positions found matching your criteria.")


# -----------------------------------------------------------------------------
# PAGE 6: REPORTS & HISTORY
# -----------------------------------------------------------------------------
elif "Reports" in nav_page:
    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Reports & Analysis History</h1>
        <p class="page-subtitle">Review historical candidate audits stored in the secure audit database.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        records = api_client.get_history(limit=30)
        if records:
            history_data = [
                {
                    "ID": r["id"],
                    "Candidate Name": r.get("candidate_name") or "Applicant",
                    "Filename": r.get("filename") or "resume.pdf",
                    "ATS Score": r.get("ats_score", 0),
                    "Grade": r.get("ats_grade") or "C",
                    "Experience": f"{r.get('experience_years', 0.0):.1f} yrs" if r.get("experience_years") else "Not detected",
                    "Date": r.get("created_at") or "-"
                }
                for r in records
            ]
            st.dataframe(pd.DataFrame(history_data), width="stretch")

            # Record management / deletion
            st.markdown("#### 🗑️ Manage History Records")
            del_c1, del_c2 = st.columns([3, 1])
            with del_c1:
                rec_choices = {f"Record #{r['id']} — {r.get('candidate_name', 'Applicant')} ({r.get('filename')})": r["id"] for r in records}
                chosen_label = st.selectbox("Select Audit Record to Delete", list(rec_choices.keys()), key="delete_record_picker")
            with del_c2:
                st.markdown("<div style='margin-top: 28px;'>", unsafe_allow_html=True)
                if st.button("Delete Record", type="secondary"):
                    target_id = rec_choices[chosen_label]
                    try:
                        api_client.delete_history_record(target_id)
                        st.success(f"Record #{target_id} deleted successfully.")
                        st.rerun()
                    except Exception as dex:
                        st.error(f"Failed to delete record: {dex}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No saved audit records found in audit history. Upload a resume to populate history.")
    except Exception as e:
        st.info(f"Audit history view unavailable: {e}")

