"""
Page 2: Analysis Results
Candidate summary card, 4-card metric row with one-line explanations,
Fix First top-3 prioritized issues card, distinct cards for Strengths, Skills,
and Sections (no nested tabs), What This Score Means popover, and Bullet Optimizer.
"""
from datetime import datetime
import html
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from frontend.api_client import api_client
from frontend.components.empty_state import render_empty_state
from frontend.components.score_ring import render_score_ring
from frontend.components.stat_card import render_stat_card
from frontend.components.skill_chips import render_skill_chips

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def render_ats_category_chart(category_breakdown: dict) -> go.Figure:
    """Renders accessible horizontal bar chart for the 8 ATS categories."""
    if not category_breakdown:
        return None

    categories = []
    percentages = []
    scores = []
    max_scores = []
    colors = []
    hover_texts = []

    for cat_name, info in category_breakdown.items():
        score = info.get("score", 0)
        max_s = info.get("max_score", 10)
        pct = int((score / max_s) * 100) if max_s > 0 else 0
        details = info.get("details", "")

        categories.append(html.escape(cat_name))
        scores.append(score)
        max_scores.append(max_s)
        percentages.append(pct)

        if pct >= 80:
            colors.append("#10b981")  # Emerald
        elif pct >= 60:
            colors.append("#2563eb")  # Blue
        elif pct >= 40:
            colors.append("#f59e0b")  # Amber
        else:
            colors.append("#ef4444")  # Red

        hover_texts.append(f"<b>{cat_name}</b><br>Score: {score}/{max_s} pts ({pct}%)<br>{details}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=percentages,
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255, 255, 255, 0.7)', width=1)),
        text=[f"{p}% ({s}/{m})" for p, s, m in zip(percentages, scores, max_scores)],
        textposition='inside',
        textfont=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=11, color='#ffffff'),
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
            tickfont=dict(size=11, color='#64748b'),
            ticksuffix='%'
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(size=12, color='#0f172a')
        )
    )
    return fig


def show_results_page():
    # If no resume data is in session state, render empty state placeholder
    if not st.session_state.get("resume_data"):
        st.markdown(render_empty_state(
            title="No Active Resume Loaded",
            description="Upload a resume (PDF, DOCX, TXT) or select one of the demo samples from the Upload page to generate an in-depth ATS diagnostic report.",
            action_label="Go to Upload Resume"
        ), unsafe_allow_html=True)
        if st.button("⬅️ Go to Upload Resume", type="primary", key="results_to_upload_btn"):
            st.switch_page("pages/1_home.py")
        return

    data = st.session_state.resume_data
    ats = st.session_state.get("ats_result", {})
    filename = st.session_state.get("active_filename", "resume.pdf")
    is_sample = st.session_state.get("is_sample", False)
    timestamp = st.session_state.get("analysis_timestamp", datetime.now().strftime("%d %b %Y, %I:%M %p"))

    candidate_name = data.get("candidate_name") or "Applicant"
    email = data.get("email") or "Not detected"
    phone = data.get("phone") or "Not detected"
    total_score = ats.get("total_score", 0)
    ats_grade = ats.get("grade", "C")

    # 1. Candidate Summary Card with "Replace resume" action
    st.markdown('<div class="card-container" style="border-left: 5px solid var(--color-primary);">', unsafe_allow_html=True)
    c_col1, c_col2 = st.columns([2.5, 1.0])

    with c_col1:
        status_tag = '<span class="status-pill status-pill-warning">Sample Resume</span>' if is_sample else '<span class="status-pill status-pill-success">Live Analysis</span>'
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            <span style="font-size: var(--font-xl); font-weight: 800; color: var(--color-text);">{html.escape(candidate_name)}</span>
            {status_tag}
        </div>
        <div style="font-size: var(--font-sm); color: var(--color-muted); display: flex; gap: 16px; flex-wrap: wrap;">
            <span>📄 <b>{html.escape(filename)}</b></span>
            <span>✉️ <b>{html.escape(email)}</b></span>
            <span>📞 <b>{html.escape(phone)}</b></span>
            <span>🕒 <b>{html.escape(timestamp)}</b></span>
        </div>
        """, unsafe_allow_html=True)

    with c_col2:
        st.markdown('<div style="text-align: right; padding-top: 4px;">', unsafe_allow_html=True)
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            if st.button("🔄 Replace", key="replace_resume_action", type="secondary", help="Upload a different resume"):
                st.session_state.resume_data = None
                st.session_state.ats_result = None
                st.session_state.active_filename = None
                st.session_state.processed_file_hash = None
                st.switch_page("pages/1_home.py")
        with rep_col2:
            try:
                pdf_bytes = api_client.generate_pdf_report(
                    candidate_name=candidate_name,
                    email=email,
                    phone=phone,
                    ats_score=total_score,
                    ats_grade=ats_grade,
                    category_breakdown=ats.get("category_breakdown", {}),
                    strengths=ats.get("strengths", []),
                    issues=ats.get("issues", []),
                    suggestions=ats.get("suggestions", []),
                    skills=data.get("skills", [])
                )
                st.download_button(
                    label="📥 PDF",
                    data=pdf_bytes,
                    file_name=f"ATS_Audit_{candidate_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    key="download_pdf_report_btn"
                )
            except Exception as e:
                st.caption(f"PDF unavailable: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Four-Card Metric Row with One-Line Explanations
    st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 12px;">Core Resume Metrics</div>', unsafe_allow_html=True)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(render_score_ring(total_score, size=110), unsafe_allow_html=True)

    with m_col2:
        skills_count = len(data.get("skills", []))
        st.markdown(render_stat_card(
            title="Technical Skills",
            value=f"{skills_count}",
            explanation="Industry competencies identified across resume text.",
            icon="🛠️",
            border_color="var(--color-primary)"
        ), unsafe_allow_html=True)

    with m_col3:
        exp_years = data.get("experience_years")
        if exp_years is not None and exp_years > 0.0:
            exp_display = f"{exp_years:.0f} Yrs" if exp_years.is_integer() else f"{exp_years:.1f} Yrs"
            exp_expl = "Calculated from chronological experience date ranges."
        else:
            exp_display = "None"
            exp_expl = "Add start and end dates (e.g. '2021 - Present') to calculate."

        st.markdown(render_stat_card(
            title="Experience Tenure",
            value=exp_display,
            explanation=exp_expl,
            icon="💼",
            border_color="var(--color-bonus)"
        ), unsafe_allow_html=True)

    with m_col4:
        pg_count = st.session_state.get("page_count")
        if pg_count is not None:
            pg_display = f"{pg_count} {'Page' if pg_count == 1 else 'Pages'}"
            pg_expl = "1-2 pages is the recommended standard for ATS parsers."
        else:
            pg_display = "1 Page"
            pg_expl = "Clean single document flow detected."

        st.markdown(render_stat_card(
            title="Document Length",
            value=pg_display,
            explanation=pg_expl,
            icon="📄",
            border_color="var(--color-success)"
        ), unsafe_allow_html=True)

    # "What this score means" popover
    with st.popover("ℹ️ What this score means"):
        st.markdown("""
        ### The 100-Point ATS Rubric
        The ATS Compatibility Score evaluates 8 quantitative dimensions:
        1. **Contact Information (10 pts):** Verified email, valid telephone format, and candidate name.
        2. **Professional Summary (10 pts):** Presence of executive summary or objective.
        3. **Work Experience with Dates (15 pts):** Standard job titles and chronological date ranges.
        4. **Education Credentials (10 pts):** Degree title, institution, and graduation year.
        5. **Technical & Domain Skills (20 pts):** Density of industry-standard competencies.
        6. **Quantifiable Accomplishments (15 pts):** Metrics, dollar values, percentages, and KPIs.
        7. **Action Verb Density (10 pts):** Active verbs starting bullet points (*spearheaded, delivered*).
        8. **ATS Formatting & Length (10 pts):** Clean parseable sections and 1–2 page length.

        **Score Bands:**
        - **80 – 100:** Strong alignment, ready for competitive job applications.
        - **60 – 79:** Competitive foundation, needs targeted metrics and skill expansion.
        - **40 – 59:** Requires structural enhancements and quantifiable bullet rewrites.
        - **0 – 39:** Missing essential sections or contact credentials.
        """)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. "Fix First" Card with Top 3 Prioritized Issues
    issues = ats.get("issues", [])
    st.markdown('<div class="card-container" style="border-left: 5px solid var(--color-warning);">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <div style="font-weight: 800; font-size: var(--font-base); color: var(--color-text);">
            🚨 Fix First: Top Prioritized Recommendations
        </div>
        <span class="status-pill status-pill-warning">High Impact</span>
    </div>
    <p style="font-size: var(--font-sm); color: var(--color-muted); margin-bottom: 12px;">
        Address these 3 high-leverage issues first to produce the biggest jump in recruiter callback rates.
    </p>
    """, unsafe_allow_html=True)

    if issues:
        prioritized = issues[:3]
        badges = [
            ("High Impact", "Low Effort", "var(--color-danger-bg)", "var(--color-danger-text)"),
            ("High Impact", "Medium Effort", "var(--color-warning-bg)", "var(--color-warning-text)"),
            ("Medium Impact", "Low Effort", "var(--color-primary-subtle)", "var(--color-primary)")
        ]
        for idx, issue_text in enumerate(prioritized):
            impact, effort, bg, fg = badges[idx] if idx < len(badges) else ("Medium Impact", "Medium Effort", "#f1f5f9", "#475569")
            st.markdown(f"""
            <div style="background-color: var(--color-background); border: 1px solid var(--color-border); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: var(--font-sm); font-weight: 600; color: var(--color-text);">
                    <span style="color: var(--color-danger); margin-right: 6px;">●</span> {html.escape(issue_text)}
                </div>
                <div style="display: flex; gap: 6px;">
                    <span style="font-size: var(--font-xs); font-weight: 700; background: {bg}; color: {fg}; padding: 2px 8px; border-radius: 4px;">{impact}</span>
                    <span style="font-size: var(--font-xs); font-weight: 600; background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 4px;">{effort}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size: var(--font-sm); color: var(--color-success-text); font-weight: 600;">✅ Excellent! No critical issues detected on this resume.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Distinct Cards for Strengths, Skills, and Sections (NO nested tabs!)
    c_str, c_cat = st.columns([1, 1], gap="medium")

    with c_str:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">💪 Verified Resume Strengths</div>', unsafe_allow_html=True)
        strengths = ats.get("strengths", [])
        if strengths:
            for s in strengths:
                st.markdown(f"""
                <div style="font-size: var(--font-sm); color: var(--color-text-secondary); margin-bottom: 6px; display: flex; align-items: flex-start; gap: 8px;">
                    <span style="color: var(--color-success); font-weight: bold;">✓</span>
                    <span>{html.escape(s)}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No standout strengths detected yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_cat:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">📊 8-Dimension Rubric Breakdown</div>', unsafe_allow_html=True)
        cat_fig = render_ats_category_chart(ats.get("category_breakdown", {}))
        if cat_fig:
            st.plotly_chart(cat_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Skills Card
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">🛠️ Identified Technical Competencies ({len(data.get("skills", []))} Found)</div>', unsafe_allow_html=True)
    skills = data.get("skills", [])
    if skills:
        st.markdown(render_skill_chips(matched=skills), unsafe_allow_html=True)
    else:
        st.caption("No standard technical skills were detected. Ensure skill keywords are explicitly listed.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Document Sections Card
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">📑 Detected Document Sections</div>', unsafe_allow_html=True)
    sections = data.get("sections", {})
    if sections:
        sec_cols = st.columns(min(4, len(sections)))
        for idx, (sec_name, sec_content) in enumerate(sections.items()):
            col_target = sec_cols[idx % len(sec_cols)]
            with col_target:
                word_count = len(str(sec_content).split())
                st.markdown(f"""
                <div style="background-color: var(--color-background); border: 1px solid var(--color-border); border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <div style="font-weight: 700; font-size: var(--font-sm); color: var(--color-primary);">{html.escape(sec_name.title())}</div>
                    <div style="font-size: var(--font-xs); color: var(--color-muted);">{word_count} words detected</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("No standard section headers detected. Standard headers include: Experience, Education, Skills, Summary.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. Bullet Optimizer Card (Google X-Y-Z Standard)
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-weight: 800; font-size: var(--font-base); color: var(--color-text); margin-bottom: 4px;">
        🪄 Google X-Y-Z Bullet Point Optimizer
    </div>
    <p style="font-size: var(--font-sm); color: var(--color-muted); margin-bottom: 12px;">
        Rewrite passive resume bullets into the executive formula: <i>"Accomplished [X] as measured by [Y], by doing [Z]"</i>.
    </p>
    """, unsafe_allow_html=True)

    default_bullet = "Developed software applications and cleaned database tables for the data science team."
    sample_text = st.text_area(
        "Enter or Paste a Resume Bullet Point to Optimize",
        value=default_bullet,
        height=80,
        key="results_bullet_opt_input"
    )

    if st.button("⚡ Audit & Rewrite Bullet", type="secondary", key="run_bullet_opt_results"):
        with st.spinner("Analyzing verb strength, quantifiable impact, and context..."):
            try:
                opt_res = api_client.optimize_bullet(sample_text)
                st.markdown(f"""
                <div style="background-color: var(--color-primary-subtle); border: 1px solid var(--color-border); border-radius: 8px; padding: 12px; margin-top: 10px;">
                    <div style="font-size: var(--font-xs); font-weight: 700; color: var(--color-primary); margin-bottom: 4px;">RECOMMENDED REWRITE</div>
                    <div style="font-size: var(--font-sm); font-weight: 600; color: var(--color-text); margin-bottom: 8px;">{html.escape(opt_res.get('improved_rewrite', ''))}</div>
                    <div style="font-size: var(--font-xs); color: var(--color-muted);">Impact Score: <b>{opt_res.get('score', 0)}/100</b> ({opt_res.get('grade', 'C')})</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Bullet optimizer service offline: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


show_results_page()
