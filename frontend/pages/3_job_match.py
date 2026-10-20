"""
Page 3: Job Match
2 columns (Resume summary left, Target JD selector/input right),
match gauge + verdict, matched/missing/bonus skill chips,
prominent sentence-level evidence mapping cards, and "Improve for this job" action.
"""
import html
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from frontend.api_client import api_client
from frontend.components.empty_state import render_empty_state
from frontend.components.skill_chips import render_skill_chips
from frontend.components.evidence_card import render_evidence_card
from frontend.components.score_ring import render_score_ring

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


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


def render_match_signals_chart(skill_score: float, semantic_score: float, exp_score: float) -> go.Figure:
    """Grouped bar visualizer showing individual signal contributions."""
    signals = ["Skill Overlap (50%)", "Semantic Vector (40%)", "Tenure Alignment (10%)"]
    vals = [round(skill_score * 100, 1), round(semantic_score * 100, 1), round(exp_score * 100, 1)]
    colors = ["#10b981", "#2563eb", "#8b5cf6"]

    fig = go.Figure(go.Bar(
        x=signals,
        y=vals,
        marker=dict(color=colors),
        text=[f"{v}%" for v in vals],
        textposition='auto',
        textfont=dict(size=12, color='#ffffff')
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=15, b=10),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor='#f1f5f9', ticksuffix='%')
    )
    return fig


def show_job_match_page():
    if not st.session_state.get("resume_data"):
        st.markdown(render_empty_state(
            title="No Resume Active for Matching",
            description="Please upload a resume or select a sample resume from the Upload page first to compare against real target job descriptions.",
            action_label="Go to Upload Resume"
        ), unsafe_allow_html=True)
        if st.button("⬅️ Go to Upload Resume", type="primary", key="match_to_upload_btn"):
            st.switch_page("pages/1_home.py")
        return

    data = st.session_state.resume_data
    candidate_name = data.get("candidate_name") or "Applicant"
    candidate_skills = data.get("skills", [])
    candidate_exp = float(data.get("experience_years") or 0.0)

    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Dual-Signal Job Matching & Explainable AI</h1>
        <p class="page-subtitle">Compare candidate resume against target job requirements with sentence-level proof and skill gap analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # 2 Columns: Resume Left, Target JD Right
    col_resume, col_jd = st.columns([1.0, 1.2], gap="large")

    with col_resume:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 6px;">📄 Active Candidate</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size: var(--font-lg); font-weight: 800; color: var(--color-text);">{html.escape(candidate_name)}</div>
            <div style="font-size: var(--font-xs); color: var(--color-muted); margin-bottom: 12px;">{html.escape(st.session_state.get('active_filename', 'resume.pdf'))}</div>
            <div style="display: flex; gap: 12px; margin-bottom: 12px; font-size: var(--font-sm); color: var(--color-text-secondary);">
                <span>💼 <b>{candidate_exp:.1f} Yrs Experience</b></span>
                <span>🛠️ <b>{len(candidate_skills)} Skills Detected</b></span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="font-weight: 600; font-size: var(--font-xs); color: var(--color-muted); margin-bottom: 6px;">IDENTIFIED SKILLS</div>', unsafe_allow_html=True)
            if candidate_skills:
                st.markdown(render_skill_chips(matched=candidate_skills[:16]), unsafe_allow_html=True)
            else:
                st.caption("No technical skills detected.")

    with col_jd:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">🎯 Target Job Description</div>', unsafe_allow_html=True)

            jobs_df = get_jobs_dataframe()
            selected_job_row = None

            if not jobs_df.empty:
                job_titles = [f"{r['job_id']}: {r['title']} @ {r['company']}" for _, r in jobs_df.head(40).iterrows()]
                chosen_job_str = st.selectbox("Select Pre-loaded Job Requisition", job_titles, key="match_job_selector")
                matched_rows = jobs_df[jobs_df["job_id"] == chosen_job_str.split(":")[0].strip()]
                if not matched_rows.empty:
                    selected_job_row = matched_rows.iloc[0]

            custom_jd_input = st.text_area(
                "Or Paste Custom Job Description",
                height=120,
                placeholder="Paste raw requisition or job post text here...",
                key="match_custom_jd_input"
            )

    # Determine JD text and required skills
    if custom_jd_input.strip():
        jd_text = custom_jd_input.strip()
        try:
            extracted_meta = api_client.extract_skills_and_exp(jd_text)
            jd_skills = extracted_meta.get("skills", [])
            jd_exp = float(extracted_meta.get("experience_years", 0.0) or 0.0)
        except Exception:
            jd_skills = []
            jd_exp = 0.0
        job_label = "Custom Job Description"
    elif selected_job_row is not None:
        jd_text = str(selected_job_row["description"])
        jd_skills = [s.strip().lower() for s in str(selected_job_row.get("required_skills", "")).replace(";", ",").split(",") if s.strip()]
        jd_exp = float(selected_job_row.get("experience_required", 0.0) or 0.0)
        job_label = f"{selected_job_row['title']} @ {selected_job_row['company']}"
    else:
        st.info("Select or paste a target job description above to begin matching.")
        return

    # Matching Action Button
    if st.button("🚀 Calculate Match & Evidence", type="primary", key="calc_match_btn", width="stretch"):
        with st.spinner("Executing dual-signal hybrid matcher and sentence evidence analysis..."):
            try:
                match_res = api_client.match_resume_with_job(
                    resume_text=data.get("raw_text", ""),
                    resume_skills=candidate_skills,
                    candidate_exp_years=candidate_exp,
                    jd_text=jd_text,
                    jd_skills=jd_skills if jd_skills else None,
                    required_exp_years=jd_exp
                )
                evidence = api_client.get_xai_evidence(
                    resume_text=data.get("raw_text", ""),
                    jd_text=jd_text,
                    top_n_requirements=4
                )
                st.session_state.active_job_match = match_res
                st.session_state.active_job_evidence = evidence
                st.session_state.active_job_label = job_label
                st.session_state.active_job_desc = jd_text
            except Exception as e:
                st.error(f"Job matching failed: {e}")

    # Render Match Results if available
    if st.session_state.get("active_job_match"):
        res = st.session_state.active_job_match
        ev_items = st.session_state.get("active_job_evidence", [])
        j_label = st.session_state.get("active_job_label", "Selected Requisition")

        with st.container(border=True):
            st.markdown(f'<div style="font-weight: 800; font-size: var(--font-lg); color: var(--color-text); margin-bottom: 12px;">Match Evaluation for {html.escape(j_label)}</div>', unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1.0, 1.8], gap="medium")
            with res_col1:
                st.markdown(render_score_ring(res.get("match_percentage", 0), size=130), unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 6px;">
                    Alignment Verdict: <span style="color: var(--color-primary);">{res.get('verdict', 'Evaluated')}</span>
                </div>
                <div style="font-size: var(--font-sm); color: var(--color-muted); margin-bottom: 12px;">
                    Weighted combination of Skill Overlap (50%), MiniLM Semantic Fit (40%), and Experience Alignment (10%).
                </div>
                """, unsafe_allow_html=True)
                sig_fig = render_match_signals_chart(
                    res.get("skill_score", 0.0),
                    res.get("semantic_score", 0.0),
                    res.get("experience_score", 0.0)
                )
                st.plotly_chart(sig_fig, use_container_width=True, config={"displayModeBar": False})

        # Skill Overlap Chips
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">🎯 Skill Alignment Breakdown</div>', unsafe_allow_html=True)

            matched_skills = res.get("matched_skills", [])
            missing_skills = res.get("missing_skills", [])
            candidate_lower = set(s.lower() for s in candidate_skills)
            target_lower = set(s.lower() for s in (jd_skills or []))
            bonus_skills = [s for s in candidate_skills if s.lower() not in target_lower][:6]

            st.markdown(render_skill_chips(
                matched=matched_skills,
                missing=missing_skills,
                bonus=bonus_skills
            ), unsafe_allow_html=True)

        # Sentence-Level Evidence Mapping Cards
        with st.container(border=True):
            st.markdown("""
            <div style="font-weight: 800; font-size: var(--font-base); color: var(--color-text); margin-bottom: 4px;">
                🔍 Explainable AI (XAI) Sentence-Level Evidence Mapping
            </div>
            <p style="font-size: var(--font-sm); color: var(--color-muted); margin-bottom: 12px;">
                Every target job requirement is paired with the exact sentence from the candidate's resume that serves as qualifying proof.
            </p>
            """, unsafe_allow_html=True)

            if ev_items:
                for ev in ev_items:
                    st.markdown(render_evidence_card(
                        requirement=ev.get("job_requirement", ""),
                        candidate_proof=ev.get("resume_evidence", ""),
                        status=ev.get("status", "Strong Evidence")
                    ), unsafe_allow_html=True)
            else:
                st.info("No sentence-level proof cards generated. Ensure the resume has detailed project and experience descriptions.")

            # "Improve for this job" button
            st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: 16px 0;'>", unsafe_allow_html=True)
            imp_col1, imp_col2 = st.columns([2.5, 1.0])
            with imp_col1:
                st.markdown("""
                <div style="font-weight: 700; font-size: var(--font-sm); color: var(--color-text);">
                    Targeted Bullet Point Optimizer
                </div>
                <div style="font-size: var(--font-xs); color: var(--color-muted);">
                    Tailor weak bullets to address missing skills for this target requisition.
                </div>
                """, unsafe_allow_html=True)
            with imp_col2:
                if st.button("✨ Improve For This Job", type="primary", key="improve_for_job_btn"):
                    st.toast("Pre-filling optimizer with missing target skills...", icon="🎯")
                    if missing_skills:
                        first_missing = missing_skills[0].title()
                        st.session_state.opt_bullet_text = f"Built and maintained core production services utilizing {first_missing}, improving system throughput by 25%."
                    else:
                        st.session_state.opt_bullet_text = "Delivered critical software systems aligned with company objectives."
                    st.switch_page("pages/2_results.py")


show_job_match_page()
