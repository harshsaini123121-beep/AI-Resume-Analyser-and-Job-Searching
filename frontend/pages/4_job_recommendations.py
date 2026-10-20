"""
Page 4: Job Recommendations
Recommendation cards with Title, Company, Match %, Top 3 Matched / Missing skills,
"Why this job" expander, location/experience/salary filters, and sort controls.
"""
import html
from pathlib import Path

import streamlit as st

from frontend.api_client import api_client
from frontend.components.empty_state import render_empty_state
from frontend.components.skill_chips import render_skill_chips

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def show_job_recommendations_page():
    if not st.session_state.get("resume_data"):
        st.markdown(render_empty_state(
            title="No Active Resume for Recommendations",
            description="Upload a resume or select a sample resume from the Upload page first to rank real-world job openings.",
            action_label="Go to Upload Resume"
        ), unsafe_allow_html=True)
        if st.button("⬅️ Go to Upload Resume", type="primary", key="recs_to_upload_btn"):
            st.switch_page("pages/1_home.py")
        return

    data = st.session_state.resume_data
    candidate_name = data.get("candidate_name") or "Applicant"
    candidate_skills = data.get("skills", [])
    candidate_exp = float(data.get("experience_years") or 0.0)

    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Personalized Job Recommendations</h1>
        <p class="page-subtitle">Ranked job opportunities retrieved from 721 tech positions via semantic vector similarity and skill overlap.</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter and Sort Controls Bar
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight: 700; font-size: var(--font-sm); color: var(--color-text); margin-bottom: 8px;">Filter & Sort Opportunities</div>', unsafe_allow_html=True)

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.0, 1.0, 1.0])

    with f_col1:
        search_kw = st.text_input("🔍 Keyword Search", placeholder="e.g. Python, Machine Learning, AWS", key="rec_search_kw")
    with f_col2:
        loc_filter = st.selectbox("Location", ["All Locations", "Remote", "Bangalore", "San Francisco", "New York", "London"], key="rec_loc_filter")
    with f_col3:
        max_recs = st.selectbox("Show Results", [5, 10, 15, 20], index=1, key="rec_max_results")
    with f_col4:
        sort_by = st.selectbox("Sort Order", ["Highest Match %", "Lowest Experience Req"], key="rec_sort_order")

    st.markdown('</div>', unsafe_allow_html=True)

    # Fetch Recommendations via API
    with st.spinner("Finding best match positions from dataset..."):
        try:
            query = search_kw.strip() if search_kw.strip() else None
            loc_arg = loc_filter if loc_filter != "All Locations" else None
            rec_res = api_client.recommend_jobs(
                resume_text=data.get("raw_text", ""),
                resume_skills=candidate_skills,
                candidate_exp_years=candidate_exp,
                top_n=max_recs,
                location_filter=loc_arg,
                search_query=query
            )
            recs = rec_res.get("recommendations", [])
        except Exception as e:
            st.error(f"Recommendation service offline: {e}")
            recs = []

    # Apply secondary sort if specified
    if sort_by == "Lowest Experience Req" and recs:
        recs = sorted(recs, key=lambda x: float(x.get("experience_required", 0.0) or 0.0))

    if not recs:
        st.info("No matching positions found. Try expanding your search keywords or location filters.")
        return

    st.markdown(f'<div style="font-size: var(--font-sm); color: var(--color-muted); margin-bottom: 12px;">Displaying <b>{len(recs)} target positions</b> sorted for <b>{html.escape(candidate_name)}</b>:</div>', unsafe_allow_html=True)

    candidate_lower = set(s.lower() for s in candidate_skills)

    for job in recs:
        match_pct = job.get("match_percentage", 0)
        title = job.get("title", "Position")
        company = job.get("company", "Company")
        location = job.get("location", "Remote")
        salary = job.get("salary", "Competitive")
        exp_req = job.get("experience_required", 0)
        desc = job.get("description", "No detailed description provided.")

        # Status badge styling based on match score
        if match_pct >= 80:
            border_c = "var(--color-success)"
            pill = f'<span class="status-pill status-pill-success">{match_pct}% Match</span>'
        elif match_pct >= 60:
            border_c = "var(--color-primary)"
            pill = f'<span class="status-pill status-pill-info">{match_pct}% Match</span>'
        else:
            border_c = "var(--color-warning)"
            pill = f'<span class="status-pill status-pill-warning">{match_pct}% Match</span>'

        # Skill overlap extraction
        req_skills_raw = str(job.get("required_skills", "")).replace(";", ",").split(",")
        req_skills = [s.strip() for s in req_skills_raw if s.strip()]

        matched_skills = [s for s in req_skills if s.lower() in candidate_lower][:3]
        missing_skills = [s for s in req_skills if s.lower() not in candidate_lower][:3]

        st.markdown(f"""
        <div class="card-container" style="border-left: 5px solid {border_c}; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                <div>
                    <div style="font-size: var(--font-lg); font-weight: 800; color: var(--color-text);">{html.escape(title)}</div>
                    <div style="font-size: var(--font-sm); font-weight: 600; color: var(--color-muted);">{html.escape(company)}</div>
                </div>
                <div>{pill}</div>
            </div>
            <div style="display: flex; gap: 16px; font-size: var(--font-xs); color: var(--color-text-secondary); margin-bottom: 10px; flex-wrap: wrap;">
                <span>📍 <b>{html.escape(location)}</b></span>
                <span>💰 <b>{html.escape(salary)}</b></span>
                <span>⏳ <b>{exp_req} yrs required</b></span>
            </div>
        """, unsafe_allow_html=True)

        if matched_skills or missing_skills:
            st.markdown(render_skill_chips(matched=matched_skills, missing=missing_skills), unsafe_allow_html=True)

        with st.expander(f"💡 Why this job? (Role Overview & Evidence)"):
            st.markdown(f"""
            **Matching Rationale:**
            - You match **{len(matched_skills)}** of the core technical requirements ({', '.join(matched_skills) if matched_skills else 'General competencies'}).
            - Experience tenure: Candidate has **{candidate_exp:.1f} years** vs **{exp_req} years** required.
            
            **Job Description Summary:**
            {desc}
            """)

        st.markdown('</div>', unsafe_allow_html=True)


show_job_recommendations_page()
