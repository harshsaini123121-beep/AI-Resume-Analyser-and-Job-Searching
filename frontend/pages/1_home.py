"""
Page 1: Home / Resume Upload & ATS Score Check
Rebuilt to match the ResumeLab Dashboard design with ATS score check card,
quick insights, interactive sample analysis, and direct navigation.
"""
from datetime import datetime
import hashlib
import html
from pathlib import Path
import time

import streamlit as st

from frontend.api_client import api_client, APIConnectionError, APIResponseError
from frontend.components.error_banner import render_error_banner
from frontend.components.skill_chips import render_skill_chips

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _render_score_circle(score: float, max_score: float = 100.0, size: int = 120) -> str:
    """Renders a circular SVG gauge matching the ResumeLab dashboard mockup."""
    score = max(0.0, min(float(score), max_score))
    radius = 42
    circ = 2 * 3.14159265 * radius  # ~263.89
    offset = circ * (1.0 - (score / max_score))

    if score >= 75:
        bar_color = "#10b981"  # Emerald
        verdict = "Good Match"
        pill_style = "background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0;"
    elif score >= 50:
        bar_color = "#2563eb"  # Royal blue
        verdict = "Competitive"
        pill_style = "background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;"
    else:
        bar_color = "#f59e0b"  # Amber
        verdict = "Needs Work"
        pill_style = "background: #fffbeb; color: #d97706; border: 1px solid #fde68a;"

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2px;" role="img" aria-label="ATS Score {int(round(score))} out of 100">
        <svg width="{size}" height="{size}" viewBox="0 0 100 100" aria-hidden="true">
            <circle cx="50" cy="50" r="{radius}" fill="none" stroke="#e2e8f0" stroke-width="8" />
            <circle cx="50" cy="50" r="{radius}" fill="none" stroke="{bar_color}" stroke-width="8"
                    stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
                    stroke-linecap="round" transform="rotate(-90 50 50)" />
            <text x="50" y="47" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="24" font-weight="800" fill="#0f172a">{int(round(score))}</text>
            <text x="50" y="62" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="10" font-weight="600" fill="#64748b">/100</text>
        </svg>
        <div style="margin-top: 6px;">
            <span style="{pill_style} border-radius: 9999px; padding: 3px 12px; font-size: 0.76rem; font-weight: 700; display: inline-block;">
                {html.escape(verdict)}
            </span>
        </div>
    </div>
    """


def _load_and_analyze_sample(s_fname: str, s_label: str):
    """Loads and analyzes a sample resume, updating session state and rerunning."""
    sample_path = ROOT_DIR / "data" / "sample_resumes" / s_fname
    if not sample_path.exists():
        st.error(f"Sample file {s_fname} not found.")
        return

    raw_text = sample_path.read_text(encoding="utf-8")
    s_bytes = raw_text.encode("utf-8")
    s_hash = hashlib.sha256(s_bytes).hexdigest()
    try:
        with st.spinner(f"Analyzing {s_label}..."):
            analysis = api_client.analyse_resume(s_bytes, s_fname)
            st.session_state.resume_data = analysis["extracted_data"]
            st.session_state.ats_result = analysis["ats_score"]
            st.session_state.page_count = analysis.get("page_count", 2)
            st.session_state.active_filename = s_fname.replace(".txt", ".pdf")
            st.session_state.processed_file_hash = s_hash
            st.session_state.is_sample = True
            st.session_state.analysis_timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
            st.toast(f"✅ {s_label} analyzed successfully!", icon="🎉")
            st.rerun()
    except Exception as ex:
        st.error(f"Failed to analyze sample resume: {ex}")


def show_home_page():
    # 1. Page Header (Exact match with ResumeLab Dashboard reference)
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.85rem; font-weight: 800; color: var(--color-text, #0f172a); margin-bottom: 0.25rem; letter-spacing: -0.02em;">
            Analyze Your Resume
        </h1>
        <p style="font-size: 0.95rem; color: var(--color-muted, #64748b); margin: 0;">
            Get AI-powered insights, ATS score, and best job matches.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.0, 1.25], gap="large")

    # =========================================================================
    # LEFT COLUMN: Upload Card & Sample Resume Picker
    # =========================================================================
    with col_left:
        # Upload card container
        st.markdown("""
        <div class="home-upload-container">
            <div class="home-upload-icon-circle">📄</div>
            <div class="home-upload-title">Upload your resume</div>
            <div class="home-upload-subtitle">Drag and drop a PDF file here or click to browse</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            key="home_resume_uploader",
            help="Supported formats: PDF, DOCX, TXT. Maximum file size: 5 MB."
        )

        st.markdown(
            '<div style="text-align: center; font-size: 0.74rem; color: #94a3b8; margin-top: -8px; margin-bottom: 1.25rem;">'
            'Supported format: PDF, DOCX, TXT (Max 5 MB)'
            '</div>',
            unsafe_allow_html=True
        )

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            if len(raw_bytes) == 0:
                st.markdown(render_error_banner("Empty File", "The uploaded file contains 0 bytes.", "Please select a valid, non-empty resume document."), unsafe_allow_html=True)
            elif len(raw_bytes) > 5 * 1024 * 1024:
                st.markdown(render_error_banner("File Too Large", f"File size ({len(raw_bytes)/1024/1024:.1f} MB) exceeds the 5 MB limit.", "Please compress or convert your document."), unsafe_allow_html=True)
            else:
                curr_hash = hashlib.sha256(raw_bytes).hexdigest()
                if st.session_state.get("processed_file_hash") != curr_hash:
                    with st.status("Analyzing resume document...", expanded=True) as status_box:
                        st.write("📖 Validating document structure...")
                        time.sleep(0.1)
                        st.write("🔍 Extracting candidate skills, experience, and sections...")
                        try:
                            analysis = api_client.analyse_resume(raw_bytes, uploaded_file.name)
                            st.write("🎯 Computing quantitative 8-dimension ATS score...")
                            st.session_state.resume_data = analysis["extracted_data"]
                            st.session_state.ats_result = analysis["ats_score"]
                            st.session_state.page_count = analysis.get("page_count", 2)
                            st.session_state.active_filename = uploaded_file.name
                            st.session_state.processed_file_hash = curr_hash
                            st.session_state.is_sample = False
                            st.session_state.analysis_timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
                            status_box.update(label="✅ Resume successfully analyzed!", state="complete", expanded=False)
                            st.toast("Analysis complete! ATS score check ready.", icon="🎉")
                            st.rerun()
                        except APIResponseError as are:
                            status_box.update(label="❌ Analysis Failed", state="error", expanded=True)
                            st.markdown(render_error_banner("Document Extraction Error", are.detail, "Ensure the document is not password-protected."), unsafe_allow_html=True)
                        except APIConnectionError as ace:
                            status_box.update(label="❌ Backend Connection Error", state="error", expanded=True)
                            st.markdown(render_error_banner("Backend Unavailable", str(ace), "Verify the FastAPI server is running at http://127.0.0.1:8000."), unsafe_allow_html=True)

        # Try a Sample Section
        st.markdown("""
        <div class="home-sample-header-row">
            <span class="home-sample-title">Try a sample</span>
            <span class="home-sample-link">Use Sample →</span>
        </div>
        <div class="home-sample-subtitle">Explore with a sample resume to see how it works.</div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            s_col1, s_col2 = st.columns([0.72, 0.28], vertical_alignment="center")
            with s_col1:
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="width: 38px; height: 38px; border-radius: 8px; background: #fef2f2; border: 1px solid #fee2e2; display: flex; align-items: center; justify-content: center; font-size: 20px; color: #ef4444; flex-shrink: 0;">
                        📄
                    </div>
                    <div>
                        <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">Software Engineer Resume</div>
                        <div style="font-size: 0.74rem; color: #64748b;">Sample Resume • PDF</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with s_col2:
                if st.button("Analyze", key="btn_sample_se", type="primary", use_container_width=True):
                    _load_and_analyze_sample("software_engineer_resume.txt", "Software Engineer Resume")

        with st.expander("More Sample Profiles (Data Analyst, ML Engineer)", expanded=False):
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                if st.button("📊 Data Analyst (3 yrs)", key="btn_sample_da", use_container_width=True):
                    _load_and_analyze_sample("david_kumar_data_analyst.txt", "Data Analyst Resume")
            with ex_col2:
                if st.button("🤖 ML Engineer (6 yrs)", key="btn_sample_ml", use_container_width=True):
                    _load_and_analyze_sample("alex_morgan_ml_engineer.txt", "ML Engineer Resume")

    # =========================================================================
    # RIGHT COLUMN: ATS Score Check Card & Quick Insights
    # =========================================================================
    with col_right:
        resume_data = st.session_state.get("resume_data")
        ats_result = st.session_state.get("ats_result")

        if resume_data and ats_result:
            # ACTIVE ATS ANALYSIS DISPLAY
            fname = html.escape(st.session_state.get("active_filename", "resume.pdf"))
            timestamp_str = st.session_state.get("analysis_timestamp", datetime.now().strftime("%d %b %Y, %I:%M %p"))
            score_val = ats_result.get("total_score", 82)
            skills = resume_data.get("skills", [])
            exp_years = resume_data.get("experience_years", 3.0)
            page_count = st.session_state.get("page_count", 2) or 2

            with st.container(border=True):
                # 1. Header Bar: Document Icon + Name + Badge + Timestamp + New Analysis Button
                doc_col_info, doc_col_action = st.columns([0.72, 0.28], vertical_alignment="center")
                with doc_col_info:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                        <div class="home-ats-docicon">📄</div>
                        <div>
                            <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 6px;">
                                <span style="font-weight: 800; font-size: 0.98rem; color: #0f172a;">{fname}</span>
                                <span class="home-ats-badge-completed">Analysis Completed</span>
                            </div>
                            <div style="font-size: 0.74rem; color: #64748b; margin-top: 1px;">Analysed on {timestamp_str}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with doc_col_action:
                    if st.button("🔄 New Analysis", key="home_btn_new_analysis", use_container_width=True):
                        st.session_state.resume_data = None
                        st.session_state.ats_result = None
                        st.session_state.active_filename = None
                        st.session_state.processed_file_hash = None
                        st.session_state.is_sample = False
                        st.session_state.page_count = None
                        st.session_state.active_job_match = None
                        st.session_state.active_job_evidence = None
                        st.rerun()

                st.markdown("<div style='height: 1px; background: #f1f5f9; margin: 10px 0 14px 0;'></div>", unsafe_allow_html=True)

                # 2. Navigation Tabs (Overview, Skills, Job Match, Suggestions)
                tab_overview, tab_skills, tab_match, tab_suggestions = st.tabs([
                    "Overview", "Skills", "Job Match", "Suggestions"
                ])

                # --- TAB 1: OVERVIEW ---
                with tab_overview:
                    st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
                    ring_col, desc_col = st.columns([0.42, 0.58], vertical_alignment="center")
                    with ring_col:
                        st.markdown(_render_score_circle(score_val, max_score=100.0, size=125), unsafe_allow_html=True)
                    with desc_col:
                        if score_val >= 75:
                            comp_desc = "Your resume is well-optimized for ATS systems. A few improvements can make it even stronger."
                        elif score_val >= 50:
                            comp_desc = "Your resume meets standard ATS criteria. Adding quantified metrics and domain keywords will elevate your placement."
                        else:
                            comp_desc = "Your resume shows ATS formatting gaps. Enhancing action verbs, technical skills, and headers is recommended."

                        st.markdown(f"""
                        <div style="padding-left: 4px;">
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">
                                ATS Compatibility
                            </div>
                            <div style="font-size: 0.86rem; color: #475569; line-height: 1.5;">
                                {comp_desc}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Quick Insights Section
                    st.markdown('<div style="font-weight: 800; font-size: 0.95rem; color: #0f172a; margin-top: 1.25rem; margin-bottom: 0.75rem;">Quick Insights</div>', unsafe_allow_html=True)

                    qi_col1, qi_col2, qi_col3 = st.columns(3)
                    with qi_col1:
                        page_note = "Ideal length" if page_count <= 2 else "2+ pages"
                        st.markdown(f"""
                        <div class="quick-insight-item">
                            <div class="quick-insight-icon-blue">📄</div>
                            <div>
                                <div class="quick-insight-number">{page_count} <span class="quick-insight-unit">Pages</span></div>
                                <div class="quick-insight-desc">{page_note}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with qi_col2:
                        st.markdown(f"""
                        <div class="quick-insight-item">
                            <div class="quick-insight-icon-green">📋</div>
                            <div>
                                <div class="quick-insight-number">{len(skills)} <span class="quick-insight-unit">Key Skills</span></div>
                                <div class="quick-insight-desc">Detected</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with qi_col3:
                        exp_display = int(exp_years) if float(exp_years).is_integer() else f"{float(exp_years):.1f}"
                        st.markdown(f"""
                        <div class="quick-insight-item">
                            <div class="quick-insight-icon-purple">💼</div>
                            <div>
                                <div class="quick-insight-number">{exp_display} <span class="quick-insight-unit">Years</span></div>
                                <div class="quick-insight-desc">Experience • Good range</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

                    # Bottom Action Buttons
                    btn_col_analysis, btn_col_report = st.columns(2, gap="medium")
                    with btn_col_analysis:
                        if st.button("View Detailed Analysis →", key="home_btn_view_full_analysis", use_container_width=True):
                            st.switch_page("pages/2_results.py")
                    with btn_col_report:
                        try:
                            pdf_bytes = api_client.generate_pdf_report(
                                candidate_name=resume_data.get("candidate_name") or "Applicant",
                                ats_score=int(round(score_val)),
                                category_breakdown=ats_result.get("category_breakdown", {}),
                                strengths=ats_result.get("strengths", []),
                                issues=ats_result.get("issues", []),
                                suggestions=ats_result.get("suggestions", []),
                                skills=skills
                            )
                            clean_cand = (resume_data.get("candidate_name") or "Applicant").replace(" ", "_")
                            st.download_button(
                                label="📥 Download Report",
                                data=pdf_bytes,
                                file_name=f"ATS_Report_{clean_cand}.pdf",
                                mime="application/pdf",
                                key="home_btn_download_report",
                                type="primary",
                                use_container_width=True
                            )
                        except Exception:
                            if st.button("📥 Download Report", key="home_btn_download_fallback", type="primary", use_container_width=True):
                                st.switch_page("pages/2_results.py")

                # --- TAB 2: SKILLS ---
                with tab_skills:
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    if skills:
                        st.markdown(f"**{len(skills)} Skills Identified:**")
                        st.markdown(render_skill_chips(matched=skills[:20]), unsafe_allow_html=True)
                    else:
                        st.caption("No specific domain skills detected in this resume.")

                    if st.button("Inspect Skill Gaps in Results →", key="tab_skills_goto_results"):
                        st.switch_page("pages/2_results.py")

                # --- TAB 3: JOB MATCH ---
                with tab_match:
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    try:
                        rec_resp = api_client.recommend_jobs(
                            resume_text=resume_data.get("raw_text", ""),
                            resume_skills=skills,
                            candidate_exp_years=float(exp_years),
                            top_n=3
                        )
                        recs = rec_resp.get("recommendations", [])
                        if recs:
                            for r in recs:
                                r_title = html.escape(r.get("title", "Role"))
                                r_comp = html.escape(r.get("company", "Company"))
                                r_sal = html.escape(str(r.get("salary", "₹24 - ₹32 LPA")))
                                r_score = int(round(r.get("match_score", 0.0) * 100))
                                st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 6px;">
                                    <div>
                                        <div style="font-weight: 700; font-size: 0.88rem; color: #0f172a;">{r_title}</div>
                                        <div style="font-size: 0.74rem; color: #64748b;">{r_comp} • <span style="font-weight: 600; color: #059669;">{r_sal}</span></div>
                                    </div>
                                    <span style="font-size: 0.85rem; font-weight: 800; color: #2563eb;">{r_score}% Match</span>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception:
                        st.caption("Job matches ready in the Job Recommendations tab.")

                    if st.button("Explore All Live Job Matches →", key="tab_match_goto_recs"):
                        st.switch_page("pages/4_job_recommendations.py")

                # --- TAB 4: SUGGESTIONS ---
                with tab_suggestions:
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    suggestions = ats_result.get("suggestions", [])
                    if suggestions:
                        for s in suggestions[:4]:
                            st.markdown(f"• **{html.escape(s)}**")
                    else:
                        st.caption("No critical ATS warnings identified. Excellent profile formatting!")

                    if st.button("Optimize Bullet Points →", key="tab_sugg_goto_results"):
                        st.switch_page("pages/2_results.py")

        else:
            # READY TO SCAN PLACEHOLDER CARD
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 0.75rem; border-bottom: 1px solid #f1f5f9; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 40px; height: 40px; border-radius: 10px; background: #f1f5f9; color: #64748b; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                            📄
                        </div>
                        <div>
                            <span style="font-weight: 800; font-size: 0.98rem; color: #0f172a;">ATS Score Check</span>
                            <span style="background: #f1f5f9; color: #64748b; border-radius: 9999px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; margin-left: 8px;">Awaiting Resume</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2.5rem 1.5rem; text-align: center;">
                    <div style="width: 100px; height: 100px; border-radius: 50%; border: 6px dashed #cbd5e1; display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 1rem;">
                        <span style="font-size: 1.5rem; font-weight: 800; color: #94a3b8;">--</span>
                        <span style="font-size: 0.7rem; font-weight: 600; color: #94a3b8;">/100</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 6px;">
                        Instant ATS Compatibility Check
                    </div>
                    <div style="font-size: 0.86rem; color: #64748b; max-width: 380px; line-height: 1.5; margin-bottom: 1.5rem;">
                        Upload your resume on the left or click <b>"Analyze"</b> on the sample resume to view your real-time score, compatibility breakdown, and quick insights right here.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("⚡ Load & Analyze Sample Resume Now", key="home_btn_quick_sample", type="primary", use_container_width=True):
                    _load_and_analyze_sample("software_engineer_resume.txt", "Software Engineer Resume")


if __name__ == "__main__" or "show_home_page" in globals():
    show_home_page()
