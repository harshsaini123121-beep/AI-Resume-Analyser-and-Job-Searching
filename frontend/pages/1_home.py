"""
Page 1: Home / Resume Upload
Clean single upload area, four-step pipeline strip, and demo data sample links.
"""
from datetime import datetime
import hashlib
import html
import time
from pathlib import Path

import streamlit as st

from frontend.api_client import api_client, APIConnectionError, APIResponseError
from frontend.components.empty_state import render_empty_state
from frontend.components.error_banner import render_error_banner

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def show_home_page():
    # Four-Step Pipeline Strip
    st.markdown("""
    <div class="pipeline-strip" role="navigation" aria-label="Workflow Steps">
        <div class="pipeline-step active">
            <span class="pipeline-step-num">1</span>
            <span>Upload</span>
        </div>
        <span class="pipeline-arrow">➔</span>
        <div class="pipeline-step">
            <span class="pipeline-step-num">2</span>
            <span>Analyse</span>
        </div>
        <span class="pipeline-arrow">➔</span>
        <div class="pipeline-step">
            <span class="pipeline-step-num">3</span>
            <span>Match</span>
        </div>
        <span class="pipeline-arrow">➔</span>
        <div class="pipeline-step">
            <span class="pipeline-step-num">4</span>
            <span>Improve</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">AI Resume Analyser & ATS Scanner</h1>
        <p class="page-subtitle">Upload your resume to receive quantitative ATS metrics, skill gap discovery, and target role matching.</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_sample = st.columns([1.4, 1.0], gap="large")

    with col_upload:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">Upload Document</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Select Resume File",
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
                key="home_resume_uploader",
                help="Supported formats: PDF, DOCX, TXT. Maximum file size: 5 MB."
            )

            st.caption("Supported formats: **PDF, DOCX, TXT** (Max 5 MB). Verified locally on your machine.")

            if uploaded_file is not None:
                raw_bytes = uploaded_file.getvalue()
                if len(raw_bytes) == 0:
                    st.markdown(render_error_banner("Empty File", "The uploaded file contains 0 bytes.", "Please select a valid, non-empty resume document."), unsafe_allow_html=True)
                elif len(raw_bytes) > 5 * 1024 * 1024:
                    st.markdown(render_error_banner("File Too Large", f"File size ({len(raw_bytes)/1024/1024:.1f} MB) exceeds the 5 MB limit.", "Please compress or convert your document."), unsafe_allow_html=True)
                else:
                    curr_hash = hashlib.sha256(raw_bytes).hexdigest()
                    if st.session_state.get("processed_file_hash") != curr_hash:
                        # Step-by-step progress status container
                        with st.status("Analyzing resume document...", expanded=True) as status_box:
                            st.write("📖 Reading document and validating structure...")
                            time.sleep(0.15)
                            st.write("🔍 Extracting candidate contact info, sections, and technical skills...")
                            time.sleep(0.15)
                            try:
                                analysis = api_client.analyse_resume(raw_bytes, uploaded_file.name)
                                st.write("🎯 Computing quantitative 8-dimension ATS score...")

                                st.session_state.resume_data = analysis["extracted_data"]
                                st.session_state.ats_result = analysis["ats_score"]
                                st.session_state.page_count = analysis.get("page_count", 1)
                                st.session_state.active_filename = uploaded_file.name
                                st.session_state.processed_file_hash = curr_hash
                                st.session_state.is_sample = False
                                st.session_state.analysis_timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
                                status_box.update(label="✅ Resume successfully analyzed!", state="complete", expanded=False)
                                st.toast("Analysis complete! View details in the Analysis Results tab.", icon="🎉")
                            except APIResponseError as are:
                                status_box.update(label="❌ Analysis Failed", state="error", expanded=True)
                                st.markdown(render_error_banner("Document Extraction Error", are.detail, "Ensure the document is not password-protected."), unsafe_allow_html=True)
                            except APIConnectionError as ace:
                                status_box.update(label="❌ Backend Connection Error", state="error", expanded=True)
                                st.markdown(render_error_banner("Backend Unavailable", str(ace), "Verify the FastAPI server is running at http://127.0.0.1:8000."), unsafe_allow_html=True)

            if st.session_state.get("resume_data"):
                safe_name = html.escape(st.session_state.get("active_filename", "resume.pdf"))
                tag = "Demo data" if st.session_state.get("is_sample") else "Active Document"
                st.markdown(f"""
                <div style="margin-top: 16px; padding: 12px; background-color: var(--color-success-bg); border: 1px solid var(--color-success-border); border-radius: var(--card-radius); display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; color: var(--color-success-text); font-size: var(--font-sm);">✓ {safe_name}</div>
                        <div style="font-size: var(--font-xs); color: var(--color-muted);">{tag} • Analysed on {st.session_state.get('analysis_timestamp', 'Just now')}</div>
                    </div>
                    <span class="badge-tag" style="background: var(--color-surface); color: var(--color-text);">Ready</span>
                </div>
                """, unsafe_allow_html=True)

    with col_sample:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 4px;">Explore Demo Data</div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: var(--font-xs); color: var(--color-muted); margin-bottom: 16px;">Test the system instantly without uploading your personal files.</p>', unsafe_allow_html=True)

            sample_roles = [
                ("software_engineer_resume.txt", "Software Engineer", "4 yrs experience • Fullstack & Cloud", "💻"),
                ("david_kumar_data_analyst.txt", "Data Analyst", "3 yrs experience • Python, SQL & Tableau", "📊"),
                ("senior_ml_engineer_resume.txt", "Senior ML Engineer", "6 yrs experience • PyTorch & NLP", "🤖"),
            ]

            for s_fname, s_title, s_desc, s_icon in sample_roles:
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--color-border);">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="font-size: 1.3rem;">{s_icon}</div>
                        <div>
                            <div style="font-weight: 700; font-size: var(--font-sm); color: var(--color-text);">{html.escape(s_title)} <span class="badge-tag" style="font-size: 10px; background: var(--color-border); color: var(--color-muted);">Demo data</span></div>
                            <div style="font-size: var(--font-xs); color: var(--color-muted);">{html.escape(s_desc)}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Load {s_title} Demo", key=f"load_sample_{s_fname}", type="secondary", use_container_width=True):
                    sample_path = ROOT_DIR / "data" / "sample_resumes" / s_fname
                    if sample_path.exists():
                        raw_text = sample_path.read_text(encoding="utf-8")
                        s_bytes = raw_text.encode("utf-8")
                        s_hash = hashlib.sha256(s_bytes).hexdigest()
                        try:
                            with st.spinner(f"Loading {s_title} demo data..."):
                                analysis = api_client.analyse_resume(s_bytes, s_fname)
                                st.session_state.resume_data = analysis["extracted_data"]
                                st.session_state.ats_result = analysis["ats_score"]
                                st.session_state.page_count = analysis.get("page_count", 1)
                                st.session_state.active_filename = s_fname
                                st.session_state.processed_file_hash = s_hash
                                st.session_state.is_sample = True
                                st.session_state.analysis_timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
                                st.toast(f"Loaded {s_title} demo data successfully!", icon="✅")
                                st.rerun()
                        except Exception as ex:
                            st.error(f"Failed to load demo: {ex}")



if __name__ == "__main__" or "show_home_page" in globals():
    show_home_page()
