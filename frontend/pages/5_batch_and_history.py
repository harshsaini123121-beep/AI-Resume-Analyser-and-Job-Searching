"""
Page 5: Batch Screening & Audit History
Unified Recruiter tools: Multi-candidate batch screening with asynchronous background progress,
sortable candidate table, expandable detail view, CSV export,
plus Score-over-time trend chart, PDF report download, and confirmed record deletion.
"""
from datetime import datetime
import hashlib
import html
import io
from pathlib import Path
import time

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from frontend.api_client import api_client, APIResponseError
from frontend.components.empty_state import render_empty_state
from frontend.components.error_banner import render_error_banner
from frontend.components.stat_card import render_stat_card

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_jobs_dataframe() -> pd.DataFrame:
    try:
        jobs = api_client.list_jobs(limit=100)
        if jobs:
            return pd.DataFrame(jobs)
    except Exception:
        pass
    return pd.DataFrame()


def show_batch_and_history_page():
    st.markdown("""
    <div class="page-main-header">
        <h1 class="page-title">Recruiter Hub: Batch Screening & Audit History</h1>
        <p class="page-subtitle">Screen candidate cohorts simultaneously against any requisition, export leaderboards, and audit compliance history.</p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Select Workflow",
        ["🥞 Batch Screening", "📊 Audit History & Reports"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # -------------------------------------------------------------
    # SUB-VIEW 1: BATCH SCREENING
    # -------------------------------------------------------------
    if "Batch Screening" in mode:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">1. Target Job Requisition</div>', unsafe_allow_html=True)

            jobs_df = get_jobs_dataframe()
            if jobs_df.empty:
                st.warning("Job requisition dataset is currently unavailable.")
                return

            job_options = [f"{r['job_id']}: {r['title']} @ {r['company']}" for _, r in jobs_df.head(40).iterrows()]
            selected_job_str = st.selectbox("Select Benchmark Role", job_options, key="batch_job_select")
            target_job_row = jobs_df[jobs_df["job_id"] == selected_job_str.split(":")[0].strip()].iloc[0]

        # Batch Upload Section
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: var(--font-base); color: var(--color-text); margin-bottom: 8px;">2. Upload Candidate Cohort</div>', unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Upload Multiple Resumes (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="batch_multi_uploader",
                help="Select up to 20 candidate resumes to screen in parallel."
            )

            col_sample_batch, _ = st.columns([1.5, 2])
            with col_sample_batch:
                if st.button("📁 Load 3 Pre-Packaged Demo Candidates", type="secondary", key="load_sample_batch_btn"):
                    sample_dir = ROOT_DIR / "data" / "sample_resumes"
                    preloaded = []
                    for p in sample_dir.glob("*.txt"):
                        try:
                            preloaded.append((p.name, p.read_text(encoding="utf-8")))
                        except Exception:
                            pass
                    st.session_state.batch_preloaded = preloaded
                    st.toast("Loaded sample candidate resumes!", icon="📂")

        candidates_payload = []
        file_hashes = []

        if uploaded_files:
            for f in uploaded_files:
                f_bytes = f.getvalue()
                if len(f_bytes) == 0:
                    st.warning(f"⚠️ **{f.name}**: File is empty (0 bytes). Skipped.")
                    continue
                if len(f_bytes) > 5 * 1024 * 1024:
                    st.warning(f"⚠️ **{f.name}**: Exceeds 5 MB limit. Skipped.")
                    continue
                try:
                    upload_res = api_client.upload_resume(f_bytes, f.name)
                    raw_t = upload_res["extracted_data"]["raw_text"]
                    f_hash = hashlib.sha256(f_bytes).hexdigest()
                    candidates_payload.append({"filename": f.name, "text": raw_t})
                    file_hashes.append(f_hash)
                except APIResponseError as are:
                    st.warning(f"⚠️ **{f.name}**: Parsing Error ({are.detail}). Skipped.")
                except Exception as ex:
                    st.warning(f"⚠️ **{f.name}**: Error ({ex}). Skipped.")

        elif st.session_state.get("batch_preloaded"):
            for fname, text in st.session_state.batch_preloaded:
                candidates_payload.append({"filename": fname, "text": text})
                file_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())

        # If candidates exist, execute screening
        if candidates_payload:
            t_skills = [s.strip().lower() for s in str(target_job_row.get("required_skills", "")).replace(";", ",").split(",") if s.strip()]
            req_exp = float(target_job_row.get("experience_required", 0.0) or 0.0)

            st.markdown(f'<div style="font-weight: 700; font-size: var(--font-base); margin-top: 14px; margin-bottom: 10px;">Evaluating {len(candidates_payload)} Candidates for <i>{html.escape(target_job_row["title"])}</i></div>', unsafe_allow_html=True)

            cache_key = f"{target_job_row['job_id']}_" + hashlib.sha256(",".join(sorted(file_hashes)).encode("utf-8")).hexdigest()

            if st.session_state.get("batch_cache_key") == cache_key and "batch_cached_results" in st.session_state:
                ranked = st.session_state.batch_cached_results
            else:
                batch_response = None
                # Asynchronous screening with progress bar
                try:
                    async_job = api_client.batch_screen_async(
                        job_id=str(target_job_row["job_id"]),
                        jd_title=str(target_job_row["title"]),
                        jd_text=str(target_job_row["description"]),
                        required_skills=t_skills,
                        experience_required=req_exp,
                        candidates=candidates_payload
                    )
                    batch_id = async_job.get("job_id")
                    progress_placeholder = st.empty()

                    for _ in range(80):
                        time.sleep(0.3)
                        poll = api_client.get_batch_progress(batch_id)
                        pct = int(poll.get("progress_pct", 0))
                        msg = poll.get("current_file") or "Screening candidate resumes..."
                        progress_placeholder.progress(min(100, max(5, pct)), text=f"Screening: {msg} ({pct}%)")

                        if poll.get("status") == "completed":
                            batch_response = poll
                            progress_placeholder.progress(100, text="✨ Screening complete!")
                            time.sleep(0.3)
                            progress_placeholder.empty()
                            break
                        elif poll.get("status") == "failed":
                            progress_placeholder.empty()
                            raise RuntimeError(poll.get("error") or "Screening failed")
                except Exception:
                    # Fallback to sync
                    try:
                        with st.spinner("Processing candidate cohort..."):
                            batch_response = api_client.batch_screen(
                                job_id=str(target_job_row["job_id"]),
                                jd_title=str(target_job_row["title"]),
                                jd_text=str(target_job_row["description"]),
                                required_skills=t_skills,
                                experience_required=req_exp,
                                candidates=candidates_payload
                            )
                    except Exception as ex:
                        st.error(f"Batch screening failed: {ex}")
                        batch_response = None

                if batch_response:
                    raw_res = batch_response.get("results", [])
                    ranked = sorted(raw_res, key=lambda x: x.get("match_percentage", 0), reverse=True)
                else:
                    ranked = []

                st.session_state.batch_cache_key = cache_key
                st.session_state.batch_cached_results = ranked

            if ranked:
                with st.container(border=True):
                    st.markdown('<div style="font-weight: 800; font-size: var(--font-lg); color: var(--color-text); margin-bottom: 12px;">🏆 Candidate Screening Leaderboard</div>', unsafe_allow_html=True)

                    medals = ["🥇 #1", "🥈 #2", "🥉 #3"]
                    table_rows = []
                    for idx, r in enumerate(ranked):
                        rank_str = medals[idx] if idx < len(medals) else f"#{idx+1}"
                        exp_val = r.get("experience_years", 0.0)
                        exp_str = f"{exp_val:.1f} yrs" if exp_val and exp_val > 0 else "Not detected"
                        missing_str = ", ".join(r.get("missing_skills", [])[:3]) if r.get("missing_skills") else "None"

                        table_rows.append({
                            "Rank": rank_str,
                            "Candidate": r.get("candidate_name") or "Applicant",
                            "Filename": r.get("filename", ""),
                            "Match Score": f"{r.get('match_percentage', 0)}%",
                            "ATS Score": f"{r.get('ats_score', 0)}/100",
                            "Experience": exp_str,
                            "Skills Matched": r.get("skills_matched_count", 0),
                            "Top Skill Gaps": missing_str
                        })

                    df_leaderboard = pd.DataFrame(table_rows)
                    st.dataframe(df_leaderboard, width="stretch")

                    # CSV Export Button
                    csv_bytes = df_leaderboard.to_csv(index=False).encode("utf-8")
                    clean_title = "".join(c for c in target_job_row["title"] if c.isalnum() or c == "_")
                    st.download_button(
                        label="📥 Export Leaderboard to CSV",
                        data=csv_bytes,
                        file_name=f"Leaderboard_{clean_title}.csv",
                        mime="text/csv",
                        key="dl_batch_csv_btn"
                    )

                    # Expandable Candidate Details
                    st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: 16px 0;'>", unsafe_allow_html=True)
                    st.markdown('<div style="font-weight: 700; font-size: var(--font-base); margin-bottom: 8px;">Candidate Detail Drilldown</div>', unsafe_allow_html=True)

                    for r in ranked:
                        with st.expander(f"{r.get('candidate_name', 'Applicant')} — {r.get('match_percentage', 0)}% Match ({r.get('filename')})"):
                            d_col1, d_col2 = st.columns(2)
                            with d_col1:
                                st.write(f"**ATS Score:** {r.get('ats_score', 0)} / 100")
                                st.write(f"**Experience:** {r.get('experience_years', 0.0):.1f} years")
                            with d_col2:
                                st.write(f"**Missing Skills:** {', '.join(r.get('missing_skills', [])) if r.get('missing_skills') else 'None'}")
                                if st.button("Set as Active Resume", key=f"set_active_{r.get('filename')}"):
                                    try:
                                        ana = api_client.analyse_resume(r.get("raw_text", "").encode("utf-8"), r.get("filename", "resume.txt"))
                                        st.session_state.resume_data = ana["extracted_data"]
                                        st.session_state.ats_result = ana["ats_score"]
                                        st.session_state.active_filename = r.get("filename")
                                        st.toast(f"Active resume set to {r.get('candidate_name')}!", icon="✅")
                                        st.switch_page("pages/2_results.py")
                                    except Exception as e:
                                        st.error(f"Failed to activate: {e}")

    # -------------------------------------------------------------
    # SUB-VIEW 2: AUDIT HISTORY & REPORTS
    # -------------------------------------------------------------
    else:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 800; font-size: var(--font-lg); color: var(--color-text); margin-bottom: 12px;">📊 Candidate Audit Database</div>', unsafe_allow_html=True)

            try:
                records = api_client.get_history(limit=50)
            except Exception as e:
                st.error(f"History service unavailable: {e}")
                records = []

            if records:
                # Score-Over-Time Trend Chart
                df_hist = pd.DataFrame(records)
                if "created_at" in df_hist.columns and "ats_score" in df_hist.columns:
                    try:
                        df_hist["date_parsed"] = pd.to_datetime(df_hist["created_at"])
                        df_sorted = df_hist.sort_values("date_parsed")

                        trend_fig = px.line(
                            df_sorted,
                            x="created_at",
                            y="ats_score",
                            title="ATS Score Trend Over Time",
                            markers=True,
                            hover_data=["candidate_name", "ats_grade"]
                        )
                        trend_fig.update_layout(
                            height=260,
                            margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Timestamp"),
                            yaxis=dict(range=[0, 100], showgrid=True, gridcolor='#f1f5f9', title="ATS Score")
                        )
                        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
                    except Exception:
                        pass

                # Table of History Records
                table_records = [
                    {
                        "ID": r["id"],
                        "Candidate": r.get("candidate_name") or "Applicant",
                        "Filename": r.get("filename") or "resume.pdf",
                        "ATS Score": f"{r.get('ats_score', 0)}/100",
                        "Grade": r.get("ats_grade") or "C",
                        "Experience": f"{r.get('experience_years', 0.0):.1f} yrs" if r.get("experience_years") else "Not detected",
                        "Date": r.get("created_at") or "-"
                    }
                    for r in records
                ]
                st.dataframe(pd.DataFrame(table_records), width="stretch")

                # Record Deletion with Confirmation Checkbox
                st.markdown("<hr style='border: none; border-top: 1px solid var(--color-border); margin: 16px 0;'>", unsafe_allow_html=True)
                st.markdown('<div style="font-weight: 700; font-size: var(--font-base); margin-bottom: 8px;">🗑️ Record Management & Compliance Deletion</div>', unsafe_allow_html=True)

                del_c1, del_c2 = st.columns([2.5, 1.0])
                with del_c1:
                    rec_map = {f"Record #{r['id']} — {r.get('candidate_name', 'Applicant')} ({r.get('filename')})": r["id"] for r in records}
                    chosen_rec_label = st.selectbox("Select Record", list(rec_map.keys()), key="del_rec_selector")
                with del_c2:
                    confirm_del = st.checkbox("Confirm Permanent Deletion", key="confirm_del_box")
                    if st.button("Delete Audit Record", type="secondary", disabled=not confirm_del):
                        rec_id = rec_map[chosen_rec_label]
                        try:
                            api_client.delete_history_record(rec_id)
                            st.success(f"Record #{rec_id} permanently deleted.")
                            st.rerun()
                        except Exception as dex:
                            st.error(f"Failed to delete: {dex}")

            else:
                st.info("No audit history found in database. Upload and analyze resumes to populate audit records.")


show_batch_and_history_page()
