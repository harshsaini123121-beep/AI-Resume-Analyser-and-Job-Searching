"""
Job matching and recommendation routes: JD matching, skill gap analysis, and recommendations.
"""

import json
import threading
import uuid
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import MatchHistoryRecord
from app.services.matcher import compute_match
from app.services.recommender import recommend_jobs, load_jobs_dataset
from app.services.llm_feedback import get_ai_feedback
from app.services.xai_explainer import explain_match_evidence
from app.services.extractor import extract_resume_data, extract_skills, extract_experience_years
from app.services.scorer import calculate_ats_score
from app.schemas import (
    JobMatchRequest,
    JobMatchResponse,
    JobRecommendationRequest,
    JobRecommendationResponse,
    JobItem,
    AIFeedbackResponse,
    XAIEvidenceRequest,
    XAIEvidenceResponse,
    XAIEvidenceItem,
    BatchScreenRequest,
    BatchScreenResponse,
    BatchCandidateResult
)

router = APIRouter(prefix="", tags=["Jobs & Matching"])

# In-memory background task registry for asynchronous batch screening
BATCH_TASKS: Dict[str, Dict[str, Any]] = {}
BATCH_TASKS_LOCK = threading.Lock()


@router.post("/match", response_model=JobMatchResponse)
def match_resume_with_job(
    request: JobMatchRequest,
    db: Session = Depends(get_db)
):
    """
    Computes hybrid match score between candidate resume and target job description:
    evaluates semantic similarity, skill overlaps/gaps, and experience alignment.
    Includes AI coaching feedback (Gemini or offline rule-based fallback).
    """
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")
    if not request.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    match_result = compute_match(
        resume_text=request.resume_text,
        resume_skills=request.resume_skills,
        jd_text=request.jd_text,
        jd_skills=request.jd_skills,
        candidate_exp_years=request.candidate_exp_years,
        required_exp_years=request.required_exp_years
    )

    # Generate AI feedback using server-side configured key
    ai_feedback_data = get_ai_feedback(
        resume_text=request.resume_text,
        jd_text=request.jd_text,
        matched_skills=match_result["matched_skills"],
        missing_skills=match_result["missing_skills"]
    )

    # Persist in match history
    try:
        record = MatchHistoryRecord(
            owner_id="default_user",
            job_title=request.jd_text.splitlines()[0][:100] if request.jd_text else "Target Position",
            match_percentage=match_result["match_percentage"],
            verdict=match_result["verdict"],
            matched_skills_json=json.dumps(match_result["matched_skills"]),
            missing_skills_json=json.dumps(match_result["missing_skills"])
        )
        db.add(record)
        db.commit()
    except Exception as e:
        print(f"[Match History] Warning: failed to save match history: {e}")

    return JobMatchResponse(
        match_percentage=match_result["match_percentage"],
        verdict=match_result["verdict"],
        verdict_color=match_result["verdict_color"],
        recommendation=match_result["recommendation"],
        skill_score=match_result["skill_score"],
        semantic_score=match_result["semantic_score"],
        experience_score=match_result["experience_score"],
        matched_skills=match_result["matched_skills"],
        missing_skills=match_result["missing_skills"],
        additional_skills=match_result["additional_skills"],
        ai_feedback=AIFeedbackResponse(**ai_feedback_data) if ai_feedback_data else None
    )


@router.post("/jobs/recommend", response_model=JobRecommendationResponse)
def get_job_recommendations(request: JobRecommendationRequest):
    """
    Scans the curated database of industry positions and returns the top N roles
    most compatible with the candidate's skills and experience.
    """
    if not request.resume_text.strip() and not request.resume_skills:
        raise HTTPException(status_code=400, detail="Either resume text or skills must be provided.")

    recs = recommend_jobs(
        resume_text=request.resume_text,
        resume_skills=request.resume_skills,
        candidate_exp_years=request.candidate_exp_years,
        top_n=request.top_n,
        location_filter=request.location_filter,
        max_experience_required=request.max_experience,
        search_query=request.search_query
    )

    items = [JobItem(**r) for r in recs]
    df = load_jobs_dataset()
    total_evaluated = len(df) if df is not None and not df.empty else len(items)

    return JobRecommendationResponse(
        total_jobs_evaluated=total_evaluated,
        recommendations=items
    )


@router.get("/jobs/list")
def list_available_jobs(
    offset: int = Query(0, ge=0),
    limit: int = Query(60, ge=1, le=100)
):
    """Returns curated jobs from data/jobs.csv for quick selection in the UI with pagination."""
    df = load_jobs_dataset()
    total = len(df) if df is not None else 0
    jobs = []
    if df is not None and not df.empty:
        sliced_df = df.iloc[offset:offset + limit]
        for _, row in sliced_df.iterrows():
            raw_skills = [
                s.strip().lower()
                for s in str(row["required_skills"]).replace(";", ",").split(",")
                if s.strip()
            ]
            jobs.append({
                "job_id": str(row["job_id"]),
                "title": str(row["title"]),
                "company": str(row["company"]),
                "location": str(row["location"]),
                "salary": str(row["salary"]),
                "experience_required": float(row["experience_required"]),
                "description": str(row["description"]),
                "required_skills": raw_skills
            })
    return {"jobs": jobs, "total": total, "offset": offset, "limit": limit}


@router.post("/match/xai-evidence", response_model=XAIEvidenceResponse)
def get_xai_evidence(request: XAIEvidenceRequest):
    """Generates sentence-level evidence mapping between job requirements and candidate resume."""
    if not request.resume_text.strip() or not request.jd_text.strip():
        raise HTTPException(status_code=400, detail="Resume text and JD text must not be empty.")
    evidence = explain_match_evidence(
        resume_text=request.resume_text,
        jd_text=request.jd_text,
        top_n_requirements=request.top_n_requirements
    )
    items = [XAIEvidenceItem(**e) for e in evidence]
    return XAIEvidenceResponse(evidence=items)


def _process_candidate_batch(request: BatchScreenRequest, job_id: Optional[str] = None) -> BatchScreenResponse:
    req_skills = request.required_skills or []
    if not req_skills and request.job_id:
        df = load_jobs_dataset()
        job_rows = df[df["job_id"] == str(request.job_id)]
        if not job_rows.empty:
            req_skills = [
                s.strip().lower()
                for s in str(job_rows.iloc[0]["required_skills"]).replace(";", ",").split(",")
                if s.strip()
            ]

    results = []
    total = len(request.candidates)
    for idx, cand in enumerate(request.candidates):
        text = cand.text
        ext = extract_resume_data(text)
        cand_exp = float(ext.get("experience_years") or 0.0)

        match = compute_match(
            resume_text=text,
            resume_skills=ext.get("skills", []),
            jd_text=request.jd_text,
            jd_skills=req_skills,
            candidate_exp_years=cand_exp,
            required_exp_years=request.experience_required
        )
        ats = calculate_ats_score(text, ext)

        results.append(BatchCandidateResult(
            candidate_name=ext.get("candidate_name") or "Applicant",
            filename=cand.filename,
            match_percentage=match["match_percentage"],
            ats_score=ats.get("total_score", 0),
            experience_years=cand_exp,
            skills_matched_count=len(match.get("matched_skills", [])),
            missing_skills=match.get("missing_skills", [])[:5]
        ))

        if job_id:
            with BATCH_TASKS_LOCK:
                if job_id in BATCH_TASKS:
                    BATCH_TASKS[job_id]["processed"] = idx + 1
                    BATCH_TASKS[job_id]["progress_percentage"] = round((idx + 1) / total * 100, 1)

    results.sort(key=lambda x: x.match_percentage, reverse=True)
    return BatchScreenResponse(
        job_title=request.jd_title or "Target Role",
        candidates_screened=len(results),
        results=results
    )


def _execute_background_batch(job_id: str, request: BatchScreenRequest):
    try:
        response = _process_candidate_batch(request, job_id=job_id)
        with BATCH_TASKS_LOCK:
            if job_id in BATCH_TASKS:
                BATCH_TASKS[job_id]["status"] = "completed"
                BATCH_TASKS[job_id]["result"] = response.model_dump()
    except Exception as e:
        with BATCH_TASKS_LOCK:
            if job_id in BATCH_TASKS:
                BATCH_TASKS[job_id]["status"] = "failed"
                BATCH_TASKS[job_id]["error"] = str(e)


@router.post("/jobs/batch-screen", response_model=BatchScreenResponse)
def batch_screen_candidates(request: BatchScreenRequest):
    """
    Screens multiple candidate resumes against a target job requisition,
    evaluating ATS compliance, tenure match, and skill overlap.
    """
    if not request.candidates:
        raise HTTPException(status_code=400, detail="Candidate list cannot be empty.")
    return _process_candidate_batch(request)


@router.post("/jobs/batch-screen-async")
def batch_screen_candidates_async(
    request: BatchScreenRequest,
    background_tasks: BackgroundTasks
):
    """
    Submits candidate resumes for asynchronous background screening.
    Returns tracking job_id and initial status.
    """
    if not request.candidates:
        raise HTTPException(status_code=400, detail="Candidate list cannot be empty.")

    job_id = str(uuid.uuid4())
    with BATCH_TASKS_LOCK:
        BATCH_TASKS[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "processed": 0,
            "total": len(request.candidates),
            "progress_percentage": 0.0,
            "result": None,
            "error": None
        }

    background_tasks.add_task(_execute_background_batch, job_id, request)
    return {"job_id": job_id, "status": "processing"}


@router.get("/jobs/batch/{job_id}/progress")
def get_batch_screening_progress(job_id: str):
    """Returns current progress or completed results of an asynchronous batch screening task."""
    with BATCH_TASKS_LOCK:
        task = BATCH_TASKS.get(job_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Batch job '{job_id}' not found.")
    return task


@router.post("/jobs/extract-skills")
def extract_skills_endpoint(payload: dict):
    """Extracts skills and experience tenure from text via the taxonomy."""
    text = str(payload.get("text", ""))
    skills = extract_skills(text)
    exp = extract_experience_years(text)
    return {"skills": skills, "experience_years": exp}


