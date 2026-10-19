"""
Job matching and recommendation routes: JD matching, skill gap analysis, and recommendations.
"""

import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import MatchHistoryRecord
from app.services.matcher import compute_match
from app.services.recommender import recommend_jobs, load_jobs_dataset
from app.services.llm_feedback import get_ai_feedback
from app.schemas import (
    JobMatchRequest,
    JobMatchResponse,
    JobRecommendationRequest,
    JobRecommendationResponse,
    JobItem,
    AIFeedbackResponse
)

router = APIRouter(prefix="", tags=["Jobs & Matching"])


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

    # Generate AI feedback
    ai_feedback_data = get_ai_feedback(
        resume_text=request.resume_text,
        jd_text=request.jd_text,
        matched_skills=match_result["matched_skills"],
        missing_skills=match_result["missing_skills"],
        api_key=request.api_key
    )

    # Persist in match history
    try:
        record = MatchHistoryRecord(
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

    return JobRecommendationResponse(
        total_jobs_evaluated=len(items),
        recommendations=items
    )


@router.get("/jobs/list")
def list_available_jobs(limit: int = Query(60, ge=1, le=100)):
    """Returns curated jobs from data/jobs.csv for quick selection in the UI."""
    df = load_jobs_dataset()
    jobs = []
    for _, row in df.head(limit).iterrows():
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
    return {"jobs": jobs, "total": len(jobs)}
