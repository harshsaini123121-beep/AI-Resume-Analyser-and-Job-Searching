"""
Job recommender service matching candidate profiles against a curated jobs database.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from app.services.matcher import compute_match, compute_semantic_similarity

DEFAULT_JOBS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "jobs.csv"

# In-memory cached dataframe
_CACHED_JOBS_DF: Optional[pd.DataFrame] = None


def load_jobs_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Loads and caches the jobs dataset from CSV."""
    global _CACHED_JOBS_DF
    if _CACHED_JOBS_DF is not None and not csv_path:
        return _CACHED_JOBS_DF

    path = Path(csv_path) if csv_path else DEFAULT_JOBS_PATH
    if not path.exists():
        # Fallback sample jobs
        return pd.DataFrame([
            {
                "job_id": 1,
                "title": "Machine Learning Engineer",
                "company": "Tech Corp",
                "location": "Remote",
                "description": "Develop and deploy deep learning models using Python and PyTorch.",
                "required_skills": "python; pytorch; machine learning; docker",
                "experience_required": 3,
                "salary": "$130,000 - $160,000"
            }
        ])

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=[
            "job_id", "title", "company", "location", "description",
            "required_skills", "experience_required", "salary"
        ])

    for col in ["job_id", "title", "company", "location", "description", "required_skills", "experience_required", "salary"]:
        if col not in df.columns:
            df[col] = "" if col != "experience_required" else 0.0

    # Ensure standard columns exist and fill NaNs
    df["job_id"] = df["job_id"].astype(str)
    df["title"] = df["title"].fillna("Software Professional")
    df["company"] = df["company"].fillna("Confidential")
    df["location"] = df["location"].fillna("Flexible")
    df["description"] = df["description"].fillna("")
    df["required_skills"] = df["required_skills"].fillna("")
    df["experience_required"] = pd.to_numeric(df["experience_required"], errors="coerce").fillna(0.0)
    df["salary"] = df["salary"].fillna("Competitive")

    if not csv_path:
        _CACHED_JOBS_DF = df

    return df


def recommend_jobs(
    resume_text: str,
    resume_skills: List[str],
    candidate_exp_years: float = 0.0,
    top_n: int = 10,
    location_filter: Optional[str] = None,
    max_experience_required: Optional[float] = None,
    search_query: Optional[str] = None,
    csv_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Ranks all jobs in the database against the candidate resume and returns the top_n recommendations.
    Supports filtering by location, experience requirements, and keyword search.
    """
    df = load_jobs_dataset(csv_path)

    # Apply filters
    filtered_df = df.copy()

    if location_filter and location_filter.strip().lower() not in ("all", "any", ""):
        filtered_df = filtered_df[
            filtered_df["location"].str.contains(location_filter.strip(), case=False, na=False)
        ]

    if max_experience_required is not None and max_experience_required > 0:
        filtered_df = filtered_df[filtered_df["experience_required"] <= max_experience_required]

    if search_query and search_query.strip():
        q = search_query.strip().lower()
        title_mask = filtered_df["title"].str.lower().str.contains(q, na=False)
        desc_mask = filtered_df["description"].str.lower().str.contains(q, na=False)
        skill_mask = filtered_df["required_skills"].str.lower().str.contains(q, na=False)
        filtered_df = filtered_df[title_mask | desc_mask | skill_mask]

    if filtered_df.empty:
        return []

    results = []

    for _, row in filtered_df.iterrows():
        # Parse required skills from semicolon or comma separated string
        raw_skills = [
            s.strip().lower()
            for s in str(row["required_skills"]).replace(";", ",").split(",")
            if s.strip()
        ]

        # Compute match using the matching service
        match_result = compute_match(
            resume_text=resume_text,
            resume_skills=resume_skills,
            jd_text=str(row["description"]),
            jd_skills=raw_skills,
            candidate_exp_years=candidate_exp_years,
            required_exp_years=float(row["experience_required"])
        )

        results.append({
            "job_id": str(row["job_id"]),
            "title": str(row["title"]),
            "company": str(row["company"]),
            "location": str(row["location"]),
            "salary": str(row["salary"]),
            "experience_required": float(row["experience_required"]),
            "description": str(row["description"]),
            "required_skills": raw_skills,
            "match_percentage": match_result["match_percentage"],
            "verdict": match_result["verdict"],
            "verdict_color": match_result["verdict_color"],
            "matched_skills": match_result["matched_skills"],
            "missing_skills": match_result["missing_skills"],
            "skill_score": match_result["skill_score"],
            "semantic_score": match_result["semantic_score"]
        })

    # Sort descending by match percentage
    results.sort(key=lambda x: x["match_percentage"], reverse=True)

    return results[:top_n]
