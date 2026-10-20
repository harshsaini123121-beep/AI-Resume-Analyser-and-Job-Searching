"""
Job recommender service matching candidate profiles against a curated jobs database.
Implements a fast two-stage recommender:
Stage 1: Sub-millisecond vector dot product retrieval to identify top-50 candidate roles.
Stage 2: Comprehensive hybrid re-ranking evaluating required skills, preferred skills,
and career experience alignment.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from app.services.matcher import compute_match
from app.services.vector_indexer import (
    build_or_load_vector_index,
    search_vector_index,
    compute_vector_similarities
)

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
        return pd.DataFrame([
            {
                "job_id": "1",
                "title": "Machine Learning Engineer",
                "company": "Tech Corp",
                "location": "Remote",
                "description": "Develop and deploy deep learning models using Python and PyTorch.",
                "required_skills": "python; pytorch; machine learning",
                "preferred_skills": "docker; kubernetes",
                "experience_required": 3.0,
                "salary": "₹24 - ₹32 LPA"
            }
        ])

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=[
            "job_id", "title", "company", "location", "description",
            "required_skills", "preferred_skills", "experience_required", "salary"
        ])

    standard_cols = [
        "job_id", "title", "company", "location", "description",
        "required_skills", "preferred_skills", "experience_required", "salary"
    ]
    for col in standard_cols:
        if col not in df.columns:
            df[col] = "" if col != "experience_required" else 0.0

    df["job_id"] = df["job_id"].astype(str)
    df["title"] = df["title"].fillna("Software Professional")
    df["company"] = df["company"].fillna("Confidential")
    df["location"] = df["location"].fillna("Flexible")
    df["description"] = df["description"].fillna("")
    df["required_skills"] = df["required_skills"].fillna("")
    df["preferred_skills"] = df["preferred_skills"].fillna("")
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
    Ranks jobs against the candidate profile using a high-performance two-stage recommendation pipeline:
    1. Fast vector inner-product retrieval of candidate pool using pre-computed disk embeddings.
    2. Detailed hybrid re-ranking (required skills, preferred skills, tenure fit, semantic alignment).
    """
    df = load_jobs_dataset(csv_path)
    if df is None or df.empty:
        return []

    # 1. Apply hard metadata filters
    filtered_df = df.copy()

    if location_filter and location_filter.strip().lower() not in ("all", "any", "", "all locations"):
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

    # 2. Stage 1: Fast Vector Retrieval using pre-computed dense index
    query_text = f"{resume_text[:1200]} Skills: {', '.join(resume_skills[:20])}"
    vector_sim_map = compute_vector_similarities(query_text, csv_path=csv_path)

    candidates = filtered_df.to_dict(orient="records")
    for cand in candidates:
        cand_id = str(cand.get("job_id", ""))
        cand["_vector_sim"] = vector_sim_map.get(cand_id, 0.4)

    # Sort by vector similarity first to prioritize top semantically relevant roles
    candidates.sort(key=lambda x: x["_vector_sim"], reverse=True)
    # Stage 1 candidate pool for full hybrid re-ranking
    candidate_pool = candidates[:min(len(candidates), max(50, top_n * 3))]

    # 3. Stage 2: Detailed Hybrid Re-ranking
    results = []
    for row in candidate_pool:
        raw_skills = [
            s.strip().lower()
            for s in str(row.get("required_skills", "")).replace(";", ",").split(",")
            if s.strip()
        ]
        raw_preferred = [
            s.strip().lower()
            for s in str(row.get("preferred_skills", "")).replace(";", ",").split(",")
            if s.strip()
        ]

        match_result = compute_match(
            resume_text=resume_text,
            resume_skills=resume_skills,
            jd_text=str(row.get("description", "")),
            jd_skills=raw_skills,
            jd_preferred_skills=raw_preferred,
            candidate_exp_years=candidate_exp_years,
            required_exp_years=float(row.get("experience_required", 0.0)),
            precomputed_semantic_score=row["_vector_sim"]
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
            "preferred_skills": raw_preferred,
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
