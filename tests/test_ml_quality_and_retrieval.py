"""
Unit and integration tests for Task 3: ML Quality & Vector Retrieval.
Covers:
- Vector indexer SHA-256 caching and sub-millisecond search
- Sliding-window token chunking and aggregation in matcher
- Separation of required vs preferred skills in compute_match
- Two-stage recommender top-50 retrieval and re-ranking
- Evaluation benchmark datasets (dev.json, test.json, LABELING_GUIDE.md)
"""

import json
from pathlib import Path
import numpy as np
import pytest

from app.services.vector_indexer import (
    build_or_load_vector_index,
    search_vector_index,
    get_file_sha256,
    JOBS_CSV_PATH,
    EMBEDDINGS_CACHE_PATH,
    HASH_CACHE_PATH
)
from app.services.matcher import (
    chunk_text,
    compute_semantic_similarity,
    compute_match,
    is_model_loaded
)
from app.services.recommender import recommend_jobs, load_jobs_dataset


def test_vector_indexer_sha256_caching():
    """Verifies SHA-256 hash detection, disk-cache creation, and instant retrieval."""
    assert JOBS_CSV_PATH.exists()
    file_hash = get_file_sha256(JOBS_CSV_PATH)
    assert len(file_hash) == 64

    # Load vector index (uses cached embeddings)
    embs, jobs = build_or_load_vector_index()
    assert embs is not None
    assert len(jobs) == 721
    assert len(embs) == 721

    # Verify cache artifacts exist on disk
    assert EMBEDDINGS_CACHE_PATH.exists()
    assert HASH_CACHE_PATH.exists()
    assert HASH_CACHE_PATH.read_text(encoding="utf-8").strip() == file_hash


def test_fast_vector_search():
    """Verifies that vector search finds relevant roles for query in vector space."""
    query = "Senior Machine Learning Engineer with deep learning and PyTorch experience."
    results = search_vector_index(query, top_k=5)
    assert len(results) == 5

    top_job, score = results[0]
    assert "job_id" in top_job
    assert "title" in top_job
    assert -1.0 <= score <= 1.0


def test_sliding_window_token_chunking():
    """Verifies sliding-window chunking on long text passages (> 180 words)."""
    short_text = "This is a short resume bullet point."
    short_chunks = chunk_text(short_text, chunk_size=180, overlap=40)
    assert len(short_chunks) == 1
    assert short_chunks[0] == short_text

    long_text = " ".join([f"skill_{i} and experience_{i} in domain_{i}" for i in range(250)])
    long_chunks = chunk_text(long_text, chunk_size=180, overlap=40)
    assert len(long_chunks) >= 2

    # Verify overlap exists between consecutive chunks
    first_words = set(long_chunks[0].split()[-30:])
    second_words = set(long_chunks[1].split()[:50])
    assert len(first_words.intersection(second_words)) > 0


def test_chunked_semantic_similarity_bounds():
    """Verifies semantic similarity returns valid clipped scores [0.0, 1.0]."""
    text1 = "Senior Python Engineer with FastAPI, Docker, and PostgreSQL."
    text2 = "Looking for a Python Developer experienced in FastAPI and Docker."
    text3 = "Pastry chef baking artisanal French bread and croissants."

    sim_high = compute_semantic_similarity(text1, text2)
    sim_low = compute_semantic_similarity(text1, text3)

    assert 0.0 <= sim_high <= 1.0
    assert 0.0 <= sim_low <= 1.0
    assert sim_high > sim_low


def test_required_vs_preferred_skills_separation():
    """Verifies separate weighting and tracking of required vs preferred skills."""
    resume_skills = ["python", "fastapi", "docker"]
    jd_req = ["python", "fastapi"]
    jd_pref = ["docker", "kubernetes", "aws"]

    res = compute_match(
        resume_text="Experienced Python backend developer building FastAPI microservices.",
        resume_skills=resume_skills,
        jd_text="Backend engineer needed with Python, FastAPI, Docker, and Kubernetes.",
        jd_skills=jd_req,
        jd_preferred_skills=jd_pref,
        candidate_exp_years=4.0,
        required_exp_years=3.0
    )

    assert "matched_preferred_skills" in res
    assert "missing_preferred_skills" in res
    assert "docker" in res["matched_preferred_skills"]
    assert "kubernetes" in res["missing_preferred_skills"]
    assert res["required_skills_count"] == 2
    assert res["preferred_skills_count"] == 3
    assert res["skill_score"] > 0.7  # 100% of required (0.75) + 33% of preferred (0.25 * 0.33)


def test_two_stage_recommender():
    """Verifies two-stage recommender returns ranked top-N jobs with preferred skills."""
    resume = "Machine Learning Engineer with 5 years in Python, PyTorch, Deep Learning, and NLP."
    skills = ["python", "pytorch", "deep learning", "nlp"]

    recs = recommend_jobs(resume, skills, candidate_exp_years=5.0, top_n=5)
    assert len(recs) == 5
    assert recs[0]["match_percentage"] >= recs[-1]["match_percentage"]

    # Verify standard fields and preferred skills exist in items
    for item in recs:
        assert "job_id" in item
        assert "title" in item
        assert "preferred_skills" in item
        assert "required_skills" in item


def test_eval_benchmark_datasets_exist():
    """Verifies that the expanded 105-pair benchmark datasets and labeling guide exist."""
    base_dir = Path(__file__).resolve().parent.parent
    eval_dir = base_dir / "data" / "eval"

    guide_path = eval_dir / "LABELING_GUIDE.md"
    dev_path = eval_dir / "dev.json"
    test_path = eval_dir / "test.json"

    assert guide_path.exists(), "LABELING_GUIDE.md must exist in data/eval/"
    assert dev_path.exists(), "dev.json must exist in data/eval/"
    assert test_path.exists(), "test.json must exist in data/eval/"

    dev_data = json.loads(dev_path.read_text(encoding="utf-8"))
    test_data = json.loads(test_path.read_text(encoding="utf-8"))

    assert len(dev_data) == 50
    assert len(test_data) == 55
    assert len(dev_data) + len(test_data) == 105

    for pair in dev_data[:5]:
        assert "resume_text" in pair
        assert "jd_text" in pair
        assert "ground_truth_score" in pair
        assert "ground_truth_category" in pair
