"""
Integration tests for Task 1 (Part 2):
- Verification of new /api/v1 endpoints:
  - POST /api/v1/resume/bullet-optimize
  - POST /api/v1/match/xai-evidence
  - POST /api/v1/resume/report-pdf
  - POST /api/v1/jobs/batch-screen
  - GET /api/v1/resume/skills-taxonomy
  - POST /api/v1/jobs/extract-skills
- Deprecation header verification on legacy /api/* endpoints
- ResumeLabClient verification and offline error handling
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from frontend.api_client import ResumeLabClient, APIConnectionError, APIResponseError

client = TestClient(app)


def test_bullet_optimize_endpoint():
    """Verifies Google X-Y-Z bullet optimization endpoint."""
    payload = {
        "bullet_text": "Responsible for managing web servers and helping developers write code."
    }
    response = client.post("/api/v1/resume/bullet-optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "grade" in data
    assert "improved_rewrite" in data
    assert "feedback" in data
    assert isinstance(data["feedback"], list)


def test_xai_evidence_endpoint():
    """Verifies XAI sentence-level evidence mapping endpoint."""
    payload = {
        "resume_text": (
            "Alex Smith\n"
            "Software Engineer with 4 years experience.\n"
            "Architected distributed microservices in Go and Kubernetes, scaling to 10M requests daily.\n"
            "Optimized PostgreSQL database queries, reducing latency by 45%."
        ),
        "jd_text": (
            "Senior Backend Engineer\n"
            "Requirements:\n"
            "- Must have experience designing microservices using Kubernetes.\n"
            "- Strong database optimization experience with SQL."
        ),
        "top_n_requirements": 3
    }
    response = client.post("/api/v1/match/xai-evidence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "evidence" in data
    assert len(data["evidence"]) > 0
    first = data["evidence"][0]
    assert "job_requirement" in first
    assert "resume_evidence" in first
    assert "status" in first


def test_pdf_report_endpoint():
    """Verifies binary PDF ATS audit report generation endpoint."""
    payload = {
        "candidate_name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+1 555-0192",
        "ats_score": 88,
        "ats_grade": "A",
        "category_breakdown": {
            "contact_info": {"score": 10, "max_score": 10, "details": "Complete"},
            "skills_coverage": {"score": 25, "max_score": 30, "details": "Strong"}
        },
        "strengths": ["Clear contact details", "Relevant skills"],
        "issues": ["Add more metrics"],
        "suggestions": ["Include quantifiable results"],
        "skills": ["python", "docker", "fastapi"]
    }
    response = client.post("/api/v1/resume/report-pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_batch_screening_endpoint():
    """Verifies recruiter batch screening endpoint."""
    payload = {
        "job_id": "1",
        "jd_title": "Machine Learning Engineer",
        "jd_text": "Looking for a Python and PyTorch ML Engineer with 3+ years experience.",
        "required_skills": ["python", "pytorch", "machine learning"],
        "experience_required": 3.0,
        "candidates": [
            {
                "filename": "candidate1.txt",
                "text": "Alice Doe\nSkills: Python, PyTorch, ML\nExperience: ML Engineer 2020 - 2024"
            },
            {
                "filename": "candidate2.txt",
                "text": "Bob Ross\nSkills: Graphic Design, Photoshop\nExperience: Artist 2021 - 2023"
            }
        ]
    }
    response = client.post("/api/v1/jobs/batch-screen", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["candidates_screened"] == 2
    assert len(data["results"]) == 2
    # Alice should rank above Bob
    assert data["results"][0]["filename"] == "candidate1.txt"
    assert data["results"][0]["match_percentage"] > data["results"][1]["match_percentage"]


def test_skills_taxonomy_and_extract_endpoints():
    """Verifies skills taxonomy and quick skill extraction endpoints."""
    # Taxonomy
    tax_resp = client.get("/api/v1/resume/skills-taxonomy")
    assert tax_resp.status_code == 200
    assert "categories" in tax_resp.json()

    # Extract
    ext_resp = client.post("/api/jobs/extract-skills", json={"text": "Seeking a Senior Python and Docker engineer with 5 years experience."})
    assert ext_resp.status_code == 200
    data = ext_resp.json()
    assert "python" in data["skills"]
    assert data["experience_years"] >= 5.0


def test_legacy_api_deprecation_headers():
    """Verifies that calling legacy /api/* returns Deprecation and Link headers."""
    # Legacy health
    resp = client.get("/health")
    assert "Deprecation" not in resp.headers

    # Legacy unversioned /api/jobs/list
    legacy_resp = client.get("/api/jobs/list?limit=5")
    assert legacy_resp.status_code == 200
    assert legacy_resp.headers.get("Deprecation") == "true"
    assert "Link" in legacy_resp.headers
    assert "/api/v1/jobs/list" in legacy_resp.headers["Link"]

    # Canonical v1 /api/v1/jobs/list does NOT have deprecation header
    v1_resp = client.get("/api/v1/jobs/list?limit=5")
    assert v1_resp.status_code == 200
    assert "Deprecation" not in v1_resp.headers


def test_api_client_offline_handling():
    """Verifies that ResumeLabClient detects offline backend gracefully."""
    # Client pointing to an inactive port
    dead_client = ResumeLabClient(base_url="http://127.0.0.1:59998", timeout=0.5, max_retries=0)
    assert dead_client.is_healthy() is False

    with pytest.raises(APIConnectionError):
        dead_client.list_jobs()
