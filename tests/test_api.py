"""
Integration tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_list_jobs_endpoint():
    response = client.get("/api/jobs/list?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert len(data["jobs"]) > 0


def test_job_match_endpoint():
    payload = {
        "resume_text": "Experienced Python and SQL developer with 3 years in machine learning and data pipelines.",
        "resume_skills": ["python", "sql", "machine learning"],
        "candidate_exp_years": 3.0,
        "jd_text": "Looking for a Python developer with SQL experience.",
        "jd_skills": ["python", "sql"],
        "required_exp_years": 2.0
    }
    response = client.post("/api/match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "match_percentage" in data
    assert "verdict" in data
    assert "python" in data["matched_skills"]


def test_job_recommendation_endpoint():
    payload = {
        "resume_text": "Machine Learning Engineer with PyTorch, NLP, and Docker experience.",
        "resume_skills": ["python", "pytorch", "nlp", "docker"],
        "candidate_exp_years": 4.0,
        "top_n": 5
    }
    response = client.post("/api/jobs/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5
