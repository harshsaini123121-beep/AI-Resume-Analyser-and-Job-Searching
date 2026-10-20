"""
Integration tests for FastAPI endpoints.
"""

import io
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

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


def test_readiness_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["skills_indexed"] > 0
    assert data["jobs_indexed"] > 0


def test_resume_upload_endpoint():
    resume_bytes = (
        "Jane Doe\n"
        "Email: jane@example.com\n"
        "SKILLS: Python, SQL, Docker\n"
        "EXPERIENCE: Software Engineer 2020 - 2023\n"
    ).encode("utf-8")
    files = {"file": ("jane_resume.txt", io.BytesIO(resume_bytes), "text/plain")}
    response = client.post("/api/resume/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "jane_resume.txt"
    assert "extracted_data" in data
    assert "python" in data["extracted_data"]["skills"]


def test_resume_analyse_endpoint():
    resume_bytes = (
        "Alice Engineer\n"
        "Email: alice@engineer.com\n"
        "Phone: (555) 019-2831\n"
        "PROFESSIONAL SUMMARY\n"
        "Cloud engineer with 4 years experience.\n"
        "CORE SKILLS\n"
        "Python, Docker, Kubernetes, AWS\n"
        "EXPERIENCE\n"
        "DevOps Engineer 2020 - Present\n"
        "- Accelerated deploy cycles by 40% using Docker.\n"
    ).encode("utf-8")
    files = {"file": ("alice_resume.txt", io.BytesIO(resume_bytes), "text/plain")}
    headers = {"X-API-Key": settings.API_KEY}
    response = client.post("/api/resume/analyse", files=files, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "alice_resume.txt"
    assert "extracted_data" in data
    assert "ats_score" in data
    assert data["ats_score"]["total_score"] > 0
    assert data["id"] is not None


def test_resume_upload_magic_bytes_failure():
    # Executable bytes disguised as PDF
    fake_exe = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00"
    files = {"file": ("malicious.pdf", io.BytesIO(fake_exe), "application/pdf")}
    response = client.post("/api/resume/upload", files=files)
    assert response.status_code == 400
    assert "header signature" in response.json().get("detail", "").lower()


def test_resume_upload_empty_file():
    empty_bytes = b""
    files = {"file": ("empty.txt", io.BytesIO(empty_bytes), "text/plain")}
    response = client.post("/api/resume/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json().get("detail", "").lower()

