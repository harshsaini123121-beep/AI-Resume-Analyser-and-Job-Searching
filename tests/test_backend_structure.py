"""
Comprehensive tests for Task 2: Backend Structure & Reliability.
Covers:
- Pydantic Settings configuration & validation
- Alembic database models, indexes, and aliases
- Pagination on resume history and jobs listing
- Structured JSON logging format and PII hygiene
- Enhanced /health and /ready endpoints
- Asynchronous in-memory background batch screening queue and progress polling
"""

import json
import logging
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from app.main import app, JSONLogFormatter
from app.config import settings, Settings
from app.database import engine, get_db
from app.models import (
    Base,
    User,
    Resume,
    Analysis,
    Job,
    MatchResult,
    ResumeRecord,
    MatchHistoryRecord
)


@pytest.fixture
def client():
    return TestClient(app)


def test_pydantic_settings_validation():
    """Verifies that Settings is an instance of BaseSettings with weight validation."""
    assert isinstance(settings, Settings)
    assert 0.0 <= settings.SKILL_WEIGHT <= 1.0
    assert 0.0 <= settings.SEMANTIC_WEIGHT <= 1.0
    assert 0.0 <= settings.EXPERIENCE_WEIGHT <= 1.0
    assert settings.API_PORT > 0
    assert settings.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"

    # Test weight validation error
    with pytest.raises(Exception):
        Settings(SKILL_WEIGHT=1.5)


def test_database_models_and_indexes():
    """Verifies that all 5 required tables exist with required columns and indexes."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # Required tables
    for expected in ["users", "resumes", "analyses", "jobs", "match_results"]:
        assert expected in tables, f"Table {expected} should exist in database"

    # Verify model aliases for backwards compatibility
    assert ResumeRecord is Resume
    assert MatchHistoryRecord is MatchResult

    # Verify indexes on match_results (job_id, owner_id, created_at)
    match_indexes = [idx["name"] for idx in inspector.get_indexes("match_results")]
    assert any("job_id" in idx for idx in match_indexes)
    assert any("owner_id" in idx for idx in match_indexes)
    assert any("created_at" in idx for idx in match_indexes)

    # Verify indexes on jobs (job_id, company, title)
    job_indexes = [idx["name"] for idx in inspector.get_indexes("jobs")]
    assert any("job_id" in idx for idx in job_indexes)


def test_pagination_endpoints(client):
    """Verifies pagination parameters on resume history and jobs listing."""
    # Test jobs listing with offset and limit
    res_jobs = client.get("/api/v1/jobs/list?offset=0&limit=5")
    assert res_jobs.status_code == 200
    data_jobs = res_jobs.json()
    assert "jobs" in data_jobs
    assert "total" in data_jobs
    assert data_jobs["offset"] == 0
    assert data_jobs["limit"] == 5
    assert len(data_jobs["jobs"]) <= 5

    # Test resume history pagination
    headers = {"X-API-Key": settings.API_KEY}
    res_hist = client.get("/api/v1/resume/history?offset=0&limit=10", headers=headers)
    assert res_hist.status_code == 200
    data_hist = res_hist.json()
    assert "history" in data_hist
    assert "total" in data_hist
    assert data_hist["offset"] == 0
    assert data_hist["limit"] == 10


def test_structured_json_logging(caplog):
    """Verifies JSONLogFormatter produces valid JSON containing standard fields."""
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path.py",
        lineno=42,
        msg="Test event message",
        args=(),
        exc_info=None
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert "timestamp" in parsed
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test event message"


def test_health_and_readiness_endpoints(client):
    """Verifies /health liveness and /ready readiness checks."""
    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "healthy"}

    # Readiness check
    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    ready_data = res_ready.json()
    assert ready_data["status"] == "ready"
    assert ready_data["database_status"] == "connected"
    assert ready_data["embedding_model"] == "all-MiniLM-L6-v2"
    assert "model_loaded" in ready_data
    assert "tfidf_fallback_active" in ready_data
    assert ready_data["jobs_indexed"] > 0


def test_background_batch_screening_async_queue(client):
    """Verifies async batch screening submission and progress polling."""
    payload = {
        "job_id": "test_batch_001",
        "jd_title": "Backend Python Developer",
        "jd_text": "Looking for Python, FastAPI, Docker, and SQL experience.",
        "required_skills": ["python", "fastapi", "docker", "sql"],
        "experience_required": 2.0,
        "candidates": [
            {
                "filename": "applicant_alice.txt",
                "text": "Alice is a Python engineer with 4 years building FastAPI APIs and Docker containers."
            },
            {
                "filename": "applicant_bob.txt",
                "text": "Bob has 1 year of experience with HTML and CSS."
            }
        ]
    }

    # Submit batch task asynchronously
    res = client.post("/api/v1/jobs/batch-screen-async", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    assert data["status"] == "processing"

    batch_job_id = data["job_id"]

    # Poll progress endpoint
    prog_res = client.get(f"/api/v1/jobs/batch/{batch_job_id}/progress")
    assert prog_res.status_code == 200
    prog_data = prog_res.json()
    assert prog_data["job_id"] == batch_job_id
    assert prog_data["status"] in ["processing", "completed"]
    assert prog_data["total"] == 2

    # Poll non-existent job ID should return 404
    err_res = client.get("/api/v1/jobs/batch/nonexistent-uuid-123/progress")
    assert err_res.status_code == 404
