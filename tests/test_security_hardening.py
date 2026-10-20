"""
Security and Privacy Hardening Regression Tests.
Verifies Task 3 security controls:
- File magic bytes validation
- Filename sanitization
- X-API-Key authentication on history and delete endpoints
- Caller-scoped history retrieval
- DELETE record endpoint
- Generic error handling without exception leakage
- Delimited prompt structure and Pydantic fallback for LLM coaching
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.database import Base, engine, get_db, SessionLocal, init_db
from app.models import ResumeRecord
from app.services.parser import validate_magic_bytes, sanitize_filename, ResumeParsingError
from app.services.llm_feedback import get_ai_feedback, generate_rule_based_feedback


@pytest.fixture(autouse=True)
def setup_test_db():
    init_db(bind_engine=engine)
    yield
    # Clean up test records
    db = SessionLocal()
    try:
        db.query(ResumeRecord).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture
def client():
    return TestClient(app)


def test_magic_bytes_rejection_pe_exe():
    """Executable renamed to .pdf or .txt must be rejected."""
    fake_exe_bytes = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00"
    
    with pytest.raises(ResumeParsingError, match="Invalid PDF header signature"):
        validate_magic_bytes(fake_exe_bytes, "pdf")

    with pytest.raises(ResumeParsingError, match="Binary executable content detected"):
        validate_magic_bytes(fake_exe_bytes, "txt")


def test_magic_bytes_valid_pdf_and_txt():
    """Valid PDF magic bytes and valid TXT content pass validation."""
    valid_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj"
    validate_magic_bytes(valid_pdf_bytes, "pdf")

    valid_txt_bytes = "Experienced software engineer with Python skills.".encode("utf-8")
    validate_magic_bytes(valid_txt_bytes, "txt")


def test_sanitize_filename_traversal_and_injection():
    """Path traversal and dangerous characters must be safely stripped."""
    u1, name1 = sanitize_filename("../../../etc/passwd")
    assert name1 == "passwd"
    assert len(u1) == 36

    u2, name2 = sanitize_filename("..\\..\\windows\\system32\\cmd.exe")
    assert name2 == "cmd.exe"

    u3, name3 = sanitize_filename("<script>alert(1)</script>resume.pdf")
    assert "<" not in name3 and ">" not in name3 and "resume.pdf" in name3

    u4, name4 = sanitize_filename("my resume (1) [final].pdf")
    assert "my_resume" in name4 or "my resume" in name4

    u5, name5 = sanitize_filename("   ")
    assert name5 == "resume.pdf"


def test_history_unauthenticated_fails(client):
    """Calling /api/resume/history without X-API-Key must return 401."""
    res = client.get("/api/resume/history")
    assert res.status_code == 401
    assert "Unauthorized" in res.json().get("detail", "")


def test_history_authenticated_scopes_by_caller(client):
    """Authenticated caller only retrieves their own records."""
    db = SessionLocal()
    try:
        rec1 = ResumeRecord(
            candidate_name="Alice Candidate",
            filename="alice.pdf",
            ats_score=85,
            ats_grade="A",
            owner_id="api_client_default"
        )
        rec2 = ResumeRecord(
            candidate_name="Bob Candidate",
            filename="bob.pdf",
            ats_score=70,
            ats_grade="B",
            owner_id="another_client"
        )
        db.add_all([rec1, rec2])
        db.commit()
    finally:
        db.close()

    headers = {"X-API-Key": settings.API_KEY}
    res = client.get("/api/resume/history", headers=headers)
    assert res.status_code == 200
    data = res.json()
    items = data["history"]
    assert len(items) == 1
    assert items[0]["candidate_name"] == "Alice Candidate"
    assert items[0]["filename"] == "alice.pdf"


def test_delete_resume_endpoint(client):
    """DELETE /api/resume/{record_id} deletes the record or returns 404."""
    db = SessionLocal()
    try:
        rec = ResumeRecord(
            candidate_name="Charlie Test",
            filename="charlie.pdf",
            ats_score=90,
            ats_grade="A+",
            owner_id="api_client_default"
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        rec_id = rec.id
    finally:
        db.close()

    headers = {"X-API-Key": settings.API_KEY}
    # Unauthenticated delete fails
    res_unauth = client.delete(f"/api/resume/{rec_id}")
    assert res_unauth.status_code == 401

    # Authenticated delete succeeds
    res_del = client.delete(f"/api/resume/{rec_id}", headers=headers)
    assert res_del.status_code == 200
    assert res_del.json().get("status") == "deleted"

    # Subsequent delete returns 404
    res_not_found = client.delete(f"/api/resume/{rec_id}", headers=headers)
    assert res_not_found.status_code == 404


def test_global_exception_handler_returns_request_id_without_leak():
    """Simulated internal error returns 500 with request_id and no raw traceback leak."""
    from unittest.mock import patch

    unhandled_client = TestClient(app, raise_server_exceptions=False)
    with patch("app.routes.jobs.recommend_jobs", side_effect=RuntimeError("Secret internal database password xyz")):
        req_body = {
            "resume_text": "Sample text",
            "resume_skills": ["python"],
            "candidate_exp_years": 2.0,
            "top_n": 5
        }
        res = unhandled_client.post("/api/jobs/recommend", json=req_body)
        assert res.status_code == 500
        data = res.json()
        assert data.get("error") == "InternalServerError"
        assert "request_id" in data
        assert len(data["request_id"]) > 0
        # Ensure raw internal details are NOT leaked to the client
        assert "Secret internal database password" not in res.text


def test_llm_feedback_fallback_and_delimiters():
    """Verify rule-based fallback succeeds gracefully without external API keys."""
    feedback = get_ai_feedback(
        resume_text="Senior Python Developer with 5 years experience.",
        jd_text="Looking for a Python Developer proficient in FastAPI and Docker.",
        matched_skills=["python"],
        missing_skills=["fastapi", "docker"],
        ats_score=80,
        api_key=""
    )
    assert feedback["source"] == "rule_based_fallback"
    assert "summary" in feedback
    assert len(feedback["strengths"]) > 0
    assert len(feedback["weaknesses"]) > 0
    assert len(feedback["rewritten_bullets"]) > 0
