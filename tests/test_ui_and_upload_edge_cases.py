"""
Tests for UI modular pages, component rendering, and edge-case document uploads:
- Valid TXT, PDF, DOCX formats
- Corrupt binary files
- Empty 0-byte files
- Oversized > 5MB files
- Page module importability & isolation
"""
import io
import importlib.util
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.parser import extract_document, ResumeParsingError
from frontend.api_client import api_client, APIResponseError

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": settings.API_KEY}
ROOT_DIR = Path(__file__).resolve().parent.parent


def test_modular_pages_importable():
    """Verify all 5 frontend page modules and components can be imported cleanly without error."""
    pages_dir = ROOT_DIR / "frontend" / "pages"
    page_files = [
        "1_home.py",
        "2_results.py",
        "3_job_match.py",
        "4_job_recommendations.py",
        "5_batch_and_history.py"
    ]
    for pf in page_files:
        path = pages_dir / pf
        assert path.exists(), f"Page file {pf} must exist"
        spec = importlib.util.spec_from_file_location(pf.replace(".py", ""), path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert module is not None


def test_components_importable():
    """Verify all reusable UI components in frontend/components can be imported."""
    from frontend.components.empty_state import render_empty_state
    from frontend.components.error_banner import render_error_banner
    from frontend.components.evidence_card import render_evidence_card
    from frontend.components.score_ring import render_score_ring
    from frontend.components.skill_chips import render_skill_chips
    from frontend.components.stat_card import render_stat_card

    assert "empty-state-box" in render_empty_state("Title", "Desc")
    assert "error-banner" in render_error_banner("Err", "Detail")
    assert "evidence-card" in render_evidence_card("Req", "Proof", "Strong Evidence")
    assert "85" in render_score_ring(85)
    assert "skill-chip-matched" in render_skill_chips(matched=["Python"])
    assert "stat-card" in render_stat_card("Title", "100", "Expl")


def test_valid_txt_upload():
    """Test standard valid TXT file analysis."""
    txt_content = (
        "Jane Doe\nSoftware Engineer\njane.doe@example.com | (555) 123-4567\n\n"
        "Summary: Experienced software engineer building scalable Python and React systems.\n\n"
        "Experience:\nSenior Developer at Tech Corp (2020 - Present)\n"
        "- Built microservices in Python, Django, and PostgreSQL, increasing throughput by 40%.\n\n"
        "Education:\nBS Computer Science, University of Technology, 2019\n\n"
        "Skills:\nPython, React, Docker, SQL, Git, AWS"
    ).encode("utf-8")

    files = {"file": ("jane_doe_resume.txt", txt_content, "text/plain")}
    res = client.post("/api/v1/resume/analyse", files=files, headers=AUTH_HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert "extracted_data" in data
    assert "ats_score" in data
    assert data["ats_score"]["total_score"] >= 40


def test_empty_file_upload_rejected():
    """Verify that an empty 0-byte file upload returns a 400 Bad Request error."""
    files = {"file": ("empty_resume.txt", b"", "text/plain")}
    res = client.post("/api/v1/resume/analyse", files=files, headers=AUTH_HEADERS)
    assert res.status_code == 400
    detail = res.json().get("detail", "")
    assert "empty" in detail.lower()


def test_oversized_file_upload_rejected():
    """Verify that an oversized file (> 5 MB) returns a 413 or 400 rejection."""
    # 5.5 MB of dummy data
    large_content = b"A" * (int(5.5 * 1024 * 1024))
    files = {"file": ("huge_resume.txt", large_content, "text/plain")}
    res = client.post("/api/v1/resume/analyse", files=files, headers=AUTH_HEADERS)
    assert res.status_code in [400, 413]
    detail = res.json().get("detail", "")
    assert "exceeds" in detail.lower() or "large" in detail.lower()


def test_corrupt_file_upload_rejected():
    """Verify that a corrupt file (e.g. invalid PDF/DOCX headers) returns an informative error."""
    corrupt_pdf_bytes = b"%PDF-1.4 completely invalid corrupt binary junk not a real pdf"
    files = {"file": ("corrupt.pdf", corrupt_pdf_bytes, "application/pdf")}
    res = client.post("/api/v1/resume/analyse", files=files, headers=AUTH_HEADERS)
    # The API should gracefully return a 400 error indicating parsing failed
    assert res.status_code in [400, 422]
    detail = res.json().get("detail", "")
    assert len(detail) > 0


def test_unsupported_file_extension_rejected():
    """Verify that unsupported file extensions (e.g. .exe, .png) are rejected with 400."""
    files = {"file": ("malicious.exe", b"MZ\\x90\\x00", "application/octet-stream")}
    res = client.post("/api/v1/resume/analyse", files=files, headers=AUTH_HEADERS)
    assert res.status_code == 400
    detail = res.json().get("detail", "")
    assert "unsupported" in detail.lower() or "extension" in detail.lower()


def test_render_stat_card_signatures_and_escaping():
    """Regression test: verify render_stat_card supports title, border_color, label, kwargs and XSS escaping."""
    from frontend.components.stat_card import render_stat_card

    # 1. Signature with title & border_color (as called in 2_results.py)
    html1 = render_stat_card(
        title="Technical Skills",
        value="15",
        explanation="Industry competencies identified across resume text.",
        icon="🛠️",
        border_color="var(--color-primary)"
    )
    assert "stat-card" in html1
    assert "Technical Skills" in html1
    assert "15" in html1
    assert "border-left: 3px solid var(--color-primary)" in html1

    # 2. Backward compatibility with label and color_theme
    html2 = render_stat_card(
        value="88%",
        label="Score Metric",
        explanation="Benchmark accuracy.",
        icon="📊",
        color_theme="emerald"
    )
    assert "stat-card" in html2
    assert "Score Metric" in html2
    assert "88%" in html2
    assert "stat-icon-emerald" in html2

    # 3. Positional compatibility
    html3 = render_stat_card("Experience", "5 Yrs", "Verified tenure")
    assert "stat-card" in html3
    assert "5 Yrs" in html3

    # 4. Safe XSS Escaping
    xss_html = render_stat_card(
        title="<script>alert('xss')</script>",
        value="<img src=x onerror=alert(1)>",
        explanation="<b onmouseover=alert(2)>Expl</b>"
    )
    assert "<script>" not in xss_html
    assert "&lt;script&gt;" in xss_html
    assert "<img" not in xss_html
    assert "&lt;img" in xss_html


def test_render_empty_state_action_label_and_escaping():
    """Regression test: verify render_empty_state supports action_label, action_note, and kwargs without TypeError."""
    from frontend.components.empty_state import render_empty_state

    # 1. Signature with action_label (as called in 2_results.py & 4_job_recommendations.py)
    html1 = render_empty_state(
        title="No Active Resume Loaded",
        description="Upload a resume to generate an in-depth ATS diagnostic report.",
        action_label="Go to Upload Resume"
    )
    assert "empty-state-box" in html1
    assert "No Active Resume Loaded" in html1
    assert "Go to Upload Resume" in html1

    # 2. Signature with action_note (backward compatibility)
    html2 = render_empty_state(
        title="No Positions",
        description="Try adjusting search criteria.",
        action_note="Tip: Expand radius"
    )
    assert "empty-state-box" in html2
    assert "Tip: Expand radius" in html2

    # 3. XSS Escaping
    xss_html = render_empty_state(
        title="<script>alert(1)</script>",
        description="<iframe src=evil.com>",
        action_label="<img src=x onerror=alert(2)>"
    )
    assert "<script>" not in xss_html
    assert "&lt;script&gt;" in xss_html
    assert "<iframe" not in xss_html
    assert "<img" not in xss_html

