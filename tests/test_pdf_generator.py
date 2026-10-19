"""
Unit tests for PDF ATS Audit Report Generation.
"""

from app.services.pdf_generator import generate_pdf_report


def test_generate_pdf_report_bytes():
    category_scores = {
        "Contact Info": {"score": 10, "max_score": 10, "details": "All contact info present"},
        "Skills Density": {"score": 15, "max_score": 15, "details": "14 skills detected"}
    }
    pdf_bytes = generate_pdf_report(
        candidate_name="Alex Morgan",
        email="alex@example.com",
        phone="+1 555-123-4567",
        ats_score=92,
        ats_grade="A+ (ATS Ready)",
        category_breakdown=category_scores,
        strengths=["Strong technical depth in Python and PyTorch"],
        issues=[],
        suggestions=["Target senior architect roles"],
        skills=["python", "pytorch", "fastapi", "docker"]
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
