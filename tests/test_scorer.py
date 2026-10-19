"""
Unit tests for the 100-point ATS Scoring Engine.
"""

from pathlib import Path
from app.services.extractor import extract_resume_data
from app.services.scorer import calculate_ats_score


def test_scorer_strong_resume():
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "alex_morgan_ml_engineer.txt"
    text = sample_path.read_text(encoding="utf-8")
    data = extract_resume_data(text)
    result = calculate_ats_score(text, data)

    assert result["total_score"] >= 80
    assert "A" in result["grade"]
    assert "Contact Info" in result["category_breakdown"]
    assert "Sections Structure" in result["category_breakdown"]
    assert len(result["strengths"]) > 0


def test_scorer_incomplete_resume():
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "incomplete_junior_resume.txt"
    text = sample_path.read_text(encoding="utf-8")
    data = extract_resume_data(text)
    result = calculate_ats_score(text, data)

    # Incomplete resume lacks sections, numbers, phone, LinkedIn, etc.
    assert result["total_score"] < 60
    assert len(result["issues"]) >= 2
    assert len(result["suggestions"]) >= 2
