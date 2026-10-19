"""
Unit tests for document parsing and text cleaning services.
"""

import pytest
from pathlib import Path
from app.services.cleaner import clean_text, mask_pii, normalize_for_matching
from app.services.parser import extract_text, extract_text_from_txt, ResumeParsingError


def test_clean_text_normalizes_bullets_and_spaces():
    raw = "• Machine Learning\n\n▪ Deep Learning\n– Python   Development   "
    cleaned = clean_text(raw)
    assert "- Machine Learning" in cleaned
    assert "- Deep Learning" in cleaned
    assert "- Python Development" in cleaned


def test_mask_pii():
    sample = "Contact me at candidate@example.com or call +1 555-123-4567."
    masked = mask_pii(sample)
    assert "candidate@example.com" not in masked
    assert "[EMAIL REDACTED]" in masked
    assert "555-123-4567" not in masked
    assert "[PHONE REDACTED]" in masked


def test_normalize_for_matching():
    text = "Machine-Learning & Deep_Learning!"
    norm = normalize_for_matching(text)
    assert "machine" in norm
    assert "learning" in norm


def test_extract_text_txt():
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "alex_morgan_ml_engineer.txt"
    if sample_path.exists():
        text = extract_text_from_txt(sample_path)
        assert "Alex Morgan" in text
        assert "Machine Learning" in text


def test_extract_text_unsupported_format():
    with pytest.raises(ResumeParsingError):
        extract_text(b"some content", "resume.xyz")


def test_extract_text_pdf():
    sample_pdf = Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "alex_morgan_ml_engineer.pdf"
    if sample_pdf.exists():
        text = extract_text(sample_pdf, "alex_morgan_ml_engineer.pdf")
        assert "Alex Morgan" in text
        assert len(text) > 200


def test_extract_text_docx():
    sample_docx = Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "alex_morgan_ml_engineer.docx"
    if sample_docx.exists():
        text = extract_text(sample_docx, "alex_morgan_ml_engineer.docx")
        assert "Alex Morgan" in text
        assert len(text) > 200

