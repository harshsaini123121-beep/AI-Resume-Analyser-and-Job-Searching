"""
Unit tests for information extraction: email, phone, links, sections, and skills.
"""

import pytest
from pathlib import Path
from app.services.extractor import (
    extract_email,
    extract_phone,
    extract_links,
    extract_name,
    detect_sections,
    extract_skills,
    extract_experience_years,
    extract_resume_data
)


def test_extract_contact_info():
    sample = """
    John Developer
    Email: john.dev@company.com
    Phone: (555) 432-8765
    LinkedIn: linkedin.com/in/johndev
    GitHub: github.com/johndev
    """
    assert extract_email(sample) == "john.dev@company.com"
    phone = extract_phone(sample)
    assert phone is not None
    assert "432" in phone
    links = extract_links(sample)
    assert links["linkedin"] is not None
    assert "johndev" in links["linkedin"]
    assert links["github"] is not None


def test_detect_sections():
    sample = """
    PROFESSIONAL SUMMARY
    Experienced engineer.

    CORE SKILLS
    Python, SQL, Docker.

    WORK EXPERIENCE
    Software Engineer at TechCorp.

    EDUCATION
    BS in Computer Science.
    """
    sections = detect_sections(sample)
    assert "summary" in sections
    assert "skills" in sections
    assert "experience" in sections
    assert "education" in sections


def test_extract_skills_and_synonyms():
    sample = "Extensive experience with python, js, k8s, docker, and deep learning."
    skills = extract_skills(sample)
    assert "python" in skills
    # js synonym -> javascript
    assert "javascript" in skills
    # k8s synonym -> kubernetes
    assert "kubernetes" in skills
    assert "docker" in skills
    assert "deep learning" in skills


def test_extract_experience_years():
    sample1 = "Over 6 years of experience in distributed systems."
    assert extract_experience_years(sample1) == 6.0

    sample2 = "Software Engineer 2018 - 2023"
    assert extract_experience_years(sample2) >= 4.0


def test_full_extract_resume_data():
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "alex_morgan_ml_engineer.txt"
    if sample_path.exists():
        text = sample_path.read_text(encoding="utf-8")
        data = extract_resume_data(text)
        assert data["email"] == "alex.morgan@email.com"
        assert "python" in data["skills"]
        assert "pytorch" in data["skills"]
        assert data["experience_years"] >= 4.0
        assert data["word_count"] > 150
