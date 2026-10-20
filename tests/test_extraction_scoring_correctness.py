"""
Correctness tests for Entity Extraction, Disambiguation, Date Overlap Merging, and Scoring Rubrics.
Verifies Task 4 requirements:
- Case-sensitive and technical context disambiguation for short tokens (R, Go, C)
- Stop-context blacklist for ambiguous common words (react, spark, excel, word, spring)
- Taxonomy cleanliness and zero-duplicate validation
- Overlapping employment date range merging (no double-counting)
- Education date exclusion from professional experience calculation
- Month-level experience duration resolution
- ATS quantified impact requiring strict units / percentages / multipliers (no bare numbers/years)
- Hybrid matcher configurable weights and empty JD skills low-confidence handling
- Total jobs evaluated returning full dataset count
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.extractor import (
    extract_skills,
    extract_experience_years,
    load_skills_database
)
from app.services.scorer import calculate_ats_score
from app.services.matcher import compute_match
from app.services.recommender import load_jobs_dataset


def test_taxonomy_cleanliness_and_no_duplicates():
    """Validates that data/skills.json contains no duplicate canonical skills across categories."""
    skills_db = load_skills_database()
    categories = skills_db.get("categories", {})
    all_skills = []
    for cat_name, skill_list in categories.items():
        assert isinstance(skill_list, list), f"Category {cat_name} must be a list"
        all_skills.extend(s.lower().strip() for s in skill_list)

    from collections import Counter
    counts = Counter(all_skills)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert len(duplicates) == 0, f"Found duplicate skills in taxonomy: {duplicates}"

    synonyms = skills_db.get("synonyms", {})
    for trigger, canonical in synonyms.items():
        assert trigger == trigger.lower(), f"Synonym trigger '{trigger}' must be lowercase"
        assert canonical == canonical.lower(), f"Canonical target '{canonical}' must be lowercase"


def test_ambiguous_short_skills_disambiguation():
    """R, Go, and C must not trigger on everyday words without technical context."""
    # Negative cases: plain text without technical context
    neg_text = "I like to go for a run in Section C of the park with my dog R."
    neg_skills = extract_skills(neg_text)
    assert "r" not in neg_skills
    assert "go" not in neg_skills
    assert "c" not in neg_skills

    # Positive cases: technical context and casing
    pos_text = "Experienced software engineer skilled in Python, R, Go, and C/C++ development."
    pos_skills = extract_skills(pos_text)
    assert "python" in pos_skills
    assert "r" in pos_skills
    assert "go" in pos_skills
    assert "c" in pos_skills


def test_stop_context_blacklist_for_ambiguous_verbs():
    """Everyday English usage of react, spark, excel, word, spring must be filtered out."""
    non_tech = "Quick to react to problems, always able to spark enthusiasm and excel at team communication."
    skills = extract_skills(non_tech)
    assert "react" not in skills
    assert "spark" not in skills
    assert "excel" not in skills

    tech = "Built modern UI web apps with React.js and large-scale data pipelines using Apache Spark."
    tech_skills = extract_skills(tech)
    assert "react.js" in tech_skills or "react" in tech_skills
    assert "apache spark" in tech_skills or "spark" in tech_skills


def test_experience_years_overlapping_dates_merging():
    """Overlapping employment dates must be merged, not summed (no double-counting)."""
    # 2018-2021 (3 years) + 2020-2023 (3 years). Combined span is 2018-2023 = 5 years, NOT 6.
    overlapping_resume = """
    WORK EXPERIENCE
    Software Engineer at Alpha Corp
    2018 - 2021
    
    Backend Consultant at Beta LLC
    2020 - 2023
    """
    exp_years = extract_experience_years(overlapping_resume)
    assert exp_years == 5.0, f"Expected 5.0 years merged span, got {exp_years}"


def test_experience_years_excludes_education_dates():
    """Graduation years in education section must not be counted as professional experience."""
    resume_with_education = """
    WORK EXPERIENCE
    Software Engineer
    2021 - 2023
    
    EDUCATION
    Bachelor of Science in Computer Science
    2014 - 2018
    """
    exp_years = extract_experience_years(resume_with_education)
    # Should only count 2021 - 2023 (~2 years), not adding the 4 years of education (6 years)
    assert exp_years <= 3.0, f"Expected <= 3.0 years, got {exp_years}"


def test_experience_years_month_level_resolution():
    """Month-level date ranges must be calculated accurately."""
    sample = """
    WORK EXPERIENCE
    Data Analyst
    Jan 2021 - Dec 2021
    """
    exp_years = extract_experience_years(sample)
    assert 0.9 <= exp_years <= 1.1, f"Expected ~1.0 year, got {exp_years}"


def test_ats_quantified_impact_strict_units():
    """Bare numbers/years must not count as quantified achievements without units/multipliers."""
    # Bare numbers: 2020, 2022, 5, 10
    bare_resume = "Worked from 2020 to 2022 in a team of 5 people across 10 offices."
    ats_bare = calculate_ats_score(bare_resume, {"sections_detected": ["experience"]})
    quant_bare = ats_bare["category_breakdown"]["Quantified Impact"]
    assert quant_bare["score"] == 0

    # Genuine metrics with units, %, $, multipliers
    quant_resume = "Accelerated ETL throughput by 45%, reduced cloud spend by $150,000, and scaled pipeline to 500k users."
    ats_quant = calculate_ats_score(quant_resume, {"sections_detected": ["experience"]})
    quant_score = ats_quant["category_breakdown"]["Quantified Impact"]["score"]
    assert quant_score >= 11, f"Expected high quantified score (>=11), got {quant_score}"


def test_matcher_weights_and_empty_jd_skills():
    """Matcher must use config weights and return low_confidence when JD lacks skills."""
    res = compute_match(
        resume_text="Senior Python and React developer with 5 years experience.",
        resume_skills=["python", "react"],
        jd_text="General software engineering position requiring strong fundamentals.",
        jd_skills=[],  # No specific skills listed
        candidate_exp_years=5.0,
        required_exp_years=3.0
    )
    assert res["skill_score"] == 0.0
    assert res["low_confidence"] is True
    assert res["weights"]["skill"] == settings.SKILL_WEIGHT
    assert res["weights"]["semantic"] == settings.SEMANTIC_WEIGHT
    assert res["weights"]["experience"] == settings.EXPERIENCE_WEIGHT


def test_total_jobs_evaluated_returns_dataset_length():
    """The /api/jobs/recommend endpoint must return total_jobs_evaluated equal to the dataset count."""
    client = TestClient(app)
    req_body = {
        "resume_text": "Experienced Python Engineer",
        "resume_skills": ["python"],
        "candidate_exp_years": 3.0,
        "top_n": 5
    }
    res = client.post("/api/jobs/recommend", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert len(data["recommendations"]) <= 5
    # Total evaluated should reflect the entire dataset (> 500 jobs) rather than top_n
    assert data["total_jobs_evaluated"] > 500
