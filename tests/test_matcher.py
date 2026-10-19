"""
Unit tests for the Hybrid Matching Engine.
"""

from pathlib import Path
from app.services.extractor import extract_resume_data
from app.services.matcher import compute_match


def test_matcher_good_alignment():
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "alex_morgan_ml_engineer.txt"
    resume_text = sample_path.read_text(encoding="utf-8")
    data = extract_resume_data(resume_text)

    jd_text = """
    Senior Machine Learning Engineer
    Required Skills: Python, PyTorch, Deep Learning, MLOps, Docker, Kubernetes.
    We need an expert to build LLM pipelines and RAG systems.
    Experience Required: 4 years.
    """
    jd_skills = ["python", "pytorch", "deep learning", "mlops", "docker", "kubernetes"]

    res = compute_match(
        resume_text=resume_text,
        resume_skills=data["skills"],
        jd_text=jd_text,
        jd_skills=jd_skills,
        candidate_exp_years=data["experience_years"],
        required_exp_years=4.0
    )

    assert res["match_percentage"] >= 65.0
    assert "python" in res["matched_skills"]
    assert "pytorch" in res["matched_skills"]
    assert res["verdict"] in ("Excellent Match", "Good Match")


def test_matcher_poor_alignment():
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "incomplete_junior_resume.txt"
    resume_text = sample_path.read_text(encoding="utf-8")
    data = extract_resume_data(resume_text)

    jd_text = """
    Principal Cloud Infrastructure Architect
    Requires 10 years experience with Terraform, AWS, Kubernetes, Helm, Golang.
    """
    jd_skills = ["terraform", "aws", "kubernetes", "helm", "golang"]

    res = compute_match(
        resume_text=resume_text,
        resume_skills=data["skills"],
        jd_text=jd_text,
        jd_skills=jd_skills,
        candidate_exp_years=data["experience_years"],
        required_exp_years=10.0
    )

    assert res["match_percentage"] < 50.0
    assert res["verdict"] in ("Average Match", "Weak Match")
    assert len(res["missing_skills"]) > 0
