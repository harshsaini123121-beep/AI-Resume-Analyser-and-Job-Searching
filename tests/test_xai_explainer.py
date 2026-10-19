"""
Unit tests for Explainable AI (XAI) Sentence Evidence Mapping.
"""

from app.services.xai_explainer import explain_match_evidence


def test_explain_match_evidence_finds_mappings():
    resume_text = """
    Alex Morgan | Senior ML Engineer
    - Architected and trained deep learning NLP transformer models in PyTorch.
    - Automated CI/CD MLOps deployment pipelines on AWS Kubernetes with Docker.
    """
    jd_text = """
    We require an engineer proficient in PyTorch for deep learning NLP models.
    Must have hands-on experience deploying MLOps on AWS Kubernetes.
    """
    evidence = explain_match_evidence(resume_text, jd_text, top_n_requirements=2)
    assert len(evidence) > 0
    assert any("Evidence" in item["status"] for item in evidence)
    assert any("PyTorch" in item["resume_evidence"] or "MLOps" in item["resume_evidence"] for item in evidence)
