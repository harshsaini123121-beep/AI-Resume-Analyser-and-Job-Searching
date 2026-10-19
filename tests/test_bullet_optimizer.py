"""
Unit tests for the Google X-Y-Z bullet point optimizer.
"""

from app.services.bullet_optimizer import analyze_bullet_point


def test_analyze_strong_bullet():
    strong = "Architected and deployed high-throughput FastAPI microservices, reducing 99th percentile response latency by 45% across 20M daily queries."
    result = analyze_bullet_point(strong)
    assert result["score"] >= 80
    assert result["has_action_verb"] is True
    assert result["has_metrics"] is True
    assert "Strong" in result["grade"]


def test_analyze_weak_passive_bullet():
    weak = "Responsible for helping out with some coding and fixing occasional bugs in the system."
    result = analyze_bullet_point(weak)
    assert result["score"] < 60
    assert result["has_metrics"] is False
    assert len(result["improved_rewrite"]) > 0
