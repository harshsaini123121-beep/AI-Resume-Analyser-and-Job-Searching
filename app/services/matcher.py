"""
Matching engine combining semantic embeddings (sentence-transformers / TF-IDF)
with structured skill overlap and experience weighting.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings
from app.services.cleaner import clean_text
from app.services.extractor import extract_skills

# Singleton model holder
_SENTENCE_MODEL = None
_MODEL_LOAD_FAILED = False


def get_embedding_model():
    """
    Lazy singleton loader for the sentence-transformers model.
    Falls back gracefully if torch/model cannot be initialized.
    """
    global _SENTENCE_MODEL, _MODEL_LOAD_FAILED
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL
    if _MODEL_LOAD_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer
        # Use lightweight, fast, top-performing all-MiniLM-L6-v2
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _SENTENCE_MODEL
    except Exception as e:
        print(f"[Matcher] Notice: sentence-transformers model initialization deferred: {e}")
        _MODEL_LOAD_FAILED = True
        return None


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Computes semantic cosine similarity (0.0 to 1.0) between two text passages.
    Uses sentence-transformers if available; falls back to TF-IDF cosine similarity.
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    model = get_embedding_model()
    if model is not None:
        try:
            embeddings = model.encode([text1, text2], convert_to_numpy=True)
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            # Clip between 0.0 and 1.0
            return float(np.clip(sim, 0.0, 1.0))
        except Exception as e:
            print(f"[Matcher] Embeddings failed, using TF-IDF fallback: {e}")

    # Fallback: TF-IDF Cosine Similarity
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        # Ultimate fallback: Jaccard word set similarity
        words1 = set(clean_text(text1).lower().split())
        words2 = set(clean_text(text2).lower().split())
        if not words1 or not words2:
            return 0.0
        return float(len(words1 & words2) / len(words1 | words2))


def parse_job_skills(jd_text: str, explicit_skills: Optional[List[str]] = None) -> List[str]:
    """
    Resolves required skills for a job description.
    Uses explicit skills if provided; otherwise extracts via the skill taxonomy.
    """
    if explicit_skills and len(explicit_skills) > 0:
        cleaned_skills = []
        for s in explicit_skills:
            # Handle semicolon/comma separated lists within strings
            sub_skills = [part.strip().lower() for part in s.replace(";", ",").split(",") if part.strip()]
            cleaned_skills.extend(sub_skills)
        if cleaned_skills:
            return sorted(list(set(cleaned_skills)))

    # Otherwise extract from description text
    return extract_skills(jd_text)


def compute_match(
    resume_text: str,
    resume_skills: List[str],
    jd_text: str,
    jd_skills: Optional[List[str]] = None,
    candidate_exp_years: float = 0.0,
    required_exp_years: float = 0.0,
    skill_weight: Optional[float] = None,
    semantic_weight: Optional[float] = None,
    exp_weight: Optional[float] = None
) -> Dict[str, Any]:
    """
    Performs hybrid matching between a resume and target job description:
    1. Skill overlap (matched vs missing)
    2. Semantic embedding cosine similarity
    3. Experience requirement ratio
    4. Weighted final score and verdict categorization
    """
    w_skill = skill_weight if skill_weight is not None else settings.SKILL_WEIGHT
    w_semantic = semantic_weight if semantic_weight is not None else settings.SEMANTIC_WEIGHT
    w_exp = exp_weight if exp_weight is not None else settings.EXPERIENCE_WEIGHT

    # Normalize skill lists
    resume_skill_set = set(s.strip().lower() for s in resume_skills if s.strip())
    resolved_jd_skills = parse_job_skills(jd_text, jd_skills)
    jd_skill_set = set(s.strip().lower() for s in resolved_jd_skills if s.strip())

    # Skill overlap
    matched_skills = sorted(list(resume_skill_set.intersection(jd_skill_set)))
    missing_skills = sorted(list(jd_skill_set - resume_skill_set))
    additional_skills = sorted(list(resume_skill_set - jd_skill_set))

    low_confidence = False
    if len(jd_skill_set) > 0:
        skill_score = len(matched_skills) / len(jd_skill_set)
    else:
        # If JD has no detected specific skills, do not invent an arbitrary 0.5 score
        skill_score = 0.0
        low_confidence = True

    # Semantic similarity
    semantic_score = compute_semantic_similarity(resume_text, jd_text)

    # Experience match ratio
    if required_exp_years <= 0.0:
        exp_score = 1.0
    else:
        exp_score = min(1.0, candidate_exp_years / required_exp_years)

    # Weighted final score (0.0 to 1.0)
    if low_confidence:
        denom = w_semantic + w_exp
        if denom > 0:
            final_score_raw = ((w_semantic / denom) * semantic_score) + ((w_exp / denom) * exp_score)
        else:
            final_score_raw = semantic_score
    else:
        final_score_raw = (
            (w_skill * skill_score) +
            (w_semantic * semantic_score) +
            (w_exp * exp_score)
        )

    final_percentage = round(final_score_raw * 100, 1)
    final_percentage = max(0.0, min(100.0, final_percentage))

    # Verdict categorization
    if final_percentage >= 80:
        verdict = "Excellent Match"
        verdict_color = "green"
        recommendation = "You strongly meet or exceed the requirements for this position. Apply immediately!"
    elif final_percentage >= 65:
        verdict = "Good Match"
        verdict_color = "blue"
        recommendation = "Solid foundation with several matching requirements. Address missing skills in your cover letter."
    elif final_percentage >= 45:
        verdict = "Average Match"
        verdict_color = "orange"
        recommendation = "Partial qualification. Consider closing key skill gaps or tailoring your bullet points before applying."
    else:
        verdict = "Weak Match"
        verdict_color = "red"
        recommendation = "Significant skill and background gaps detected for this role. Consider related foundational positions."

    return {
        "match_percentage": final_percentage,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "recommendation": recommendation,
        "skill_score": round(skill_score, 3),
        "semantic_score": round(semantic_score, 3),
        "experience_score": round(exp_score, 3),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_skills,
        "candidate_experience_years": candidate_exp_years,
        "required_experience_years": required_exp_years,
        "low_confidence": low_confidence,
        "weights": {
            "skill": w_skill,
            "semantic": w_semantic,
            "experience": w_exp
        }
    }
