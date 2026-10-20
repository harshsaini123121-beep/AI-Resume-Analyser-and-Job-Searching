"""
Matching engine combining semantic embeddings (sentence-transformers / TF-IDF)
with structured skill overlap, required vs preferred skill separation,
sliding-window token chunking, and experience weighting.
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
        _SENTENCE_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return _SENTENCE_MODEL
    except Exception as e:
        print(f"[Matcher] Notice: sentence-transformers model initialization deferred: {e}")
        _MODEL_LOAD_FAILED = True
        return None


def is_model_loaded() -> bool:
    """Returns True if the embedding model is loaded in memory."""
    return _SENTENCE_MODEL is not None


def is_tfidf_fallback_active() -> bool:
    """Returns True if TF-IDF fallback is active due to model load failure."""
    return _MODEL_LOAD_FAILED


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    """
    Splits text into overlapping sliding-window chunks (approx 240-280 tokens).
    Prevents truncation on long multi-page resumes and extensive job descriptions.
    """
    words = text.strip().split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Computes semantic cosine similarity (0.0 to 1.0) between two text passages.
    Uses sliding-window token chunking (~256 tokens) with aggregated max/mean similarity
    to capture all sections of long multi-page resumes and extensive job descriptions.
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)

    model = get_embedding_model()
    if model is not None:
        try:
            embs1 = model.encode(chunks1, convert_to_numpy=True)
            embs2 = model.encode(chunks2, convert_to_numpy=True)

            # Cosine similarity matrix between all combinations of chunks
            sim_matrix = cosine_similarity(embs1, embs2)

            # Aggregated score: 65% peak alignment (max chunk match) + 35% global topical fit (mean match)
            max_sim = float(np.max(sim_matrix))
            mean_sim = float(np.mean(sim_matrix))
            aggregated = 0.65 * max_sim + 0.35 * mean_sim
            return float(np.clip(aggregated, 0.0, 1.0))
        except Exception as e:
            print(f"[Matcher] Dense embeddings failed, using TF-IDF fallback: {e}")

    # Fallback: TF-IDF Cosine Similarity with chunking
    try:
        all_chunks = chunks1 + chunks2
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_chunks)
        mat1 = tfidf_matrix[:len(chunks1)]
        mat2 = tfidf_matrix[len(chunks1):]
        sim_matrix = cosine_similarity(mat1, mat2)
        max_sim = float(np.max(sim_matrix))
        mean_sim = float(np.mean(sim_matrix))
        aggregated = 0.65 * max_sim + 0.35 * mean_sim
        return float(np.clip(aggregated, 0.0, 1.0))
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
            sub_skills = [part.strip().lower() for part in s.replace(";", ",").split(",") if part.strip()]
            cleaned_skills.extend(sub_skills)
        if cleaned_skills:
            return sorted(list(set(cleaned_skills)))

    return extract_skills(jd_text)


def compute_match(
    resume_text: str,
    resume_skills: List[str],
    jd_text: str,
    jd_skills: Optional[List[str]] = None,
    jd_preferred_skills: Optional[List[str]] = None,
    candidate_exp_years: float = 0.0,
    required_exp_years: float = 0.0,
    skill_weight: Optional[float] = None,
    semantic_weight: Optional[float] = None,
    exp_weight: Optional[float] = None,
    precomputed_semantic_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Performs hybrid matching between a resume and target job description:
    1. Skill overlap (separating required vs preferred/nice-to-have skills)
    2. Semantic embedding cosine similarity with sliding-window chunking
    3. Experience requirement ratio
    4. Weighted final score and verdict categorization
    """
    w_skill = skill_weight if skill_weight is not None else settings.SKILL_WEIGHT
    w_semantic = semantic_weight if semantic_weight is not None else settings.SEMANTIC_WEIGHT
    w_exp = exp_weight if exp_weight is not None else settings.EXPERIENCE_WEIGHT

    # Normalize skill sets
    resume_skill_set = set(s.strip().lower() for s in resume_skills if s.strip())
    resolved_jd_skills = parse_job_skills(jd_text, jd_skills)
    req_skill_set = set(s.strip().lower() for s in resolved_jd_skills if s.strip())
    pref_skill_set = set(s.strip().lower() for s in (jd_preferred_skills or []) if s.strip())

    # Skill overlaps
    matched_req_skills = sorted(list(resume_skill_set.intersection(req_skill_set)))
    missing_req_skills = sorted(list(req_skill_set - resume_skill_set))

    matched_pref_skills = sorted(list(resume_skill_set.intersection(pref_skill_set)))
    missing_pref_skills = sorted(list(pref_skill_set - resume_skill_set))

    all_target_skills = req_skill_set.union(pref_skill_set)
    matched_skills = sorted(list(resume_skill_set.intersection(all_target_skills)))
    missing_skills = missing_req_skills
    additional_skills = sorted(list(resume_skill_set - all_target_skills))

    low_confidence = False
    if len(req_skill_set) > 0:
        req_ratio = len(matched_req_skills) / len(req_skill_set)
        if len(pref_skill_set) > 0:
            pref_ratio = len(matched_pref_skills) / len(pref_skill_set)
            # 75% weight on required mandatory skills, 25% on preferred/bonus skills
            skill_score = 0.75 * req_ratio + 0.25 * pref_ratio
        else:
            skill_score = req_ratio
    else:
        if len(pref_skill_set) > 0:
            skill_score = len(matched_pref_skills) / len(pref_skill_set)
        else:
            skill_score = 0.0
            low_confidence = True

    # Semantic similarity: use precomputed score if supplied, otherwise compute via chunking
    if precomputed_semantic_score is not None:
        semantic_score = float(np.clip(precomputed_semantic_score, 0.0, 1.0))
    else:
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
        "matched_preferred_skills": matched_pref_skills,
        "missing_preferred_skills": missing_pref_skills,
        "required_skills_count": len(req_skill_set),
        "preferred_skills_count": len(pref_skill_set),
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
