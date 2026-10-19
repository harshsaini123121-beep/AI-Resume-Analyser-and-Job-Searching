"""
Explainable AI (XAI) service: Sentence-level evidence mapping between job requirements
and resume sentences. Provides transparent reasoning behind match scores.
"""

import re
from typing import Dict, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.cleaner import clean_text


def split_into_sentences(text: str) -> List[str]:
    """Splits text into meaningful sentences or bullet points."""
    # Split by newlines and common sentence boundaries
    raw_lines = text.splitlines()
    sentences = []
    for line in raw_lines:
        line_clean = line.strip().lstrip("-•*▪–— ")
        if len(line_clean) > 20:
            # Check for multiple sentences inside one line
            sub_sentences = re.split(r"(?<=[.!?])\s+", line_clean)
            for s in sub_sentences:
                s_strip = s.strip()
                if len(s_strip) > 20 and not s_strip.isupper():
                    sentences.append(s_strip)
    return sentences


def explain_match_evidence(resume_text: str, jd_text: str, top_n_requirements: int = 6) -> List[Dict[str, Any]]:
    """
    Finds the exact evidence sentences in the candidate's resume that substantiate
    each core requirement in the job description using cosine similarity.
    """
    resume_sentences = split_into_sentences(resume_text)
    jd_sentences = split_into_sentences(jd_text)

    if not resume_sentences or not jd_sentences:
        return []

    # Filter JD sentences that look like requirements
    requirement_cues = ["experience", "proficient", "strong", "knowledge", "skills", "must", "required", "responsible", "design", "build", "lead"]
    prioritized_jd = []
    for s in jd_sentences:
        if any(cue in s.lower() for cue in requirement_cues):
            prioritized_jd.append(s)

    if not prioritized_jd:
        prioritized_jd = jd_sentences[:top_n_requirements]
    else:
        prioritized_jd = prioritized_jd[:top_n_requirements]

    # Compute sentence-to-sentence cosine similarity matrix
    all_corpus = prioritized_jd + resume_sentences
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(all_corpus)

    jd_matrix = matrix[:len(prioritized_jd)]
    res_matrix = matrix[len(prioritized_jd):]

    sim_matrix = cosine_similarity(jd_matrix, res_matrix)

    evidence_mappings = []

    for i, req in enumerate(prioritized_jd):
        best_match_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_match_idx])
        best_sentence = resume_sentences[best_match_idx]

        if best_score >= 0.12:
            status = "Strong Evidence" if best_score >= 0.30 else "Moderate Evidence"
            evidence_mappings.append({
                "job_requirement": req,
                "resume_evidence": best_sentence,
                "confidence_score": round(best_score * 100, 1),
                "status": status
            })
        else:
            evidence_mappings.append({
                "job_requirement": req,
                "resume_evidence": "No direct matching sentence detected in resume.",
                "confidence_score": round(best_score * 100, 1),
                "status": "Gap Detected"
            })

    return evidence_mappings
