"""
High-performance vector indexer for searching across hundreds or thousands of jobs
using pre-computed dense embeddings and cosine nearest-neighbor search.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.matcher import get_embedding_model
from app.services.cleaner import clean_text

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
JOBS_CSV_PATH = DATA_DIR / "jobs.csv"
EMBEDDINGS_CACHE_PATH = DATA_DIR / "jobs_embeddings.npy"
METADATA_CACHE_PATH = DATA_DIR / "jobs_meta.pkl"

_INDEX_INITIALIZED = False
_CACHED_EMBEDDINGS: Optional[np.ndarray] = None
_CACHED_JOBS_LIST: Optional[List[Dict[str, Any]]] = None
_TFIDF_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF_MATRIX = None


def build_or_load_vector_index(csv_path: Optional[str] = None, force_rebuild: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Builds or loads cached embeddings for the entire jobs dataset.
    Enables sub-millisecond similarity search across 700+ jobs.
    """
    global _INDEX_INITIALIZED, _CACHED_EMBEDDINGS, _CACHED_JOBS_LIST, _TFIDF_VECTORIZER, _TFIDF_MATRIX

    if _INDEX_INITIALIZED and not force_rebuild and _CACHED_EMBEDDINGS is not None:
        return _CACHED_EMBEDDINGS, _CACHED_JOBS_LIST

    path = Path(csv_path) if csv_path else JOBS_CSV_PATH
    df = pd.read_csv(path)
    df["title"] = df["title"].fillna("Software Professional")
    df["company"] = df["company"].fillna("Enterprise")
    df["location"] = df["location"].fillna("Remote")
    df["description"] = df["description"].fillna("")
    df["required_skills"] = df["required_skills"].fillna("")
    df["experience_required"] = pd.to_numeric(df["experience_required"], errors="coerce").fillna(0.0)
    df["salary"] = df["salary"].fillna("Competitive")

    jobs_list = df.to_dict(orient="records")

    # Combine title, skills, and description into searchable text passage
    corpus = [
        f"{j['title']} | Skills: {j['required_skills']} | {j['description'][:1200]}"
        for j in jobs_list
    ]

    model = get_embedding_model()
    embeddings = None

    if model is not None:
        try:
            print(f"[Vector Indexer] Generating dense embeddings for {len(corpus)} jobs using sentence-transformers...")
            embeddings = model.encode(corpus, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            # Normalize vectors for fast inner-product cosine search
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            # Cache to disk
            np.save(str(EMBEDDINGS_CACHE_PATH), embeddings)
            print(f"[Vector Indexer] Successfully indexed {len(jobs_list)} jobs in dense vector space.")
        except Exception as e:
            print(f"[Vector Indexer] Embeddings build notice: {e}. Falling back to TF-IDF.")
            embeddings = None

    if embeddings is None:
        # Fallback to high-performance TF-IDF vector matrix
        print(f"[Vector Indexer] Building sparse TF-IDF matrix for {len(corpus)} jobs...")
        _TFIDF_VECTORIZER = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
        _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(corpus)
        embeddings = _TFIDF_MATRIX

    _CACHED_EMBEDDINGS = embeddings
    _CACHED_JOBS_LIST = jobs_list
    _INDEX_INITIALIZED = True

    return _CACHED_EMBEDDINGS, _CACHED_JOBS_LIST


def search_vector_index(query_text: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
    """
    Finds the top_k most semantically relevant jobs for a given query or resume text.
    Returns list of (job_dict, similarity_score).
    """
    embeddings, jobs = build_or_load_vector_index()
    if embeddings is None or len(jobs) == 0:
        return []

    model = get_embedding_model()

    if model is not None and isinstance(embeddings, np.ndarray):
        query_vec = model.encode([query_text], convert_to_numpy=True)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        sims = np.dot(embeddings, query_vec.T).flatten()
    else:
        # TF-IDF sparse similarity
        global _TFIDF_VECTORIZER, _TFIDF_MATRIX
        if _TFIDF_VECTORIZER is None:
            build_or_load_vector_index(force_rebuild=True)
        query_vec = _TFIDF_VECTORIZER.transform([query_text])
        sims = cosine_similarity(_TFIDF_MATRIX, query_vec).flatten()

    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(jobs[idx], float(sims[idx])) for idx in top_indices]
    return results
