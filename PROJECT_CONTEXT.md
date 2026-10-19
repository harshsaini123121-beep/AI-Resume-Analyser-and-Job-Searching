# Project Context: AI Resume Analyser & Job Matching System

## Overview
A production-grade AI Resume Analyser and Job Matching platform that empowers job seekers and recruiters to:
1. Parse resumes from multi-page PDFs, Word documents (.docx), and text files.
2. Extract rich structured attributes: contact details (email, phone, LinkedIn, GitHub), educational background, career tenure / years of experience, standard resume sections, and over 400+ categorized technical and soft skills with synonym matching.
3. Compute an objective, 100-point ATS (Applicant Tracking System) score evaluating section completeness, quantifiable achievements, active verbs, formatting, and skill density.
4. Perform hybrid job matching against target Job Descriptions using dual-signal scoring: semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and keyword skill overlap (`matched` vs `missing`).
5. Recommend top-matching jobs from a realistic, curated database of 60+ industry roles with dynamic filtering.
6. Provide actionable AI coaching feedback via Gemini (with resilient offline rule-based fallbacks) including executive summaries, strengths, weaknesses, bullet rewrites with metrics, and targeted learning paths.
7. Offer both a RESTful FastAPI backend and a Streamlit interactive web interface with Plotly charts and downloadable PDF reports.

## Architecture
- **Backend API**: FastAPI (`app.main:app`), SQLAlchemy ORM with SQLite database (`resume_analyser.db`).
- **Frontend Dashboard**: Streamlit (`frontend/streamlit_app.py`) featuring modern glassmorphism aesthetic, interactive Plotly score gauges, category bar charts, skill badges, and job comparison cards.
- **NLP / ML Core**:
  - `pdfplumber` & `pypdf` for PDF parsing; `python-docx` for DOCX parsing.
  - Regex pattern matching, section detection, and dictionary-based entity extraction.
  - `sentence-transformers` (`all-MiniLM-L6-v2`) + `scikit-learn` for semantic embeddings and cosine similarity.
  - `google-generativeai` for optional cloud LLM coaching with offline rule-based fallback.
