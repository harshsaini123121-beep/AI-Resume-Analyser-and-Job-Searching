"""
SQLAlchemy ORM models for storing users, resumes, analyses, jobs, and match results.
Includes database indexes on job_id, owner_id, and created_at.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Index
from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)
    full_name = Column(String(255), nullable=True)
    role = Column(String(50), default="candidate")
    created_at = Column(DateTime, default=utc_now, index=True)


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    storage_uuid = Column(String(64), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    owner_id = Column(String(255), nullable=False, default="default_user", index=True)
    candidate_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(100), nullable=True)
    experience_years = Column(Float, default=0.0)
    skills_json = Column(Text, nullable=True)  # JSON-encoded array of skills
    ats_score = Column(Integer, default=0)
    ats_grade = Column(String(50), default="")
    raw_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, index=True)


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    resume_id = Column(Integer, nullable=True, index=True)
    owner_id = Column(String(255), nullable=False, default="default_user", index=True)
    ats_score = Column(Integer, default=0)
    ats_grade = Column(String(50), default="")
    breakdown_json = Column(Text, nullable=True)
    strengths_json = Column(Text, nullable=True)
    issues_json = Column(Text, nullable=True)
    suggestions_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, index=True)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String(64), nullable=False, unique=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    location = Column(String(255), nullable=True)
    experience_required = Column(Float, default=0.0)
    salary = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    required_skills_json = Column(Text, nullable=True)
    preferred_skills_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, index=True)


class MatchResult(Base):
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String(64), nullable=True, index=True)
    resume_id = Column(Integer, nullable=True, index=True)
    owner_id = Column(String(255), nullable=False, default="default_user", index=True)
    candidate_name = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)
    match_percentage = Column(Float, default=0.0)
    verdict = Column(String(100), default="")
    matched_skills_json = Column(Text, nullable=True)
    missing_skills_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, index=True)


# Backwards compatibility aliases for existing codebase and tests
ResumeRecord = Resume
MatchHistoryRecord = MatchResult
