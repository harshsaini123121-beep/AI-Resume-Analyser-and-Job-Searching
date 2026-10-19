"""
SQLAlchemy ORM models for storing resume analyses, ATS reports, and job match queries.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from app.database import Base


class ResumeRecord(Base):
    __tablename__ = "resume_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    candidate_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(100), nullable=True)
    experience_years = Column(Float, default=0.0)
    skills_json = Column(Text, nullable=True)  # JSON-encoded array of skills
    ats_score = Column(Integer, default=0)
    ats_grade = Column(String(50), default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class MatchHistoryRecord(Base):
    __tablename__ = "match_history_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    candidate_name = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)
    match_percentage = Column(Float, default=0.0)
    verdict = Column(String(100), default="")
    matched_skills_json = Column(Text, nullable=True)
    missing_skills_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
