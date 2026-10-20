"""
Pydantic schemas for request validation and response serialization.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class EducationItem(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[int] = None
    raw_text: Optional[str] = None


class ExtractedDataResponse(BaseModel):
    candidate_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    links: Dict[str, Optional[str]] = Field(default_factory=dict)
    skills: List[str] = Field(default_factory=list)
    skills_count: int = 0
    education: List[EducationItem] = Field(default_factory=list)
    experience_years: float = 0.0
    sections_detected: List[str] = Field(default_factory=list)
    word_count: int = 0
    bullet_count: int = 0
    raw_text: str


class CategoryScore(BaseModel):
    score: int
    max_score: int
    details: str


class ATSScoreResponse(BaseModel):
    total_score: int
    grade: str
    category_breakdown: Dict[str, CategoryScore]
    issues: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class BulletRewrite(BaseModel):
    original: str
    improved: str


class AIFeedbackResponse(BaseModel):
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    improvements: List[str]
    rewritten_bullets: List[BulletRewrite]
    learning_path: List[str]
    source: str = "ai_feedback"


class JobMatchRequest(BaseModel):
    resume_text: str
    resume_skills: List[str] = Field(default_factory=list)
    candidate_exp_years: float = 0.0
    jd_text: str
    jd_skills: Optional[List[str]] = None
    required_exp_years: float = 0.0


class JobMatchResponse(BaseModel):
    match_percentage: float
    verdict: str
    verdict_color: str
    recommendation: str
    skill_score: float
    semantic_score: float
    experience_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    additional_skills: List[str]
    ai_feedback: Optional[AIFeedbackResponse] = None


class JobRecommendationRequest(BaseModel):
    resume_text: str
    resume_skills: List[str] = Field(default_factory=list)
    candidate_exp_years: float = 0.0
    top_n: int = 10
    location_filter: Optional[str] = None
    max_experience: Optional[float] = None
    search_query: Optional[str] = None


class JobItem(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    salary: str
    experience_required: float
    description: str
    required_skills: List[str]
    match_percentage: float
    verdict: str
    verdict_color: str
    matched_skills: List[str]
    missing_skills: List[str]


class JobRecommendationResponse(BaseModel):
    total_jobs_evaluated: int
    recommendations: List[JobItem]


class ResumeUploadResponse(BaseModel):
    filename: str
    extracted_data: ExtractedDataResponse


class FullAnalysisResponse(BaseModel):
    id: Optional[int] = None
    filename: str
    extracted_data: ExtractedDataResponse
    ats_score: ATSScoreResponse
