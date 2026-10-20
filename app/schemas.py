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
    page_count: Optional[int] = 1
    extracted_data: ExtractedDataResponse


class FullAnalysisResponse(BaseModel):
    id: Optional[int] = None
    filename: str
    page_count: Optional[int] = 1
    extracted_data: ExtractedDataResponse
    ats_score: ATSScoreResponse



# -----------------------------------------------------------------------------
# Part 2: Bullet Optimizer, XAI Evidence, Batch Screening, and PDF Report Schemas
# -----------------------------------------------------------------------------

class BulletOptimizeRequest(BaseModel):
    bullet_text: str


class BulletOptimizeResponse(BaseModel):
    score: int
    grade: str = ""
    status: Optional[str] = ""
    has_action_verb: bool
    has_metrics: bool
    has_context: bool
    feedback: List[str]
    improved_rewrite: str
    metrics_found: List[str] = Field(default_factory=list)


class XAIEvidenceRequest(BaseModel):
    resume_text: str
    jd_text: str
    top_n_requirements: int = 5


class XAIEvidenceItem(BaseModel):
    job_requirement: str
    resume_evidence: str
    confidence_score: Optional[float] = 0.0
    similarity_score: Optional[float] = 0.0
    status: str



class XAIEvidenceResponse(BaseModel):
    evidence: List[XAIEvidenceItem]


class BatchCandidateInput(BaseModel):
    filename: str
    text: str


class BatchScreenRequest(BaseModel):
    job_id: Optional[str] = None
    jd_title: Optional[str] = "Target Role"
    jd_text: str
    required_skills: Optional[List[str]] = Field(default_factory=list)
    experience_required: float = 0.0
    candidates: List[BatchCandidateInput]


class BatchCandidateResult(BaseModel):
    candidate_name: str
    filename: str
    match_percentage: float
    ats_score: int
    experience_years: float
    skills_matched_count: int
    missing_skills: List[str]


class BatchScreenResponse(BaseModel):
    job_title: str
    candidates_screened: int
    results: List[BatchCandidateResult]


class PDFReportRequest(BaseModel):
    candidate_name: str = "Applicant"
    email: Optional[str] = None
    phone: Optional[str] = None
    ats_score: int
    ats_grade: str
    category_breakdown: Dict[str, Any] = Field(default_factory=dict)
    strengths: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)

