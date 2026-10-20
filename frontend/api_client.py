"""
ResumeLab Typed API Client.
Encapsulates all communication between the Streamlit frontend and the FastAPI backend.
Features:
- Configurable base URL, timeout, and automatic connection retries
- Transparent error mapping with custom APIConnectionError and APIResponseError
- 100% typed responses eliminating business logic from the UI
"""

import os
import time
import io
from typing import Dict, List, Optional, Any, Union
import httpx

# Resolve default backend address
DEFAULT_HOST = os.getenv("API_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("API_PORT", "8000")
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
DEFAULT_API_KEY = os.getenv("APP_API_KEY", "resumelab_secret_api_key_2026")


class APIConnectionError(Exception):
    """Raised when the backend API is unreachable or times out."""
    pass


class APIResponseError(Exception):
    """Raised when the backend returns a non-2xx status code."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"API Error ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


class ResumeLabClient:
    """
    Robust typed HTTP client for the ResumeLab FastAPI backend.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        timeout: float = 30.0,
        max_retries: int = 2
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self, additional: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {"X-API-Key": self.api_key}
        if additional:
            headers.update(additional)
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        is_binary: bool = False
    ) -> Union[Dict[str, Any], bytes]:
        url = f"{self.base_url}{endpoint}"
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(
                        method=method,
                        url=url,
                        headers=self._headers(),
                        json=json_data,
                        files=files,
                        params=params
                    )

                    if response.status_code >= 400:
                        try:
                            err_body = response.json()
                            detail = err_body.get("detail") or err_body.get("message") or response.text
                        except Exception:
                            detail = response.text or f"HTTP {response.status_code}"
                        raise APIResponseError(response.status_code, str(detail))

                    if is_binary:
                        return response.content
                    return response.json()

            except httpx.ConnectError as ce:
                last_exception = ce
                if attempt < self.max_retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise APIConnectionError(
                    f"Unable to connect to backend at {self.base_url}. Ensure the FastAPI server is running."
                ) from ce
            except httpx.TimeoutException as te:
                last_exception = te
                if attempt < self.max_retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise APIConnectionError(
                    f"Request to {url} timed out after {self.timeout}s."
                ) from te
            except httpx.RequestError as re:
                raise APIConnectionError(f"HTTP request error: {str(re)}") from re

        raise APIConnectionError(f"Failed to communicate with backend: {last_exception}")

    # -------------------------------------------------------------------------
    # Health & Readiness
    # -------------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Returns True if the backend is reachable and healthy."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/health")
                return res.status_code == 200 and res.json().get("status") == "healthy"
        except Exception:
            return False

    def get_readiness(self) -> Dict[str, Any]:
        """Checks readiness including database and taxonomy indices."""
        return self._request("GET", "/ready")

    # -------------------------------------------------------------------------
    # Resume Upload & ATS Analysis
    # -------------------------------------------------------------------------

    def upload_resume(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Uploads and parses a resume file, extracting structured fields."""
        files = {"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")}
        return self._request("POST", "/api/v1/resume/upload", files=files)

    def analyse_resume(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Performs full ATS analysis, scoring, and persists to audit history."""
        files = {"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")}
        return self._request("POST", "/api/v1/resume/analyse", files=files)

    # -------------------------------------------------------------------------
    # Matching & Recommendations
    # -------------------------------------------------------------------------

    def match_resume_with_job(
        self,
        resume_text: str,
        resume_skills: List[str],
        candidate_exp_years: float,
        jd_text: str,
        jd_skills: Optional[List[str]] = None,
        required_exp_years: float = 0.0
    ) -> Dict[str, Any]:
        """Computes hybrid alignment, skill gaps, and Gemini AI feedback."""
        payload = {
            "resume_text": resume_text,
            "resume_skills": resume_skills,
            "candidate_exp_years": candidate_exp_years,
            "jd_text": jd_text,
            "jd_skills": jd_skills,
            "required_exp_years": required_exp_years
        }
        return self._request("POST", "/api/v1/match", json_data=payload)

    def get_xai_evidence(
        self,
        resume_text: str,
        jd_text: str,
        top_n_requirements: int = 5
    ) -> List[Dict[str, Any]]:
        """Maps job requirements to supporting resume sentences."""
        payload = {
            "resume_text": resume_text,
            "jd_text": jd_text,
            "top_n_requirements": top_n_requirements
        }
        res = self._request("POST", "/api/v1/match/xai-evidence", json_data=payload)
        return res.get("evidence", [])

    def recommend_jobs(
        self,
        resume_text: str,
        resume_skills: List[str],
        candidate_exp_years: float = 0.0,
        top_n: int = 10,
        location_filter: Optional[str] = None,
        max_experience: Optional[float] = None,
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieves top N matching positions for candidate profile."""
        payload = {
            "resume_text": resume_text,
            "resume_skills": resume_skills,
            "candidate_exp_years": candidate_exp_years,
            "top_n": top_n,
            "location_filter": location_filter,
            "max_experience": max_experience,
            "search_query": search_query
        }
        return self._request("POST", "/api/v1/jobs/recommend", json_data=payload)

    def list_jobs(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Fetches curated job listings from the database."""
        res = self._request("GET", "/api/v1/jobs/list", params={"limit": limit})
        return res.get("jobs", [])

    # -------------------------------------------------------------------------
    # Batch Screening & Bullet Optimizer
    # -------------------------------------------------------------------------

    def batch_screen(
        self,
        jd_text: str,
        candidates: List[Dict[str, str]],
        job_id: Optional[str] = None,
        jd_title: Optional[str] = None,
        required_skills: Optional[List[str]] = None,
        experience_required: float = 0.0
    ) -> Dict[str, Any]:
        """Screens multiple resumes against a target job."""
        payload = {
            "job_id": job_id,
            "jd_title": jd_title or "Target Role",
            "jd_text": jd_text,
            "required_skills": required_skills or [],
            "experience_required": experience_required,
            "candidates": candidates
        }
        return self._request("POST", "/api/v1/jobs/batch-screen", json_data=payload)

    def batch_screen_async(
        self,
        jd_text: str,
        candidates: List[Dict[str, str]],
        job_id: Optional[str] = None,
        jd_title: Optional[str] = None,
        required_skills: Optional[List[str]] = None,
        experience_required: float = 0.0
    ) -> Dict[str, Any]:
        """Submits candidate resumes for asynchronous background batch screening."""
        payload = {
            "job_id": job_id,
            "jd_title": jd_title or "Target Role",
            "jd_text": jd_text,
            "required_skills": required_skills or [],
            "experience_required": experience_required,
            "candidates": candidates
        }
        return self._request("POST", "/api/v1/jobs/batch-screen-async", json_data=payload)

    def get_batch_progress(self, job_id: str) -> Dict[str, Any]:
        """Polls the status, progress percentage, or completed results of a batch job."""
        return self._request("GET", f"/api/v1/jobs/batch/{job_id}/progress")

    def optimize_bullet(self, bullet_text: str) -> Dict[str, Any]:
        """Rewrites and audits a bullet point using Google X-Y-Z rubric."""
        return self._request("POST", "/api/v1/resume/bullet-optimize", json_data={"bullet_text": bullet_text})

    # -------------------------------------------------------------------------
    # PDF Report & Audit History
    # -------------------------------------------------------------------------

    def generate_pdf_report(
        self,
        candidate_name: str,
        ats_score: int,
        ats_grade: str,
        category_breakdown: Dict[str, Any],
        strengths: List[str],
        issues: List[str],
        suggestions: List[str],
        skills: List[str],
        email: Optional[str] = None,
        phone: Optional[str] = None
    ) -> bytes:
        """Requests server-side PDF generation and returns raw bytes."""
        payload = {
            "candidate_name": candidate_name,
            "email": email,
            "phone": phone,
            "ats_score": ats_score,
            "ats_grade": ats_grade,
            "category_breakdown": category_breakdown,
            "strengths": strengths,
            "issues": issues,
            "suggestions": suggestions,
            "skills": skills
        }
        return self._request("POST", "/api/v1/resume/report-pdf", json_data=payload, is_binary=True)

    def get_history(self, offset: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves scoped audit history for authenticated caller with pagination."""
        res = self._request("GET", "/api/v1/resume/history", params={"offset": offset, "limit": limit})
        return res.get("history", [])

    def delete_history_record(self, record_id: int) -> bool:
        """Deletes an audit history record by ID."""
        res = self._request("DELETE", f"/api/v1/resume/{record_id}")
        return res.get("status") == "deleted"

    def get_skills_taxonomy(self) -> Dict[str, Any]:
        """Fetches the categorized skills taxonomy from the backend."""
        return self._request("GET", "/api/v1/resume/skills-taxonomy")

    def extract_skills_and_exp(self, text: str) -> Dict[str, Any]:
        """Extracts skills and experience tenure from text via the backend API."""
        return self._request("POST", "/api/v1/jobs/extract-skills", json_data={"text": text})


# Global singleton client instance
api_client = ResumeLabClient()
