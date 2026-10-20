# ResumeLab REST API Reference

Base URL: `http://127.0.0.1:8000`

Interactive OpenAPI Documentation: `http://127.0.0.1:8000/docs`

---

## 1. Authentication & Headers

Protected endpoints require the API Key sent via HTTP header:
```http
X-API-Key: your-api-key
```
*(Default development key: `test-api-key` or configured via `API_KEY` in `.env`)*

---

## 2. Health & Diagnostics

### `GET /health`
Returns system health, database status, and active ML model.
- **Response `200 OK`**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-10-20T04:40:00Z",
    "model_loaded": true,
    "model_name": "all-MiniLM-L6-v2",
    "database": "connected",
    "version": "1.0.0"
  }
  ```

---

## 3. Resume Endpoints

### `POST /api/v1/resume/analyse`
Parses a resume file, extracts structured entities, and executes the 100-point ATS audit.
- **Headers**: `X-API-Key: <key>`
- **Content-Type**: `multipart/form-data`
- **Body**: `file: Binary file (.pdf, .docx, .txt)`
- **Response `200 OK`**:
  ```json
  {
    "status": "success",
    "extracted_data": {
      "candidate_name": "Jane Doe",
      "email": "jane.doe@example.com",
      "phone": "(555) 123-4567",
      "skills": ["python", "react", "docker", "sql"],
      "experience_years": 4.5,
      "sections": { "summary": "...", "experience": "..." }
    },
    "ats_score": {
      "total_score": 85,
      "grade": "A",
      "category_breakdown": {
        "Contact Information": { "score": 10, "max_score": 10, "details": "Complete" }
      },
      "strengths": ["Strong action verb density"],
      "issues": ["Include metrics on project outcomes"],
      "suggestions": ["Add percentage improvements"]
    },
    "page_count": 1
  }
  ```

### `POST /api/v1/resume/bullet-optimize`
Audits and rewrites a single resume bullet point according to Google's X-Y-Z formula.
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "bullet_text": "Developed web applications and maintained database tables for the team."
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "original_bullet": "Developed web applications and maintained database tables for the team.",
    "score": 65,
    "grade": "B-",
    "has_action_verb": true,
    "has_quantifiable_metric": false,
    "has_context_or_tech": true,
    "feedback": ["Quantify impact with metrics or percentages."],
    "improved_rewrite": "Engineered scalable web applications and optimized relational database schemas, reducing query latency by 35%."
  }
  ```

### `POST /api/v1/resume/report-pdf`
Generates a downloadable branded ATS audit report PDF.
- **Content-Type**: `application/json`
- **Body**: Audit metrics, candidate metadata, and category breakdown.
- **Response `200 OK`**: `application/pdf` binary stream.

---

## 4. Job Matching & Explainable AI

### `POST /api/v1/match`
Executes dual-signal hybrid matching between a resume and a job description.
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "resume_text": "Experienced Python backend developer...",
    "resume_skills": ["python", "docker", "fastapi"],
    "candidate_exp_years": 4.0,
    "jd_text": "Seeking a Senior Python Engineer...",
    "jd_skills": ["python", "aws", "docker"],
    "required_exp_years": 3.0
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "match_percentage": 84,
    "verdict": "Strong Match",
    "skill_score": 0.67,
    "semantic_score": 0.88,
    "experience_score": 1.0,
    "matched_skills": ["python", "docker"],
    "missing_skills": ["aws"]
  }
  ```

### `POST /api/v1/match/xai-evidence`
Extracts sentence-level proof pairing candidate resume lines with job description requirements.
- **Content-Type**: `application/json`
- **Body**: `resume_text`, `jd_text`, `top_n_requirements`
- **Response `200 OK`**:
  ```json
  [
    {
      "job_requirement": "Experience building REST APIs with FastAPI and Docker.",
      "resume_evidence": "Designed and deployed containerized REST microservices using FastAPI and Docker.",
      "similarity_score": 0.91,
      "status": "Strong Evidence"
    }
  ]
  ```

---

## 5. Recruiter Batch Screening

### `POST /api/v1/jobs/batch-screen`
Screens and ranks multiple candidate resumes against a target job requisition.
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "job_id": "JOB-101",
    "jd_title": "Machine Learning Engineer",
    "jd_text": "...",
    "required_skills": ["python", "pytorch"],
    "experience_required": 3.0,
    "candidates": [
      {"filename": "alice.txt", "text": "..."},
      {"filename": "bob.txt", "text": "..."}
    ]
  }
  ```
- **Response `200 OK`**: Leaderboard ranked by `match_percentage` descending.

---

## 6. Audit History & Deletion

### `GET /api/resume/history`
Retrieves audit records scoped to the authenticated caller.
- **Headers**: `X-API-Key: <key>`
- **Response `200 OK`**: List of audit entries.

### `DELETE /api/resume/{record_id}`
Permanently deletes an audit record from the database.
- **Headers**: `X-API-Key: <key>`
- **Response `200 OK`**: `{"status": "deleted", "id": 12}`
