"""
Resume processing routes: upload, extraction, ATS scoring, and history retrieval.
Enforces file format, size, non-empty content validation, and database audit persistence.
"""

import os
import json
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ResumeRecord
from app.services.parser import extract_document, ResumeParsingError
from app.services.extractor import extract_resume_data
from app.services.scorer import calculate_ats_score
from app.schemas import (
    ResumeUploadResponse,
    FullAnalysisResponse,
    ExtractedDataResponse,
    ATSScoreResponse
)

router = APIRouter(prefix="/resume", tags=["Resume"])

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _validate_uploaded_file(file: UploadFile, content: bytes) -> str:
    """Validates filename, extension, empty content, and size limits."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: .pdf, .docx, .txt"
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File cannot be empty (0 bytes).")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({len(content)} bytes) exceeds the maximum allowed limit of 5 MB."
        )

    return ext


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    """
    Uploads a resume file (PDF, DOCX, TXT), extracts text and parses structured attributes:
    contact info, education, years of experience, section headers, and skills.
    """
    content = await file.read()
    _validate_uploaded_file(file, content)

    try:
        doc = extract_document(content, file.filename)
    except ResumeParsingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected parsing error: {str(e)}")

    extracted_data = extract_resume_data(doc["text"])

    return ResumeUploadResponse(
        filename=file.filename,
        extracted_data=ExtractedDataResponse(**extracted_data)
    )


@router.post("/analyse", response_model=FullAnalysisResponse)
async def analyse_resume(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Performs full ATS compliance analysis on an uploaded resume:
    evaluates against 100-point rubric, computes category breakdowns,
    generates warnings, and records the analysis into database history.
    """
    content = await file.read()
    _validate_uploaded_file(file, content)

    try:
        doc = extract_document(content, file.filename or "resume.pdf")
    except ResumeParsingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected parsing error: {str(e)}")

    extracted_data = extract_resume_data(doc["text"])
    ats_score_result = calculate_ats_score(doc["text"], extracted_data)

    # Persist in SQLite audit history
    record = ResumeRecord(
        filename=file.filename or "resume.pdf",
        candidate_name=extracted_data.get("candidate_name"),
        email=extracted_data.get("email"),
        phone=extracted_data.get("phone"),
        experience_years=extracted_data.get("experience_years", 0.0),
        skills_json=json.dumps(extracted_data.get("skills", [])),
        ats_score=ats_score_result.get("total_score", 0),
        ats_grade=ats_score_result.get("grade", "")
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return FullAnalysisResponse(
        id=record.id,
        filename=record.filename,
        extracted_data=ExtractedDataResponse(**extracted_data),
        ats_score=ATSScoreResponse(**ats_score_result)
    )


@router.get("/history")
def get_resume_history(limit: int = 20, db: Session = Depends(get_db)):
    """Returns recent resume analyses saved in the database."""
    records = db.query(ResumeRecord).order_by(ResumeRecord.created_at.desc()).limit(limit).all()
    history = []
    for r in records:
        skills = []
        if r.skills_json:
            try:
                skills = json.loads(r.skills_json)
            except Exception:
                pass
        history.append({
            "id": r.id,
            "filename": r.filename,
            "candidate_name": r.candidate_name,
            "email": r.email,
            "experience_years": r.experience_years,
            "skills": skills,
            "ats_score": r.ats_score,
            "ats_grade": r.ats_grade,
            "created_at": r.created_at.isoformat() if r.created_at else None
        })
    return {"history": history}
