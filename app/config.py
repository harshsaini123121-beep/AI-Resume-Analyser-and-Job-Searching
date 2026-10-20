"""
Application configuration and environment variable loading.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_NAME: str = "AI Resume Analyser & Job Matching System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # API Keys & Auth
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    API_KEY: str = os.getenv("APP_API_KEY", "resumelab_secret_api_key_2026")

    # Server Settings & CORS
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        raw = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000"
        )
        return [o.strip() for o in raw.split(",") if o.strip()]

    # Rate Limiting
    RATE_LIMIT_DEFAULT: str = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")
    RATE_LIMIT_UPLOAD: str = os.getenv("RATE_LIMIT_UPLOAD", "15/minute")
    RATE_LIMIT_MATCH: str = os.getenv("RATE_LIMIT_MATCH", "30/minute")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'resume_analyser.db'}")

    # Weights
    SKILL_WEIGHT: float = float(os.getenv("SKILL_WEIGHT", "0.5"))
    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.4"))
    EXPERIENCE_WEIGHT: float = float(os.getenv("EXPERIENCE_WEIGHT", "0.1"))

    # Directories
    DATA_DIR: Path = BASE_DIR / "data"
    SKILLS_FILE: Path = DATA_DIR / "skills.json"
    JOBS_FILE: Path = DATA_DIR / "jobs.csv"
    SAMPLE_RESUMES_DIR: Path = DATA_DIR / "sample_resumes"


settings = Settings()
