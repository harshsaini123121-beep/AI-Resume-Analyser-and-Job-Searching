"""
Application configuration and environment variable loading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_NAME: str = "AI Resume Analyser & Job Matching System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Server Settings
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

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
