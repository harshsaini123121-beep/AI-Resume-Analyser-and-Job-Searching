"""
Application configuration and environment variable loading using Pydantic Settings.
Enforces startup validation of configuration parameters.
"""

from pathlib import Path
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "AI Resume Analyser & Job Matching System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # API Keys & Auth
    GEMINI_API_KEY: str = Field(default="", validation_alias="GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = Field(default="gemini-1.5-flash", validation_alias="GEMINI_MODEL_NAME")
    API_KEY: str = Field(default="resumelab_secret_api_key_2026", validation_alias="APP_API_KEY")

    # Server Settings & CORS
    API_HOST: str = Field(default="127.0.0.1", validation_alias="API_HOST")
    API_PORT: int = Field(default=8000, validation_alias="API_PORT")
    CORS_ORIGINS_RAW: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000",
        validation_alias="CORS_ORIGINS"
    )

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS_RAW.split(",") if o.strip()]

    # Rate Limiting
    RATE_LIMIT_DEFAULT: str = Field(default="60/minute", validation_alias="RATE_LIMIT_DEFAULT")
    RATE_LIMIT_UPLOAD: str = Field(default="15/minute", validation_alias="RATE_LIMIT_UPLOAD")
    RATE_LIMIT_MATCH: str = Field(default="30/minute", validation_alias="RATE_LIMIT_MATCH")

    # Database (supports postgresql via DATABASE_URL or sqlite for dev)
    DATABASE_URL: str = Field(
        default=f"sqlite:///{BASE_DIR / 'resume_analyser.db'}",
        validation_alias="DATABASE_URL"
    )

    # Weights
    SKILL_WEIGHT: float = Field(default=0.5, validation_alias="SKILL_WEIGHT")
    SEMANTIC_WEIGHT: float = Field(default=0.4, validation_alias="SEMANTIC_WEIGHT")
    EXPERIENCE_WEIGHT: float = Field(default=0.1, validation_alias="EXPERIENCE_WEIGHT")

    # ML & Embedding Models
    EMBEDDING_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL_NAME")
    ALLOW_TFIDF_FALLBACK: bool = Field(default=True, validation_alias="ALLOW_TFIDF_FALLBACK")

    # Directories
    DATA_DIR: Path = BASE_DIR / "data"
    SKILLS_FILE: Path = BASE_DIR / "data" / "skills.json"
    JOBS_FILE: Path = BASE_DIR / "data" / "jobs.csv"
    SAMPLE_RESUMES_DIR: Path = BASE_DIR / "data" / "sample_resumes"

    @field_validator("SKILL_WEIGHT", "SEMANTIC_WEIGHT", "EXPERIENCE_WEIGHT")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {v}")
        return v


settings = Settings()
