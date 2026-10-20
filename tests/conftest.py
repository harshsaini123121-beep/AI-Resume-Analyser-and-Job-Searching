"""
Global pytest fixtures for AI Resume Analyser & Job Matching test suite.
Provides:
- SQLite database initialization and schema verification
- TestClient with dependency override for clean, isolated database transactions
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db, init_db, engine, SessionLocal
from app.models import ResumeRecord, MatchHistoryRecord


@pytest.fixture(scope="session", autouse=True)
def initialize_database():
    """Ensures database tables and schema columns are created before any tests run."""
    init_db(bind_engine=engine)
    yield


@pytest.fixture
def db_session():
    """Provides a transactional database session per test with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.query(ResumeRecord).delete()
        db.query(MatchHistoryRecord).delete()
        db.commit()
        db.close()


@pytest.fixture
def test_client(db_session):
    """Provides FastAPI TestClient with database dependency override."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()
