"""
Database connection and session handling using SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# SQLite requires check_same_thread=False for multi-threaded FastAPI handlers
connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency that creates and closes a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
