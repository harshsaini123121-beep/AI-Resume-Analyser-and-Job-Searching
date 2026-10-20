"""
Database connection and session handling using SQLAlchemy.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# SQLite requires check_same_thread=False for multi-threaded FastAPI handlers
connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db(bind_engine=None):
    """Initializes tables and metadata for SQLite and PostgreSQL."""
    target_engine = bind_engine or engine
    Base.metadata.create_all(bind=target_engine)
    if "sqlite" in str(target_engine.url):
        with target_engine.connect() as conn:
            try:
                res = conn.execute(text("PRAGMA table_info(resumes)")).fetchall()
                col_names = [r[1] for r in res]
                if col_names and "storage_uuid" not in col_names:
                    conn.execute(text("ALTER TABLE resumes ADD COLUMN storage_uuid VARCHAR(64)"))
                conn.commit()
            except Exception:
                pass


def get_db():
    """Dependency that creates and closes a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
