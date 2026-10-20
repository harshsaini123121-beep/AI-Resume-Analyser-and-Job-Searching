"""
FastAPI main application entry point.
Configured with strict CORS allowlist, request-ID tracking, slowapi rate limiting,
structured JSON logging, and sanitized error responses that never leak internal exceptions or secret paths.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import sys
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from app.config import settings
from app.database import engine, Base, init_db
from app.routes.resume import router as resume_router
from app.routes.jobs import router as jobs_router
from app.services.extractor import load_skills_database
from app.services.recommender import load_jobs_dataset
from app.services.matcher import is_model_loaded, is_tfidf_fallback_active


# Configure Structured JSON Logging
class JSONLogFormatter(logging.Formatter):
    """Formats log records as structured JSON without exposing PII or raw text."""
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data)


logger = logging.getLogger("app.main")
logger.setLevel(logging.INFO)

# Set root logger handler to JSON formatter
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONLogFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database tables and preheats data caches on startup."""
    logger.info("[Startup] Initializing database tables...")
    init_db(bind_engine=engine)
    logger.info("[Startup] Pre-loading skills taxonomy and jobs database...")
    load_skills_database()
    load_jobs_dataset()
    logger.info("[Startup] AI Resume Analyser backend initialized successfully.")
    yield
    logger.info("[Shutdown] Cleaning up server resources...")


# Initialize SlowAPI Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-grade AI Resume Analyser and Job Matching API. "
        "Provides multi-format parsing, NLP entity extraction, ATS rubric evaluation, "
        "hybrid semantic matching, top-N job recommendation, and Gemini AI coaching."
    ),
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure CORS with strict explicit origin allowlist
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_audit_middleware(request: Request, call_next):
    """Injects unique request ID for correlation and logs structured JSON audit data."""
    start_time = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id

    # If calling legacy unversioned /api/* paths, attach standard HTTP deprecation headers
    path = request.url.path
    if path.startswith("/api/") and not path.startswith("/api/v1/"):
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "Wed, 01 Jan 2027 00:00:00 GMT"
        successor = path.replace("/api/", "/api/v1/", 1)
        response.headers["Link"] = f'<{successor}>; rel="successor-version"'

    # Structured JSON log entry (strictly no PII, no raw resume text)
    client_ip = request.client.host if request.client else "unknown"
    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "request_id": request_id,
        "method": request.method,
        "path": path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
        "client_ip": client_ip
    }
    logger.info(json.dumps(audit_entry))

    return response


# Mount API v1 Routers (canonical primary paths)
app.include_router(resume_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")

# Mount legacy unversioned /api paths for full backwards compatibility
app.include_router(resume_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")


@app.get("/")
def root():
    """Root status endpoint."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint for instant liveness monitoring."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """
    Readiness check verifying database connection, model status,
    and loaded taxonomies. Returns 503 if database is disconnected.
    """
    # Verify database connection
    db_connected = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error(f"[Readiness] Database health check failed: {e}")

    skills = load_skills_database()
    jobs_df = load_jobs_dataset()
    model_loaded = is_model_loaded()
    tfidf_active = is_tfidf_fallback_active()

    payload = {
        "status": "ready" if db_connected else "not_ready",
        "database_status": "connected" if db_connected else "disconnected",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "model_loaded": model_loaded,
        "tfidf_fallback_active": tfidf_active,
        "skills_indexed": len(skills.get("categories", {})),
        "jobs_indexed": len(jobs_df) if jobs_df is not None else 0
    }

    if not db_connected:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler returning generic error messages with request ID.
    Never exposes str(exc), internal paths, or credentials to clients.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    err_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "ERROR",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "error_type": type(exc).__name__,
        "message": "Unhandled internal server error"
    }
    logger.error(json.dumps(err_entry))
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal server error occurred. Please contact support with this request ID.",
            "request_id": request_id
        }
    )
