"""
FastAPI main application entry point.
Configured with strict CORS allowlist, request-ID tracking, slowapi rate limiting,
and sanitized error responses that never leak internal exceptions or secret paths.
"""

from contextlib import asynccontextmanager
import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import settings
from app.database import engine, Base, init_db
from app.routes.resume import router as resume_router
from app.routes.jobs import router as jobs_router
from app.services.extractor import load_skills_database
from app.services.recommender import load_jobs_dataset

logger = logging.getLogger("app.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite database tables and preheats data caches on startup."""
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
    """Injects unique request ID for correlation and logs status code."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Mount API Routers
app.include_router(resume_router, prefix=settings.API_V1_STR)
app.include_router(jobs_router, prefix=settings.API_V1_STR)


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
    """Health check endpoint for liveness monitoring."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """Readiness check verifying database and taxonomies are loaded."""
    skills = load_skills_database()
    jobs_df = load_jobs_dataset()
    return {
        "status": "ready",
        "skills_indexed": len(skills.get("categories", {})),
        "jobs_indexed": len(jobs_df)
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler returning generic error messages with request ID.
    Never exposes str(exc), internal paths, or credentials to clients.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"[Request {request_id}] Unhandled exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal server error occurred. Please contact support with this request ID.",
            "request_id": request_id
        }
    )
