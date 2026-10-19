"""
FastAPI main application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import engine, Base
from app.routes.resume import router as resume_router
from app.routes.jobs import router as jobs_router
from app.services.extractor import load_skills_database
from app.services.recommender import load_jobs_dataset


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite database tables and preheats data caches on startup."""
    print("[Startup] Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("[Startup] Pre-loading skills taxonomy and jobs database...")
    load_skills_database()
    load_jobs_dataset()
    print("[Startup] AI Resume Analyser backend initialized successfully.")
    yield
    print("[Shutdown] Cleaning up server resources...")


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

# Configure CORS for local Streamlit frontend and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler returning clean structured error messages."""
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "message": str(exc)}
    )
