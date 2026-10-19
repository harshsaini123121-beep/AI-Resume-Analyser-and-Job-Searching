"""
Unified application launcher for the AI Resume Analyser & Job Matching System.
Usage:
    python run.py             # Launches both Backend and Frontend
    python run.py --backend   # Launches FastAPI backend on :8000
    python run.py --frontend  # Launches Streamlit frontend on :8501
    python run.py --test      # Runs test suite with pytest
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Check if running inside venv; if not, use venv's python if available
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_tests():
    """Executes the automated pytest suite."""
    print("==================================================")
    print("Running Automated Test Suite...")
    print("==================================================")
    cmd = [PYTHON_EXE, "-m", "pytest", "tests/", "-v", "--tb=short"]
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(result.returncode)


def run_backend():
    """Starts the FastAPI backend with Uvicorn."""
    print("==================================================")
    print("Starting FastAPI Backend at http://127.0.0.1:8000")
    print("API Docs: http://127.0.0.1:8000/docs")
    print("==================================================")
    cmd = [
        PYTHON_EXE, "-m", "uvicorn",
        "app.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_frontend():
    """Starts the Streamlit web dashboard."""
    print("==================================================")
    print("Starting Streamlit Dashboard at http://localhost:8501")
    print("==================================================")
    cmd = [
        PYTHON_EXE, "-m", "streamlit", "run",
        "frontend/streamlit_app.py",
        "--server.port", "8501",
        "--server.headless", "false"
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_both():
    """Launches FastAPI in the background and Streamlit in the foreground."""
    print("==================================================")
    print("Launching AI Resume Analyser & Job Matching System")
    print("==================================================")
    print("[1/2] Starting FastAPI backend on http://127.0.0.1:8000...")
    backend_proc = subprocess.Popen(
        [
            PYTHON_EXE, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8000"
        ],
        cwd=str(BASE_DIR)
    )

    # Wait for backend to initialize
    time.sleep(2.5)

    print("[2/2] Starting Streamlit dashboard on http://localhost:8501...")
    frontend_proc = None
    try:
        cmd = [
            PYTHON_EXE, "-m", "streamlit", "run",
            "frontend/streamlit_app.py",
            "--server.port", "8501"
        ]
        frontend_proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        if backend_proc.poll() is None:
            backend_proc.terminate()
        if frontend_proc and frontend_proc.poll() is None:
            frontend_proc.terminate()
        print("All processes stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Resume Analyser Launcher")
    parser.add_argument("--backend", action="store_true", help="Launch FastAPI backend only")
    parser.add_argument("--frontend", action="store_true", help="Launch Streamlit frontend only")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    parser.add_argument("--both", action="store_true", help="Launch both backend and frontend")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.backend:
        run_backend()
    elif args.frontend:
        run_frontend()
    else:
        run_both()
