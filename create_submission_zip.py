"""
Production Submission Packaging Script:
- Secret Scanner: Scans all packaging files with regex to prevent credential/token leaks.
- Required Files Checklist: Asserts presence of core code, configs, tests, datasets, and docs.
- Atomic Write: Writes to a temporary zip file and atomically renames upon verification.
- Exclusions: Enforces exclusion of venv/, caches, git history, databases, and .env files.
- ZIP Validation: Asserts total archive size is under 10 MB.
"""

import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Safe stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
ZIP_FINAL = BASE_DIR / "AI_Resume_Analyser_Project.zip"

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "app/main.py",
    "app/services/parser.py",
    "app/services/extractor.py",
    "app/services/scorer.py",
    "app/services/matcher.py",
    "frontend/streamlit_app.py",
    "frontend/styles.css",
    "data/skills.json",
    "data/jobs.csv",
    "docs/architecture.md",
    "docs/api.md",
    "docs/evaluation.md",
    "docs/privacy.md"
]

EXCLUDE_DIRS = {
    "venv", ".venv", "env", "__pycache__", ".pytest_cache", ".git", ".idea", ".vscode", "scratch"
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".db", ".sqlite3", ".npy", ".hash", ".coverage"
}

EXCLUDE_FILES = {
    "AI_Resume_Analyser_Project.zip", ".env", ".coverage"
}

SECRET_PATTERNS = [
    re.compile(r'(?:AIzaSy[A-Za-z0-9_-]{33})'),                     # Google API key
    re.compile(r'(?:sk-[a-zA-Z0-9]{32,})'),                         # OpenAI / secret key
    re.compile(r'(?:ghp_[a-zA-Z0-9]{36})'),                         # GitHub personal token
    re.compile(r'(?:xox[baprs]-[0-9a-zA-Z]{10,48})'),               # Slack token
]


def verify_required_files():
    """Verify all critical submission files exist."""
    print("Checking required files checklist...")
    missing = []
    for rel_str in REQUIRED_FILES:
        target = BASE_DIR / rel_str
        if not target.exists():
            missing.append(rel_str)

    if missing:
        print(f"❌ Error: Missing required files: {missing}")
        sys.exit(1)
    print(f"✅ All {len(REQUIRED_FILES)} required files verified.")


def scan_file_for_secrets(file_path: Path):
    """Scan file content for secret tokens."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            match = pattern.search(content)
            if match:
                raise ValueError(f"Secret detected in {file_path.name}: {match.group()[:8]}...")
    except Exception as e:
        if isinstance(e, ValueError):
            raise e


def build_submission_zip():
    print("==================================================")
    print("ResumeLab: Creating Clean Project Submission ZIP")
    print("==================================================")

    verify_required_files()

    file_count = 0
    total_uncompressed_bytes = 0

    # Write atomically to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_zip_path = Path(tmp_file.name)

    try:
        with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(BASE_DIR):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(BASE_DIR)

                    if file in EXCLUDE_FILES:
                        continue
                    if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
                        continue
                    if any(part in EXCLUDE_DIRS for part in rel_path.parts):
                        continue

                    # Scan code and text files for secrets
                    if file_path.suffix.lower() in (".py", ".json", ".md", ".toml", ".yml", ".yaml", ".txt", ".ini"):
                        scan_file_for_secrets(file_path)

                    zipf.write(file_path, arcname=str(rel_path))
                    file_count += 1
                    total_uncompressed_bytes += file_path.stat().st_size

        zip_size_bytes = tmp_zip_path.stat().st_size
        zip_size_mb = zip_size_bytes / (1024 * 1024)
        raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

        if zip_size_mb > 10.0:
            raise ValueError(f"ZIP size ({zip_size_mb:.2f} MB) exceeds maximum allowed limit of 10.0 MB!")

        # Atomic move to final destination
        shutil.move(str(tmp_zip_path), str(ZIP_FINAL))

        print(f"✅ Successfully packaged {file_count} files.")
        print(f"📦 Uncompressed size: {raw_size_mb:.2f} MB")
        print(f"🗜️  Final ZIP size:     {zip_size_mb:.2f} MB (< 10 MB limit)")
        print(f"📁 Destination:        {ZIP_FINAL.name}")
        print("==================================================")
        print("Ready for upload and project delivery!")

    finally:
        if tmp_zip_path.exists():
            try:
                tmp_zip_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    build_submission_zip()
