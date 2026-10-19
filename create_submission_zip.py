"""
Helper script to package the project into a clean, lightweight submission ZIP file.
Includes:
- All source code (app/, frontend/, tests/)
- All datasets (data/jobs.csv, data/skills.json, sample resumes)
- Benchmark scripts, evaluation reports, and README
Excludes:
- venv/ (2.3 GB virtual environment)
- __pycache__/ and *.pyc
- .pytest_cache/
- Temporary SQLite databases
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ZIP_NAME = BASE_DIR / "AI_Resume_Analyser_Project.zip"

EXCLUDE_DIRS = {
    "venv", ".venv", "env", "__pycache__", ".pytest_cache", ".git", ".idea", ".vscode"
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".db", ".sqlite3"
}

EXCLUDE_FILES = {
    "AI_Resume_Analyser_Project.zip", ".env"
}


def build_submission_zip():
    print("==================================================")
    print("Creating Clean Project Submission ZIP...")
    print("==================================================")

    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(BASE_DIR):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(BASE_DIR)

                # Check exclusions
                if file in EXCLUDE_FILES:
                    continue
                if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
                    continue
                if any(part in EXCLUDE_DIRS for part in rel_path.parts):
                    continue

                zipf.write(file_path, arcname=str(rel_path))
                file_count += 1
                total_uncompressed_bytes += file_path.stat().st_size

    zip_size_mb = ZIP_NAME.stat().st_size / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print(f"Packaged {file_count} files.")
    print(f"Original size: {raw_size_mb:.2f} MB")
    print(f"ZIP file size: {zip_size_mb:.2f} MB")
    print(f"Created: {ZIP_NAME.name}")
    print("==================================================")
    print("Ready to upload directly to Google Drive!")


if __name__ == "__main__":
    build_submission_zip()
