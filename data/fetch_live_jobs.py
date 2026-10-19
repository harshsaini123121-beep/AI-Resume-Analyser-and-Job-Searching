"""
Script to extract real-world job postings from open job APIs (Remotive, Arbeitnow)
and enrich data/jobs.csv with hundreds of live industry positions.
"""

import csv
import html
import re
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_CSV_PATH = BASE_DIR / "data" / "jobs.csv"


def clean_html_text(raw_html: str) -> str:
    """Removes HTML markup and normalizes whitespace."""
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def estimate_experience_required(text: str) -> float:
    """Estimates required experience from job description text."""
    pattern = r"(\d+(?:\.\d+)?)\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)(?:\s+of)?\s+(?:relevant\s+)?experience"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return min(12.0, max(0.5, val))
        except ValueError:
            pass
    if any(k in text.lower() for k in ["senior", "lead", "staff", "principal"]):
        return 5.0
    if any(k in text.lower() for k in ["junior", "entry", "associate", "intern"]):
        return 1.0
    return 3.0


def fetch_remotive_jobs(limit_per_category: int = 40):
    """Fetches jobs across multiple categories from Remotive API."""
    categories = ["software-dev", "data", "devops", "qa", "product"]
    fetched = []

    for cat in categories:
        url = f"https://remotive.com/api/remote-jobs?category={cat}&limit={limit_per_category}"
        print(f"[Remotive] Fetching category '{cat}'...")
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                data = r.json()
                jobs = data.get("jobs", [])
                print(f"[Remotive] Retrieved {len(jobs)} jobs for category '{cat}'.")
                for j in jobs:
                    desc = clean_html_text(j.get("description", ""))
                    tags = j.get("tags", [])
                    skills_str = "; ".join(tags) if tags else ""
                    exp = estimate_experience_required(desc)
                    salary = j.get("salary", "") or "Competitive"

                    fetched.append({
                        "job_id": f"rem_{j['id']}",
                        "title": j.get("title", "Software Engineer").strip(),
                        "company": j.get("company_name", "Tech Co").strip(),
                        "location": j.get("candidate_required_location", "Remote").strip(),
                        "description": desc[:3000],
                        "required_skills": skills_str,
                        "experience_required": exp,
                        "salary": salary
                    })
        except Exception as e:
            print(f"[Remotive] Error fetching {cat}: {e}")

    return fetched


def fetch_arbeitnow_jobs(pages: int = 3):
    """Fetches real-world jobs from Arbeitnow public job board API."""
    fetched = []
    for page in range(1, pages + 1):
        url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
        print(f"[Arbeitnow] Fetching page {page}...")
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                data = r.json()
                jobs = data.get("data", [])
                print(f"[Arbeitnow] Retrieved {len(jobs)} jobs on page {page}.")
                for j in jobs:
                    desc = clean_html_text(j.get("description", ""))
                    tags = j.get("tags", [])
                    skills_str = "; ".join(tags) if tags else ""
                    exp = estimate_experience_required(desc)
                    location = j.get("location", "Remote")
                    if j.get("remote", False):
                        location += " (Remote)"

                    fetched.append({
                        "job_id": f"arb_{j['slug'][:15]}",
                        "title": j.get("title", "Software Professional").strip(),
                        "company": j.get("company_name", "Enterprise").strip(),
                        "location": location.strip(),
                        "description": desc[:3000],
                        "required_skills": skills_str,
                        "experience_required": exp,
                        "salary": "Competitive Industry Standard"
                    })
        except Exception as e:
            print(f"[Arbeitnow] Error fetching page {page}: {e}")

    return fetched


def run_ingestion():
    print("==================================================")
    print("Extracting Real-World Job Postings from Public APIs")
    print("==================================================")

    remotive_jobs = fetch_remotive_jobs(limit_per_category=50)
    arbeitnow_jobs = fetch_arbeitnow_jobs(pages=3)
    all_new_jobs = remotive_jobs + arbeitnow_jobs

    print(f"\nTotal new real-world jobs harvested: {len(all_new_jobs)}")

    # Read existing IDs to avoid duplicates
    existing_ids = set()
    rows = []
    if JOBS_CSV_PATH.exists():
        with open(JOBS_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(str(row["job_id"]))
                rows.append(row)

    print(f"Existing jobs in dataset: {len(rows)}")

    # Append non-duplicate jobs
    added_count = 0
    fieldnames = ["job_id", "title", "company", "location", "description", "required_skills", "experience_required", "salary"]

    with open(JOBS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for job in all_new_jobs:
            if job["job_id"] not in existing_ids:
                writer.writerow(job)
                existing_ids.add(job["job_id"])
                added_count += 1

    print(f"Successfully added {added_count} new unique jobs to data/jobs.csv!")
    print(f"Total jobs dataset size now: {len(rows) + added_count}")
    print("==================================================")


if __name__ == "__main__":
    run_ingestion()
