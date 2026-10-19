# AI Resume Analyser & Job Matching System
### Complete Build Guide (using Google Antigravity)

---

## 1. What Are We Building?

A web app where:

1. A user **uploads a resume** (PDF/DOCX).
2. The system **extracts and analyses** it: skills, education, experience, contact info.
3. It gives a **resume score** (ATS-style) with **improvement suggestions**.
4. The user pastes a **job description** (or picks from a job database) and the system gives a **match percentage**, shows **matching skills** and **missing skills**.
5. The system **recommends the top N jobs** that best fit the resume.

### Core Features (MVP - build these first)

| # | Feature | Description |
|---|---------|-------------|
| 1 | Resume upload | PDF and DOCX support |
| 2 | Text extraction | Convert file to clean text |
| 3 | Information extraction | Name, email, phone, skills, education, experience |
| 4 | Resume vs Job Description match | Score 0-100% |
| 5 | Skill gap analysis | Matched skills vs missing skills |
| 6 | Job recommendations | Top 5-10 jobs from a dataset |
| 7 | AI suggestions | LLM-written feedback to improve the resume |

### Bonus Features (after the MVP works)

- User login and saved history
- ATS score breakdown (formatting, keywords, sections, length)
- Resume rewriting suggestions per bullet point
- Course/certification recommendations for missing skills
- Charts (skill radar, match gauge)
- Export report as PDF
- Live job fetching from an API (Adzuna, Remotive, etc.)

---

## 2. Tech Stack

Keep it simple and Python-based, so Antigravity's agents can build it quickly.

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.10+ | Best NLP ecosystem |
| Backend API | **FastAPI** | Fast, clean, auto docs at `/docs` |
| Frontend | **Streamlit** (easy) or React + Vite (advanced) | Streamlit = UI in pure Python |
| PDF parsing | `pdfplumber` (+ `PyPDF2` fallback) | Reliable text extraction |
| DOCX parsing | `python-docx` | Reads Word files |
| NLP | `spaCy` (`en_core_web_sm`) | Entity extraction, tokenising |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Semantic similarity, runs locally and free |
| Similarity | `scikit-learn` (cosine similarity, TF-IDF) | Simple and effective |
| LLM | **Gemini API** (or any LLM API) | Feedback and suggestions |
| Database | **SQLite** + SQLAlchemy | Zero setup |
| Charts | `plotly` | Interactive charts in Streamlit |
| PDF report (bonus) | `reportlab` or `fpdf2` | Export analysis |

> **Recommendation:** If you're a beginner, build the **Streamlit** frontend first. You can call your logic functions directly, and add FastAPI later if you want a proper API.

---

## 3. System Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────┐
│   Frontend   │     │                 Backend                  │
│  (Streamlit) │────▶│  FastAPI                                 │
│              │     │   ├─ /upload-resume                      │
│  Upload CV   │     │   ├─ /analyse                            │
│  Paste JD    │◀────│   ├─ /match                              │
│  See results │     │   └─ /recommend-jobs                     │
└──────────────┘     │                                          │
                     │  Services                                │
                     │   ├─ parser.py        (PDF/DOCX → text)  │
                     │   ├─ extractor.py     (skills, edu, exp) │
                     │   ├─ matcher.py       (embeddings+cosine)│
                     │   ├─ scorer.py        (ATS score)        │
                     │   └─ llm_feedback.py  (Gemini)           │
                     │                                          │
                     │  Data: SQLite + jobs.csv + skills.json   │
                     └──────────────────────────────────────────┘
```

### Data Flow

```
Resume file → Text extraction → Cleaning → Entity/skill extraction
                                               │
Job Description → Cleaning → Skill extraction ─┤
                                               ▼
                       Embeddings + Keyword overlap → Match score
                                               ▼
                          Skill gap + LLM suggestions → Report
```

---

## 4. Project Folder Structure

```
resume-analyser/
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── config.py              # settings, env variables
│   ├── models.py              # DB models
│   ├── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── parser.py          # PDF/DOCX → text
│   │   ├── cleaner.py         # text cleaning
│   │   ├── extractor.py       # skills, education, experience, contact
│   │   ├── matcher.py         # resume ↔ job similarity
│   │   ├── scorer.py          # ATS score
│   │   ├── recommender.py     # top job recommendations
│   │   └── llm_feedback.py    # Gemini suggestions
│   └── routes/
│       ├── resume.py
│       └── jobs.py
├── data/
│   ├── skills.json            # master skills list
│   ├── jobs.csv               # job dataset
│   └── sample_resumes/        # test resumes
├── frontend/
│   └── streamlit_app.py
├── tests/
│   ├── test_parser.py
│   ├── test_extractor.py
│   └── test_matcher.py
├── .env                       # API keys (never commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Step-by-Step Build Plan

### PHASE 0: Setup (Day 1)

**Install prerequisites**
- Python 3.10+
- Git
- Google Antigravity (download from the official site, sign in with your Google account)
- A Gemini API key from Google AI Studio (free tier is enough)

**Open the project in Antigravity**
1. Create an empty folder `resume-analyser`.
2. Open it as a workspace in Antigravity.
3. Open the **Agent Manager** (where you give tasks to agents) and the **Editor** (where you view and edit code).

**How Antigravity works (quick mental model)**
- **Editor view**: like VS Code, with an agent side panel.
- **Agent Manager**: you give high-level tasks; agents plan, write code, run terminal commands and even test in a browser.
- **Artifacts**: agents produce task lists, implementation plans, screenshots and walkthroughs. **Review the plan before approving**.
- Always ask for a **plan first**, then approve, then let it build.

**Tip:** Create a `README.md` and a `PROJECT_CONTEXT.md` (paste this guide into it) so every agent has context.

---

### PHASE 1: Environment & Skeleton (Day 1)

**Prompt for Antigravity agent:**

```
Create a Python project called resume-analyser with this folder structure:
app/ (with services/ and routes/), data/, frontend/, tests/.
Set up a virtual environment, create requirements.txt with:
fastapi, uvicorn, streamlit, pdfplumber, python-docx, spacy,
sentence-transformers, scikit-learn, pandas, numpy, plotly,
sqlalchemy, python-dotenv, google-generativeai, pydantic, python-multipart, pytest.
Install the packages and download the spaCy model en_core_web_sm.
Add a .gitignore (venv, .env, __pycache__, uploads).
Show me the plan first before creating files.
```

**Manual commands (if needed):**
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**.env file**
```
GEMINI_API_KEY=your_key_here
```

---

### PHASE 2: Resume Parsing (Day 2)

**Goal:** Take a PDF/DOCX and return clean text.

**Logic (`parser.py`):**
- If `.pdf` → use `pdfplumber` to read each page's text.
- If `.docx` → use `python-docx` to read paragraphs and tables.
- Else → raise an error.
- If extracted text is empty, the PDF is probably a scanned image. Show a message (OCR with `pytesseract` is an optional bonus).

**Cleaning (`cleaner.py`):**
- Remove extra whitespace and odd characters
- Normalise bullets (•, ▪, -)
- Lowercase a copy for matching (keep the original for display)

**Prompt:**
```
In app/services/parser.py write functions extract_text_from_pdf(file),
extract_text_from_docx(file) and extract_text(file, filename) that routes by
extension. In cleaner.py write clean_text(text). Add unit tests in
tests/test_parser.py using files in data/sample_resumes/. Handle errors gracefully.
```

---

### PHASE 3: Information Extraction (Day 3-4)

**Goal:** Pull structured data from raw text.

**What to extract**

| Field | Method |
|-------|--------|
| Email | Regex `[\w\.-]+@[\w\.-]+\.\w+` |
| Phone | Regex for 10-digit and international formats |
| Name | spaCy `PERSON` entity (usually first one) or first line |
| LinkedIn / GitHub | Regex for URLs |
| Skills | Match against `skills.json` (phrase matching with spaCy `PhraseMatcher`) |
| Education | Keywords: B.Tech, B.E., M.Tech, MBA, B.Sc, PhD, University, College + year regex |
| Experience (years) | Regex on date ranges (e.g. `Jan 2021 - Present`) and "X years" |
| Sections | Detect headings: Summary, Skills, Experience, Education, Projects, Certifications |

**Create `data/skills.json`** (grow this list to 300-500+ skills):
```json
{
  "programming": ["python", "java", "c++", "javascript", "sql", "r"],
  "web": ["html", "css", "react", "node.js", "django", "flask", "fastapi"],
  "data": ["pandas", "numpy", "power bi", "tableau", "excel", "statistics"],
  "ml_ai": ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "scikit-learn"],
  "cloud_devops": ["aws", "azure", "gcp", "docker", "kubernetes", "git", "ci/cd"],
  "databases": ["mysql", "postgresql", "mongodb", "redis"],
  "soft_skills": ["communication", "leadership", "teamwork", "problem solving"]
}
```
Also add a **synonyms map** (e.g. `"js" → "javascript"`, `"ml" → "machine learning"`, `"node" → "node.js"`).

**Prompt:**
```
Write app/services/extractor.py with functions: extract_email, extract_phone,
extract_name (spaCy), extract_links, extract_skills (PhraseMatcher using
data/skills.json plus a synonym map), extract_education, extract_experience_years,
detect_sections. Return everything as a dictionary matching a Pydantic schema
ResumeData in schemas.py. Add tests.
```

---

### PHASE 4: Matching Engine, the Core (Day 5-6)

Use **two signals** and combine them:

**1. Semantic similarity (embeddings)**
- Model: `all-MiniLM-L6-v2`
- Embed the full resume and the full job description
- Cosine similarity → 0 to 1

**2. Skill overlap (keywords)**
- `matched = resume_skills ∩ jd_skills`
- `missing = jd_skills − resume_skills`
- `skill_score = len(matched) / len(jd_skills)`

**3. Final score formula (tweakable)**
```
final_score = 0.5 * skill_score + 0.4 * semantic_score + 0.1 * experience_match
```

Where `experience_match` = 1 if resume years ≥ required years, else a ratio.

**Output example:**
```json
{
  "match_percentage": 78.4,
  "matched_skills": ["python", "sql", "machine learning"],
  "missing_skills": ["docker", "aws"],
  "semantic_score": 0.81,
  "skill_score": 0.69,
  "verdict": "Strong match"
}
```

**Verdict bands:** `>= 80` Excellent · `60-79` Good · `40-59` Average · `< 40` Weak.

**Prompt:**
```
Create app/services/matcher.py. Load sentence-transformers all-MiniLM-L6-v2 once
(singleton). Implement compute_match(resume_text, resume_skills, jd_text) that
returns match_percentage, matched_skills, missing_skills, semantic_score,
skill_score and a verdict using weights 0.5 skill, 0.4 semantic, 0.1 experience.
Add tests with a good-match and a bad-match example.
```

---

### PHASE 5: ATS Resume Scoring (Day 7)

Score the resume itself (without a job description) out of 100:

| Category | Points | Check |
|----------|--------|-------|
| Contact info | 10 | email, phone, LinkedIn present |
| Sections present | 20 | Summary, Skills, Experience, Education, Projects |
| Skills count | 15 | at least 8-10 relevant skills |
| Quantified achievements | 15 | numbers and percentages in bullets (`increased by 30%`) |
| Action verbs | 10 | led, built, developed, optimised... |
| Length | 10 | 300-800 words (1-2 pages) |
| Formatting | 10 | no tables/images only, consistent bullets |
| Spelling/clarity | 10 | optional, via LLM |

**Prompt:**
```
Create app/services/scorer.py with calculate_ats_score(resume_text, extracted_data)
returning total score out of 100, a per-category breakdown, and a list of
specific issues found (for example "No quantified achievements found").
```

---

### PHASE 6: Job Recommendations (Day 8)

**Get a job dataset:**
- Kaggle: search "LinkedIn job postings", "Indeed jobs dataset", "Naukri jobs dataset"
- Or API: Adzuna, Remotive, or JSearch (RapidAPI) for live jobs
- Or create 50-100 sample jobs by hand (fine for a college project)

**`data/jobs.csv` columns:**
```
job_id, title, company, location, description, required_skills, experience_required, salary
```

**Logic (`recommender.py`):**
1. Pre-compute embeddings for all job descriptions once (cache as `.npy` / pickle).
2. Embed the resume.
3. Cosine similarity against all jobs, then combine with skill overlap.
4. Return the top 10 with their match % and missing skills.

**Prompt:**
```
Create app/services/recommender.py. Load data/jobs.csv with pandas, precompute and
cache job embeddings, and implement recommend_jobs(resume_text, resume_skills, top_n=10).
Also generate a 60-row realistic jobs.csv covering data science, web dev, backend,
DevOps, and analytics roles.
```

---

### PHASE 7: LLM-Powered Feedback (Day 9)

Use Gemini to produce human-friendly output:

**Prompt template (inside `llm_feedback.py`):**
```
You are an expert recruiter and resume coach.
Resume:
{resume_text}

Target job description:
{jd_text}

Matched skills: {matched}
Missing skills: {missing}

Return JSON with:
1. summary: 2-3 line overall assessment
2. strengths: list of 3-5 items
3. weaknesses: list of 3-5 items
4. improvements: list of specific, actionable suggestions
5. rewritten_bullets: 3 weak bullet points rewritten with measurable impact
6. learning_path: courses or projects to close the skill gaps
Respond only with valid JSON.
```

**Important tips:**
- Ask for JSON output and parse it safely (`try/except`, strip code fences).
- Add a **fallback**: if the API fails, show rule-based suggestions from the scorer.
- Never send personal data you don't need. Consider masking phone and email before sending.
- Cache results so you don't repeat API calls.

---

### PHASE 8: Backend API (Day 10)

**Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/resume/upload` | Upload file, return extracted data |
| POST | `/api/resume/analyse` | ATS score + suggestions |
| POST | `/api/match` | Resume + JD → match result |
| POST | `/api/jobs/recommend` | Resume → top jobs |
| GET | `/api/history` | Past analyses (bonus) |

Run with: `uvicorn app.main:app --reload`, then open `http://localhost:8000/docs` to test.

**Prompt:**
```
Create FastAPI routes in app/routes/ for the endpoints above using the services
I built. Use Pydantic schemas, validate file type and size (max 5 MB), enable CORS,
and return clear error messages. Store each analysis in SQLite via SQLAlchemy.
```

---

### PHASE 9: Frontend (Day 11-12)

**Streamlit UI layout:**

- **Sidebar:** navigation (Analyse Resume · Match with Job · Job Recommendations)
- **Page 1: Analyse Resume**
  - File uploader
  - Extracted info card (name, email, skills as chips)
  - ATS score gauge + category bar chart
  - Issues and suggestions list
- **Page 2: Match with Job**
  - Text area for the job description
  - Match percentage gauge
  - Two columns: ✅ Matched skills / ❌ Missing skills
  - AI feedback section
- **Page 3: Job Recommendations**
  - Table or cards of top jobs with match %
  - Filters: location, experience

**Prompt:**
```
Create frontend/streamlit_app.py with three pages as described. Use plotly for a
gauge chart (match %) and a bar chart (ATS breakdown). Show skills as coloured
chips. Call the FastAPI backend via requests. Make it clean, with a loading
spinner and error handling. Then run it in the browser and verify each page works
with a sample resume.
```

Run: `streamlit run frontend/streamlit_app.py`

> **Antigravity advantage:** its browser agent can open your running app, click through the pages and screenshot results. Ask it: *"Open the running Streamlit app, upload data/sample_resumes/test1.pdf, and verify the match page works. Give me screenshots."*

---

### PHASE 10: Testing & Evaluation (Day 13)

**Functional testing**
- 10+ different resumes (clean, messy, 2-column, scanned, DOCX)
- Edge cases: empty file, huge file, wrong format, non-English text

**Evaluate your matching quality (great for your report/viva):**
1. Create 20 resume-job pairs and label them manually (good/average/bad).
2. Run your system and compare the predictions with your labels.
3. Report accuracy, and show a confusion matrix or correlation.
4. Compare approaches: TF-IDF only vs Embeddings only vs Hybrid. Show the hybrid is best.

**Unit tests:** `pytest tests/`

---

### PHASE 11: Deployment (Day 14)

| Option | Notes |
|--------|-------|
| **Streamlit Community Cloud** | Easiest for a Streamlit-only app; free |
| **Hugging Face Spaces** | Great for ML demos; free |
| **Render / Railway** | For FastAPI + frontend |
| **Docker** | Package for any server |

Before deploying:
- Put API keys in the platform's secrets, never in code
- Pin versions in `requirements.txt`
- Add a `README.md` with screenshots and a demo GIF

---

## 6. Best Practices When Using Antigravity

1. **Plan before code.** Ask: *"Create an implementation plan, don't write code yet."* Review and edit the plan.
2. **One phase at a time.** Don't ask for the whole project in one prompt.
3. **Give context.** Keep `PROJECT_CONTEXT.md` updated and tell the agent to read it.
4. **Review diffs.** Read what the agent changes; you must understand your own project (you'll be asked about it in a viva or interview).
5. **Ask for tests.** *"Write and run tests for what you just built."*
6. **Use the browser agent** to verify the UI.
7. **Commit often** with Git after every working phase.
8. **When stuck:** paste the full error and ask the agent to *explain the cause first, then fix*.
9. **Don't share secrets** (API keys) in prompts. Use the `.env` file.

### Master Prompt (use at the very start)

```
I'm building an "AI Resume Analyser & Job Matching System" in Python.
Stack: FastAPI backend, Streamlit frontend, pdfplumber/python-docx for parsing,
spaCy for extraction, sentence-transformers (all-MiniLM-L6-v2) + scikit-learn for
matching, Gemini API for suggestions, SQLite for storage.
Features: resume upload, info extraction, ATS score, resume-vs-JD match %,
skill gap analysis, job recommendations, AI feedback.
First, read PROJECT_CONTEXT.md, then give me a detailed implementation plan with
milestones. Do not write code until I approve the plan.
```

---

## 7. Timeline Summary

| Days | Phase | Output |
|------|-------|--------|
| 1 | Setup + skeleton | Project runs, packages installed |
| 2 | Parsing | Text from PDF/DOCX |
| 3-4 | Extraction | Structured resume data |
| 5-6 | Matching engine | Match %, skill gap |
| 7 | ATS scoring | Resume score + issues |
| 8 | Recommendations | Top jobs list |
| 9 | LLM feedback | AI suggestions |
| 10 | Backend API | Working endpoints |
| 11-12 | Frontend | Full UI |
| 13 | Testing + evaluation | Metrics and bug fixes |
| 14 | Deployment + docs | Live demo, README |

---

## 8. Common Problems & Fixes

| Problem | Fix |
|---------|-----|
| PDF text comes out empty | Likely a scanned PDF → use OCR (`pytesseract` + `pdf2image`) |
| Text order jumbled (2-column resumes) | Try `pdfplumber` with layout settings or `PyMuPDF (fitz)` |
| Skills not detected | Expand `skills.json`, add synonyms, lowercase both sides |
| Embedding model slow on first run | It downloads once; load it a single time at startup |
| Gemini returns invalid JSON | Strip ```` ``` ```` fences, use `try/except`, retry once |
| Match score too high for unrelated jobs | Increase the weight of skill overlap; add a minimum threshold |
| CORS errors | Add `CORSMiddleware` in FastAPI |
| Memory errors on deploy | Use a smaller model or cache embeddings |

---

## 9. Ethics, Privacy & Fairness (include in your report)

- Resumes hold personal data: store minimally, allow deletion, and don't log full text.
- Don't use gender, age, name, photo, or religion in scoring.
- State clearly that the score is **guidance, not a hiring decision**.
- Be transparent about how the score is calculated.

---

## 10. Ideas to Make It Stand Out

- **Explainability:** highlight the exact resume sentences that matched the JD
- **Resume vs multiple JDs** in one go
- **Skill-gap roadmap:** weekly plan to learn missing skills
- **Chrome extension** that grabs a job description from LinkedIn and runs the match
- **Cover letter generator** tailored to the JD
- **Mock interview questions** generated from the resume + JD
- **Dashboard** with analytics (most common missing skills across all users)

---

## 11. Final Deliverables Checklist

- [ ] Working app (local + deployed link)
- [ ] GitHub repo with clean commits
- [ ] `README.md` with setup steps, screenshots, demo GIF
- [ ] `requirements.txt`
- [ ] Sample resumes and jobs dataset
- [ ] Unit tests passing
- [ ] Evaluation results (accuracy / comparison table)
- [ ] Architecture diagram
- [ ] Project report / presentation slides
- [ ] 2-3 minute demo video

---

## 12. Learning Resources

- FastAPI docs: fastapi.tiangolo.com
- Streamlit docs: docs.streamlit.io
- spaCy course: course.spacy.io
- Sentence-Transformers: sbert.net
- Gemini API: ai.google.dev
- Kaggle datasets for job postings and resumes

---

**Start with Phase 1 today. Get a tiny end-to-end flow working first (upload → extract text → match score), then improve each piece.** 🚀
