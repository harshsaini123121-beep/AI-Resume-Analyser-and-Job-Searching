# Contributing to ResumeLab

Thank you for your interest in contributing to ResumeLab! We welcome contributions to improve our document parsing, NLP extraction, matching algorithms, and UI components.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd "ML workshop"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows
   source venv/bin/activate      # Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

---

## Coding Standards & Testing

- **Client-Server Boundary**: Frontend code in `frontend/` MUST NOT import directly from `app.*`. All communication must flow through `frontend.api_client`.
- **Security & XSS Prevention**: 100% of user-supplied strings rendered into HTML must be escaped using `html.escape()`.
- **Run the Test Suite**:
   ```bash
   pytest tests/ -v
   ```
   Ensure all 96+ tests pass before opening a Pull Request.

---

## Pull Request Guidelines

1. Create a descriptive feature branch: `git checkout -b feature/my-feature`.
2. Commit your changes with clear messages.
3. Verify formatting and run the full test suite.
4. Submit the PR with a clear summary of changes and test outcomes.
