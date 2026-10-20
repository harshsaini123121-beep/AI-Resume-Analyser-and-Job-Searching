# Privacy, Ethics & Data Governance

## 1. Local-First Processing Architecture

ResumeLab is engineered with a **zero-cloud default** policy:
- **Local Inference**: Document extraction, NLP entity parsing, ATS scoring, and vector embeddings execute 100% locally on your infrastructure (`sentence-transformers/all-MiniLM-L6-v2` runs locally via PyTorch).
- **No Third-Party Data Transmission**: Candidate resumes, contact credentials, and work history are never transmitted to external APIs or remote model providers during standard operation.

---

## 2. Personally Identifiable Information (PII) Protection

- **Automated PII Masking**: Email addresses and phone numbers can be redacted via `mask_pii()` before any external processing.
- **Filename Sanitization**: Uploaded filenames are stripped of user path information, directory traversal attempts, and HTML tags, storing them behind randomized UUIDs.
- **Demographic Blindness**: The scoring engine evaluates purely objective competencies: verified skills, quantifiable impact metrics, action verb density, and tenure length. Demographics such as candidate name, gender, ethnicity, age, and location are explicitly excluded from mathematical scoring features.

---

## 3. Data Retention & Right to Erasure

- **Caller-Scoped Isolation**: Records in the SQLite database (`resume_analyser.db`) are partitioned by authenticated caller identity.
- **Right to Erasure (GDPR Compliance)**: Users and recruiters have full control over stored records. Any audit report can be permanently deleted via the UI or `DELETE /api/resume/{record_id}` endpoint.
- **Ephemeral Processing**: In batch screening mode, candidate resumes processed in-memory can be cleared immediately upon session completion.
