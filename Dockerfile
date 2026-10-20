# Multi-stage production Dockerfile for AI Resume Analyser & Job Matching Suite
# Stage 1: Build stage (Installs dependencies and pre-caches SentenceTransformer model)
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with CPU-only wheels where applicable
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download and cache all-MiniLM-L6-v2 model so container runs offline
ENV PYTHONPATH=/install/lib/python3.12/site-packages
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# Stage 2: Minimal hardened runtime image (Non-root, zero build tools)
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install minimal runtime dependencies (curl for container healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user (UID 1000)
RUN useradd -u 1000 -m -s /bin/bash appuser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded model cache to non-root user home directory
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Copy application source code with non-root ownership
COPY --chown=appuser:appuser . .

# Ensure correct permissions
RUN chown -R appuser:appuser /home/appuser/.cache /app

# Switch to non-root user
USER appuser

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PORT=8000

# Expose API and Streamlit UI ports
EXPOSE 8000 8501

# Healthcheck targeting backend liveness
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Default unified launcher
CMD ["python", "run.py"]
