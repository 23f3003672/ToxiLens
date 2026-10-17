# ToxiLens Lightweight API Container
# Optimized for 512 MB RAM cloud tiers (Render / DigitalOcean / Fly.io)

FROM python:3.11-slim AS base

# Install essential runtime dependencies for RDKit rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    rdkit \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    pillow \
    fastapi \
    uvicorn \
    pydantic \
    requests

# Copy project code and trained model bundle
COPY src/ /app/src/
COPY api/ /app/api/
COPY data/ /app/data/
COPY models/ /app/models/

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
