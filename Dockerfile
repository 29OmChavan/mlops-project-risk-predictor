FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by light/xgboost OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions first for Docker layer caching
COPY requirements.txt setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project configuration, templates, and trained artifacts
COPY config/ ./config/
COPY params.yaml schema.yaml main.py app.py ./
COPY templates/ ./templates/
COPY artifacts/ ./artifacts/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
