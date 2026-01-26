# Multi-stage build for production
FROM python:3.10.13-slim-bookworm as builder

WORKDIR /app

# Install build dependencies and apply security updates
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel to fix CVE-2024-6345, CVE-2025-47273, CVE-2026-24049
RUN pip install --no-cache-dir --upgrade \
    pip>=23.3 \
    setuptools>=70.0.0 \
    wheel>=0.46.2

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (packaging needed explicitly for limits/flask-limiter)
RUN pip install --no-cache-dir --user packaging && \
    pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10.13-slim-bookworm

WORKDIR /app

# Apply security updates to production image
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 modeluser && \
    mkdir -p /app/models /app/logs && \
    chown -R modeluser:modeluser /app

# Copy Python dependencies from builder
COPY --from=builder --chown=modeluser:modeluser /root/.local /home/modeluser/.local

# Install packaging in production stage (required by limits/flask-limiter)
RUN pip install --no-cache-dir packaging

# Copy application code
COPY --chown=modeluser:modeluser src/ ./src/
COPY --chown=modeluser:modeluser api/ ./api/

# Switch to non-root user
USER modeluser

# Add local bin to PATH
ENV PATH=/home/modeluser/.local/bin:$PATH

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/glm_model.joblib \
    PORT=5000 \
    FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

EXPOSE 5000

# Run the API
CMD ["python", "/app/api/app.py"]
