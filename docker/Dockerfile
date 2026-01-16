# Multi-stage build for production
FROM python:3.10.13-slim-bookworm as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10.13-slim-bookworm

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 modeluser && \
    mkdir -p /app/models /app/logs && \
    chown -R modeluser:modeluser /app

# Copy Python dependencies from builder
COPY --from=builder --chown=modeluser:modeluser /root/.local /home/modeluser/.local

# Copy application code
COPY --chown=modeluser:modeluser glm_model_production.py .
COPY --chown=modeluser:modeluser api_service.py .

# Switch to non-root user
USER modeluser

# Add local bin to PATH
ENV PATH=/home/modeluser/.local/bin:$PATH

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/glm_model.joblib \
    PORT=5000 \
    FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

EXPOSE 5000

# Run the API
CMD ["python", "api_service.py"]
