# Financial Analyst Agent System Dockerfile
# Multi-stage build for optimized production image.

# Stages:
    # 1. builder - Install dependenceis and build wheels
    # 2. runtime - Minimal image with only the necessary dependencies

# Usage:
    # docker build -t financial-analyst-agent-system .
    # docker run -p 8000:8000 --env-file .env financial-analyst-agent-system

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==2.2.1

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev, no root package yet)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main --no-interaction --no-ansi

# Copy application code
COPY . .

# Install dependencies without packaging the app
RUN poetry install --no-root --only main --no-interaction --no-ansi

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appgroup . .

# Create data directories
RUN mkdir -p /app/data/chroma /app/data/filings \
    && chown -R appuser:appgroup /app/data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    