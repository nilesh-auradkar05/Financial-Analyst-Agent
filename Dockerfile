# syntax=docker/dockerfile:1.7
# Financial Analyst API: reproducible, GPU-compatible production runtime.
# CUDA/PyTorch wheels remain installed when required by pyproject.toml/uv.lock.
# NVIDIA GPU access is configured at deployment, not in the image.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE} AS builder

WORKDIR /app

# Use a pinned uv release instead of mutable :latest.
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /usr/local/bin/

# Use the same system CPython (not a downloaded second interpreter).
# BuildKit cache mounts retain package downloads outside the image layers.
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

# Keep toolchain available only in the build stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install locked production dependencies before copying frequently changing code.
# --no-default-groups is essential: uv normally adds the dev group by default.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group api --no-install-project

# This assumes app/ and any package data are included in the build context.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group api --no-editable

FROM ${PYTHON_IMAGE} AS runtime
WORKDIR /app

# OpenMP is needed by several numerical libraries. Retain CA certificates from base.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Avoid root at runtime; match the original UID/GID to preserve volume ownership.
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appgroup . /app/

# Keep runtime model caches and mutable state outside the image via Compose volumes.
RUN mkdir -p /app/data/chroma /app/data/filings /app/.runtime /app/.cache \
    && chown -R appuser:appgroup /app/data /app/.runtime /app/.cache

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

USER appuser
EXPOSE 8000

# Uses stdlib; no curl package needed in the final stage.
# A long start period in Compose accommodates FinBERT warmup.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8).close()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
