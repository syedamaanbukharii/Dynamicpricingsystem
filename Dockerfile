# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: builder -- install Python dependencies into a virtualenv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tooling needed to compile some wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create an isolated virtualenv we can copy into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime -- slim image, non-root, with Playwright system deps
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app

# Runtime libraries: libpq for psycopg, plus the shared libs Chromium needs so
# Playwright-based scraping works inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
        libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Create a non-root user and a writable data directory.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p ${APP_HOME} \
    && chown -R appuser:appuser ${APP_HOME}

WORKDIR ${APP_HOME}

# Application code.
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser docker/entrypoint.sh ./docker/entrypoint.sh
COPY --chown=appuser:appuser pyproject.toml README.md ./

# Ensure data directories exist and are writable by the app user.
RUN mkdir -p data/raw data/processed data/features data/models \
    && chown -R appuser:appuser data \
    && chmod +x docker/entrypoint.sh

USER appuser

# Install Playwright's Chromium for the app user (optional at runtime; the app
# degrades gracefully if scraping is unused).
RUN python -m playwright install chromium || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["api"]
