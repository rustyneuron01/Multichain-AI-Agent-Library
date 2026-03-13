# Score Vision Validator Docker Image
# Multi-stage build for optimized final image size
FROM python:3.11-slim-bullseye as builder

# ---------- Build stage ----------

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment with UV for faster package management
RUN pip install --no-cache-dir uv

# Create virtual environment in builder stage
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install Rust (nightly toolchain pinned for reproducibility)
ENV RUST_TOOLCHAIN=nightly-2024-09-30
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    /root/.cargo/bin/rustup toolchain install "${RUST_TOOLCHAIN}" && \
    /root/.cargo/bin/rustup default "${RUST_TOOLCHAIN}" && \
    /root/.cargo/bin/rustup toolchain remove stable

# --------------------
# Non‑root build user
# --------------------
# Define and create a dedicated unprivileged user for the build layer.
# Using an ARG here keeps the name flexible while ensuring the same UID/GID
# across layers to avoid permission issues when copying artifacts later.
ARG LOCAL_USER=builder
ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
RUN groupadd -g "${LOCAL_GID}" "${LOCAL_USER}" && \
    useradd -m -u "${LOCAL_UID}" -g "${LOCAL_GID}" -s /bin/bash "${LOCAL_USER}"

# Extend PATH for the new user
ENV PATH="/home/${LOCAL_USER}/.local/bin:/home/${LOCAL_USER}/.cargo/bin:${VIRTUAL_ENV}/bin:/usr/local/bin:${PATH}"

# Switch to the non‑root build user
USER "${LOCAL_USER}"

# Copy dependency files first (leverages Docker cache)
COPY pyproject.toml setup.py requirements.txt ./

# Copy source code
COPY miner/ ./miner/
COPY validator/ ./validator/

# Install the validator package with all dependencies
RUN uv pip install --no-cache-dir -e .[validator]

# ---------- Production stage ----------
FROM python:3.11-slim-bullseye as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional system libraries
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Create non‑root application user
ARG APP_USER=validator
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "${APP_GID}" "${APP_USER}" && \
    useradd -u "${APP_UID}" -g "${APP_GID}" -m -s /bin/bash "${APP_USER}"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Application directory
WORKDIR /app

# Copy application source with proper ownership
COPY --chown=${APP_USER}:${APP_USER} . .

# Persistent data directories
RUN mkdir -p /app/data /app/logs /app/debug_frames && \
    chown -R ${APP_USER}:${APP_USER} /app

# Switch to non‑root app user
USER "${APP_USER}"

# Environment variables
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check – simple Python import
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "validator.main"]

# Expose port (if needed for metrics/API)
EXPOSE 8000

# OCI labels
LABEL org.opencontainers.image.title="Score Vision Validator" \
      org.opencontainers.image.description="Soccer Video Analysis Validator for Bittensor Subnet 44" \
      org.opencontainers.image.version="0.3.0" \
      org.opencontainers.image.vendor="Score Vision Team" \
      org.opencontainers.image.source="https://github.com/score-technologies/score-vision"
