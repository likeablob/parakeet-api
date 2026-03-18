# Use a slim Python image
FROM python:3.12-slim

# Install system dependencies (ffmpeg is required for non-WAV audio support)
RUN <<EOF
set -eux
apt-get update
apt-get install -y ffmpeg
rm -rf /var/lib/apt/lists/*
EOF

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies (skip project install to avoid README error during build)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and README (required by hatchling/pyproject.toml)
COPY src/ ./src/
COPY README.md ./

# Complete installation including the project itself
RUN uv sync --frozen --no-dev

# Expose the default port
EXPOSE 8816

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    STT__MODELS_DIR=/app/models \
    SERVER__HOST=0.0.0.0 \
    SERVER__PORT=8816

# Run the API server using 'serve' subcommand
ENTRYPOINT ["uv", "run", "--no-sync", "parakeet-api", "serve"]
