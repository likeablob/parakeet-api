# Use a slim Python image
FROM python:3.12-slim

ARG TARGETARCH

# Install ffmpeg static build (smaller than apt-get install ffmpeg)
RUN <<EOF
set -eux
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates xz-utils
case "${TARGETARCH}" in
    amd64) FFMPEG_ARCH="linux64" ;;
    arm64) FFMPEG_ARCH="linuxarm64" ;;
    *) echo "Unsupported architecture: ${TARGETARCH}" && exit 1 ;;
esac
curl -L "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-${FFMPEG_ARCH}-gpl.tar.xz" \
    | tar xJ -C /usr/local --strip-components=1 --wildcards "ffmpeg-master-latest-${FFMPEG_ARCH}-gpl/bin/ffmpeg"
apt-get purge -y curl xz-utils
apt-get autoremove -y
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
COPY .env.example ./

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
