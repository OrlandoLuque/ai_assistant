# Multi-stage Dockerfile for ai_assistant server
# Build stage: compile with all features
FROM rust:1.82-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock* ./

# Create a dummy main to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    echo "pub fn placeholder() {}" > src/lib.rs && \
    cargo build --release --features full 2>/dev/null || true

# Copy actual source
COPY src/ src/
COPY LICENSE ./

# Build the real binary
RUN cargo build --release --features full

# Runtime stage: minimal image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aiassistant && useradd -r -g aiassistant -d /app -s /sbin/nologin aiassistant

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/ai_assistant /app/ai_assistant

# Create config and data directories
RUN mkdir -p /app/config /app/data && chown -R aiassistant:aiassistant /app

USER aiassistant

# Default server port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8090/health || exit 1

ENV RUST_LOG=info
ENV AI_CONFIG_PATH=/app/config/config.toml

ENTRYPOINT ["/app/ai_assistant"]
