# Multi-stage Dockerfile for LLM-TradeBot
# Stage 1: Builder - Compile dependencies and install TA-Lib
# Stage 2: Runtime - Minimal production image

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.13-slim-bookworm AS builder

# Install build dependencies for TA-Lib compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and compile TA-Lib from source
# TA-Lib is required for technical indicators (RSI, MACD, Bollinger Bands, etc.)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements.txt for dependency installation
WORKDIR /build
COPY requirements.txt .

# Install Python dependencies into user site-packages
# Using --user flag to install in /root/.local for easy copying
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install runtime dependencies
# - curl: Required for Docker HEALTHCHECK command
# - libgomp1: Required by LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib libraries from builder stage
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Add Python packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Create non-root user for security
# UID 1000 matches most host users for volume permission compatibility
RUN useradd -m -u 1000 -s /bin/bash tradingbot

# Copy application code
# Only copy trading directory (source code)
# models/ and data/ will be mounted as volumes
COPY --chown=tradingbot:tradingbot trading/ ./trading/

# Copy requirements.txt for reference
COPY --chown=tradingbot:tradingbot requirements.txt .

# Create directories for volumes
# These will be overridden by volume mounts in docker-compose.yml
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R tradingbot:tradingbot /app/models /app/data /app/logs

# Switch to non-root user
USER tradingbot

# Expose dashboard port
EXPOSE 5173

# Health check configuration
# Checks /health endpoint every 30s with 10s timeout
# Allows 60s startup period for model loading
# Retries 3 times before marking unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5173/health || exit 1

# Default command - run trading bot CLI
# This can be overridden in docker-compose.yml for different services
CMD ["python", "-m", "trading.cli", "run"]
