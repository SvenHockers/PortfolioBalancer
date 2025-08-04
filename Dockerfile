# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata labels
LABEL maintainer="Portfolio Rebalancer Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="portfolio-rebalancer" \
      org.label-schema.description="Automated portfolio rebalancing system" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-org/portfolio-rebalancer" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r portfolio && useradd -r -g portfolio -u 1000 portfolio

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY .env.example ./

# Create necessary directories including cache for yfinance
RUN mkdir -p data logs cache/yfinance && \
    chown -R portfolio:portfolio /app && \
    chmod -R 755 /app

# Switch to non-root user
USER portfolio

# Set Python path and yfinance cache location
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV YFINANCE_CACHE_DIR=/app/cache/yfinance

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from src.portfolio_rebalancer.common.config import get_config; get_config()" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "src.portfolio_rebalancer.scheduler"]

# Development stage
FROM production as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to portfolio user
USER portfolio

# Override command for development
CMD ["python", "-c", "print('Development container ready. Use docker exec to run commands.')"]

# Testing stage
FROM development as testing

# Copy test configuration
COPY pytest.ini .
COPY .coveragerc .

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]