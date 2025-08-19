# PHMGA Production Deployment Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM python:3.9-slim as production

# Create non-root user
RUN groupadd -r phmga && useradd -r -g phmga phmga

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY tutorials/ ./tutorials/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Copy production configuration
COPY production/config/ ./production/config/
COPY production/scripts/ ./production/scripts/

# Set permissions
RUN chown -R phmga:phmga /app && \
    chmod +x ./production/scripts/*.sh

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R phmga:phmga /app/data /app/logs /app/cache

# Switch to non-root user
USER phmga

# Set environment variables
ENV PYTHONPATH=/app
ENV PHMGA_ENV=production
ENV PHMGA_LOG_LEVEL=INFO
ENV PHMGA_DATA_PATH=/app/data
ENV PHMGA_CACHE_PATH=/app/cache
ENV PHMGA_LOG_PATH=/app/logs

# Expose ports
EXPOSE 8000 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "tutorials.part3.production.main"]

# Labels for metadata
LABEL maintainer="PHMGA Tutorial Series"
LABEL version="1.0.0"
LABEL description="Production PHMGA Framework Container"
LABEL org.opencontainers.image.source="https://github.com/PHMbench/PHMGA"
