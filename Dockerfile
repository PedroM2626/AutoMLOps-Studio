# Use Python 3.11 slim image for better compatibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-base.txt requirements-full.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY free_mlops/ ./free_mlops/
COPY start_mlflow.py ./
COPY .env.example ./.env

# Create necessary directories
RUN mkdir -p ./data ./artifacts ./mlruns ./models

# Expose ports
EXPOSE 8501 5000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - Streamlit
CMD ["streamlit", "run", "free_mlops/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
