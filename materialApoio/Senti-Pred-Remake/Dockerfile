# Use a base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables (should be provided at runtime via .env or docker run)
ENV MLFLOW_TRACKING_URI=""
ENV DAGSHUB_REPO_OWNER=""
ENV DAGSHUB_REPO_NAME=""

# Command to run the training script
CMD ["python", "train_sentiment.py"]
