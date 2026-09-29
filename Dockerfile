# Use a lightweight base image with Python 3.13
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install system dependencies some libraries need (e.g. LightGBM, XGBoost, OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install the Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create a non-root user and directories with minimal required permissions
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --create-home appuser \
    && mkdir -p mlruns data_lake models \
    && chown -R appuser:appgroup /app

USER appuser

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Default command to run Streamlit on the port Hugging Face Spaces expects
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
