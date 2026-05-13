# Dockerfile — Voice Feature Extraction API
# Port: 5800

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenSMILE and audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-multipart

# Copy application files
COPY app.py .
COPY .env .

# Expose port
EXPOSE 5800

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5800"]
