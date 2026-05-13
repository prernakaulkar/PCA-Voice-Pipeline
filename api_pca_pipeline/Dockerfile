# Dockerfile — PCA Pipeline API
# Port: 5900

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY .env .

# Mount point for models folder (mapped via docker-compose)
VOLUME ["/app/models"]

# Expose port
EXPOSE 5900

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5900"]
