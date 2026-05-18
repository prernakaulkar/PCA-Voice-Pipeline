# Dockerfile — PCA Pipeline API
# Port: 5900

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY api_pca_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_pca_pipeline/app.py .
COPY api_pca_pipeline/.env .

# Bake models directly into container
COPY models/ /app/models/

EXPOSE 5900

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5900"]