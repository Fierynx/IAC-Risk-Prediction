FROM python:3.11-slim

# Install git since feature extraction might use it, and system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir imbalanced-learn

# Copy project files
COPY scripts/ /app/scripts/
COPY models/ /app/models/

# Rename extract_features.py if needed, or rely on python path
ENV PYTHONPATH="${PYTHONPATH}:/app/scripts"

ENTRYPOINT ["python", "/app/scripts/predict_risk.py"]
