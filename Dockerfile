FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model artifact
COPY src /app/src
COPY models/registered /app/models/registered

# Expose for FastAPI
EXPOSE 8000

ENV PYTHONPATH=/app/src
ENV SERVE_MODEL_PATH=/app/models/registered/model.joblib
ENV SERVE_MODEL_METADATA_PATH=/app/models/registered/model_metadata.json
ENV PREDICTION_THRESHOLD=0.5

CMD ["uvicorn", "healthml.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]