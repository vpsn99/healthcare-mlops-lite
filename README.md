Run Locally (Windows):
1) Create venv and install dependencies:
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

2) Generate synthetic EHR data using Synthea (separate folder), then copy:
   patients.csv, encounters.csv, conditions.csv, observations.csv -> data/raw/v1/

3) Create masked dataset:
   $env:PYTHONPATH="src"
   python -m healthml.data.ingest

4) Build features + readmission label:
   python -m healthml.data.feature_builder

5) Train + track in MLflow:
   python -m healthml.train.train --config configs/train.yaml
   mlflow ui --backend-store-uri ./mlruns

6) Serve API:
   uvicorn healthml.serving.api:app --reload --port 8000

7) Test:
   curl.exe http://127.0.0.1:8000/health
   .\scripts\smoke_test_api.ps1

Run with docker:
docker build -f docker\Dockerfile -t healthml-api .
docker run -p 8000:8000 healthml-api
curl.exe http://127.0.0.1:8000/health

