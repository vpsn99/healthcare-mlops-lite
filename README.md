# Healthcare MLOps Lite

A production-style, end-to-end MLOps project built using synthetic
healthcare data generated with Synthea (https://github.com/synthetichealth/synthea).

This project demonstrates a complete ML lifecycle including:

-   Partitioned raw data ingestion (YYYYMM format)
-   Privacy-preserving masking with deterministic pseudonymization
-   Time-aware feature engineering
-   Rare-class safe model training
-   MLflow experiment tracking
-   Automatic model promotion based on PR-AUC
-   Versioned run artifacts
-   FastAPI-based model serving
-   Docker containerization
-   CI pipeline with linting and testing

------------------------------------------------------------------------

## Project Structure

    data/
      raw/YYYYMM/                # Monthly raw Synthea data
      masked/YYYYMM/             # Privacy-masked partitions
      features/YYYYMM/           # Engineered features per as_of date

    models/
      runs/<run_id>/             # Immutable model artifacts per training run
      registered/                # Current champion model

    src/healthml/
      data/                      # Ingestion & feature engineering
      privacy/                   # Masking & date shifting
      train/                     # Training & promotion logic
      serving/                   # FastAPI inference layer

    configs/
      train.yaml                 # Training configuration

    scripts/
      run_backtest.py            # Multi-month backtesting utility

------------------------------------------------------------------------

## Training

Train using a specific data partition:

``` bash
python -m healthml.train.train --config configs/train.yaml --as-of 202601
```

The system: - Logs metrics to MLflow - Saves artifacts to
`models/runs/<run_id>/` - Automatically promotes the best model to
`models/registered/`

Promotion metric: **PR-AUC** (recommended for imbalanced datasets)

------------------------------------------------------------------------

## Serving

Start the API:

``` bash
uvicorn healthml.serving.api:app --host 127.0.0.1 --port 8000
```

Health endpoint:

    GET /health

Prediction endpoint:

    POST /predict

------------------------------------------------------------------------

## CI/CD

Includes: - Ruff linting - Pytest test suite - Coverage reporting -
Docker build validation

------------------------------------------------------------------------

## Notes

-   Models are versioned and committed to Git for reproducibility.
-   Data is synthetic (Synthea) and contains no real patient
    information.
-   Designed for educational and portfolio demonstration purposes.

------------------------------------------------------------------------

Virendra Pratap Singh
Senior Data Architect | Data Engineering | Analytics Platforms
https://www.linkedin.com/in/virendra-pratap-singh-iitg/
