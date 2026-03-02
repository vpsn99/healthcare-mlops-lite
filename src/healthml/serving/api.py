import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from healthml.serving.predictor import Predictor
from healthml.serving.schemas import PredictRequest, PredictResponse

load_dotenv()

MODEL_PATH = os.getenv("SERVE_MODEL_PATH", "models/registered/model.joblib")
META_PATH = os.getenv("SERVE_MODEL_METADATA_PATH", "models/registered/model_metadata.json")
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

_predictor: Predictor | None = None


def get_predictor() -> Predictor | None:
    """
    Lazy-load predictor so that:
    - CI/tests can call /health without requiring model artifacts
    - Service can report 'degraded' when model isn't available
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    try:
        _predictor = Predictor(model_path=MODEL_PATH, metadata_path=META_PATH)
        return _predictor
    except FileNotFoundError:
        # Common in CI: model artifacts are not present
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try loading on startup (fast path), but don't crash the app if missing.
    get_predictor()
    yield
    # Cleanup
    global _predictor
    _predictor = None


app = FastAPI(
    title="Healthcare Readmission Risk API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Healthcare Readmission Risk API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    predictor = get_predictor()

    if predictor is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "model_path": MODEL_PATH,
            "model_run_id": os.getenv("HEALTHML_MODEL_RUN_ID") or None,
            "threshold": THRESHOLD,
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(predictor.model_path),
        "model_run_id": predictor.model_run_id,
        "threshold": predictor.threshold,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    payload = req.model_dump()
    payload.pop("patient_token", None)

    try:
        p = float(predictor.predict_proba(payload))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    pred = 1 if p >= THRESHOLD else 0

    # Keep your response field name as in your schema (mlflow_run_id)
    return PredictResponse(
        prediction=pred,
        probability=p,
        threshold=THRESHOLD,
        mlflow_run_id=(predictor.metadata.get("run_id") if getattr(predictor, "metadata", None) else predictor.model_run_id),
    )