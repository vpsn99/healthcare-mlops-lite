import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from healthml.serving.predictor import Predictor
from healthml.serving.schemas import PredictRequest, PredictResponse

load_dotenv()

app = FastAPI(title="Healthcare Readmission Risk API", version="1.0.0")

MODEL_PATH = os.getenv("SERVE_MODEL_PATH", "models/registered/model.joblib")
META_PATH = os.getenv("SERVE_MODEL_METADATA_PATH", "models/registered/model_metadata.json")
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

predictor: Predictor | None = None


@app.on_event("startup")
def _startup():
    global predictor
    predictor = Predictor(model_path=MODEL_PATH, metadata_path=META_PATH)


@app.get("/health")
def health():
    if predictor is None:
        return {"status": "not_ready"}
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_run_id": predictor.metadata.get("run_id"),
        "threshold": THRESHOLD
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert request to dict and drop patient_token (not used as a feature)
    payload = req.model_dump()
    payload.pop("patient_token", None)

    try:
        p = predictor.predict_proba(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    pred = 1 if p >= THRESHOLD else 0
    return PredictResponse(
        prediction=pred,
        probability=p,
        threshold=THRESHOLD,
        model_run_id=predictor.metadata.get("run_id")
    )