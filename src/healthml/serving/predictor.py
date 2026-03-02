import json
import os
from pathlib import Path

import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path: str | None = None, metadata_path: str | None = None):
        """
        Load model either from:
          - HEALTHML_MODEL_RUN_ID -> models/runs/<run_id>/model.joblib
          - else model_path (default models/registered/model.joblib)
        """
        run_id = os.getenv("HEALTHML_MODEL_RUN_ID")

        if run_id:
            base = Path("models") / "runs" / run_id
            self.model_path = base / "model.joblib"
            self.metadata_path = base / "model_metadata.json"
        else:
            self.model_path = Path(model_path or "models/registered/model.joblib")
            self.metadata_path = Path(metadata_path or "models/registered/model_metadata.json")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)

        self.metadata = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.model_run_id = self.metadata.get("run_id", run_id)
        self.threshold = float(os.getenv("HEALTHML_THRESHOLD", "0.5"))

    def predict(self, features: dict) -> dict:
        # Expect a single-row dict; model pipeline handles preprocessing
        X = pd.DataFrame([features])

        proba = float(self.model.predict_proba(X)[:, 1][0])
        pred = int(proba >= self.threshold)

        return {
            "prediction": pred,
            "probability": proba,
            "threshold": self.threshold,
            "model_run_id": self.model_run_id,
        }

    def predict_proba(self, features: dict) -> float:
        # Convert dict -> DataFrame (1 row)
        df = pd.DataFrame([features])

        # If metadata includes expected feature columns, enforce ordering
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}")
            df = df[self.feature_cols]

        proba = float(self.model.predict_proba(df)[:, 1][0])
        return proba