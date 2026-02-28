import json
from pathlib import Path

import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path: str, metadata_path: str | None = None):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)

        self.metadata = {}
        if self.metadata_path and self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        # If training wrote feature_cols into metadata, keep it
        self.feature_cols = self.metadata.get("feature_cols")

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