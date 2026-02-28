from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # numeric
    age_years: float
    HEALTHCARE_EXPENSES: float
    HEALTHCARE_COVERAGE: float
    INCOME: float
    encounter_count: int
    avg_enc_duration_days: float
    active_span_days: float
    condition_count: int

    # categorical
    GENDER: str
    RACE: str
    ETHNICITY: str
    MARITAL: str
    STATE: str

    # Optional: client can pass it, but we won't require it
    patient_token: Optional[str] = Field(default=None)


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    mlflow_run_id: Optional[str] = None