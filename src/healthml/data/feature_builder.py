import os
import pandas as pd
from datetime import datetime

MASKED_DIR = "data/masked/v1"
FEATURES_DIR = "data/features/v1"


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def build_readmit_label(encounters: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level label: 1 if ANY qualifying inpatient stay is followed by another qualifying inpatient stay
    within 30 days of discharge.

    Qualifying inpatient stay filters reduce synthetic "chaining" artifacts:
      - ENCOUNTERCLASS == inpatient
      - length of stay >= 1 day
      - gap between discharge and next admission >= 1 day
    """
    enc = encounters.copy()

    # Require ENCOUNTERCLASS if present
    if "ENCOUNTERCLASS" in enc.columns:
        enc["ENCOUNTERCLASS"] = enc["ENCOUNTERCLASS"].astype(str).str.lower()
        enc = enc[enc["ENCOUNTERCLASS"] == "inpatient"].copy()

    enc["START"] = _to_dt(enc.get("START"))
    enc["STOP"] = _to_dt(enc.get("STOP"))
    enc = enc.dropna(subset=["patient_token", "START", "STOP"])

    # Length of stay (days)
    enc["los_days"] = (enc["STOP"] - enc["START"]).dt.total_seconds() / 86400.0
    enc = enc[enc["los_days"] >= 2.0].copy()

    enc = enc.sort_values(["patient_token", "START"])
    enc["NEXT_START"] = enc.groupby("patient_token")["START"].shift(-1)

    enc["gap_days"] = (enc["NEXT_START"] - enc["STOP"]).dt.total_seconds() / 86400.0

    # within 30 days AND not same-day/next-minute
    enc["readmit_30d_event"] = enc["gap_days"].between(2, 30, inclusive="both")

    labels = (
        enc.groupby("patient_token")["readmit_30d_event"]
        .any()
        .astype(int)
        .reset_index()
        .rename(columns={"readmit_30d_event": "readmit_30d"})
    )
    return labels


def build_patient_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    patients = pd.read_csv(f"{MASKED_DIR}/patients_masked.csv")
    encounters = pd.read_csv(f"{MASKED_DIR}/encounters_masked.csv")
    conditions = pd.read_csv(f"{MASKED_DIR}/conditions_masked.csv")

    # --- Patient features ---
    patients["BIRTHDATE"] = _to_dt(patients["BIRTHDATE"])
    today = pd.Timestamp(datetime.utcnow().date())
    patients["age_years"] = ((today - patients["BIRTHDATE"]).dt.days / 365.25).round(1)

    patient_feats = patients[[
        "patient_token",
        "age_years",
        "GENDER",
        "RACE",
        "ETHNICITY",
        "MARITAL",
        "STATE",
        "HEALTHCARE_EXPENSES",
        "HEALTHCARE_COVERAGE",
        "INCOME",
    ]].copy()

    # --- Encounter aggregate features ---
    encounters["START"] = _to_dt(encounters.get("START"))
    encounters["STOP"] = _to_dt(encounters.get("STOP"))
    encounters["enc_duration_days"] = (
        (encounters["STOP"] - encounters["START"]).dt.total_seconds() / 86400.0
    )

    enc_agg = encounters.groupby("patient_token").agg(
        encounter_count=("START", "count"),
        avg_enc_duration_days=("enc_duration_days", "mean"),
        first_encounter=("START", "min"),
        last_encounter=("START", "max"),
    ).reset_index()

    enc_agg["active_span_days"] = (
        (enc_agg["last_encounter"] - enc_agg["first_encounter"])
        .dt.total_seconds() / 86400.0
    )

    # --- Condition aggregate features ---
    # Synthea conditions.csv typically has a CODE/DESCRIPTION field; we just count rows per patient for now
    cond_agg = conditions.groupby("patient_token").agg(
        condition_count=("patient_token", "count")
    ).reset_index()

    # --- Label ---
    labels = build_readmit_label(encounters)

    # --- Join everything ---
    df = patient_feats.merge(enc_agg, on="patient_token", how="left")
    df = df.merge(cond_agg, on="patient_token", how="left")
    df = df.merge(labels, on="patient_token", how="left")

    # Fill missing aggregates (patients with no encounters/conditions)
    df["encounter_count"] = df["encounter_count"].fillna(0).astype(int)
    df["avg_enc_duration_days"] = df["avg_enc_duration_days"].fillna(0.0)
    df["active_span_days"] = df["active_span_days"].fillna(0.0)
    df["condition_count"] = df["condition_count"].fillna(0).astype(int)
    df["readmit_30d"] = df["readmit_30d"].fillna(0).astype(int)

    vc = df["readmit_30d"].value_counts(dropna=False)
    print("Label distribution:\n", vc)
    if df["readmit_30d"].nunique() < 2:
        raise ValueError(
            "Degenerate labels: readmit_30d has only one class. "
            "Generate more patients (e.g., -p 2000) or relax/tighten rules."
        )

    out_path = f"{FEATURES_DIR}/patient_features.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote features: {out_path}")
    print("Shape:", df.shape)
    print("Readmit rate:", df["readmit_30d"].mean())


if __name__ == "__main__":
    build_patient_features()