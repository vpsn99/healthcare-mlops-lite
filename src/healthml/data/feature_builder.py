import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from healthml.data.partitions import latest_partition, list_partitions


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build patient-level features from masked monthly partitions (YYYYMM).")
    ap.add_argument("--masked-root", default="data/masked", help="Root folder containing masked YYYYMM partitions")
    ap.add_argument("--as-of", default=None, help="YYYYMM cutoff (use this masked partition). Default: latest")
    ap.add_argument("--out-root", default="data/features", help="Root folder for feature outputs")
    return ap.parse_args()


def build_readmit_label(encounters: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level label: 1 if ANY qualifying inpatient stay is followed by another qualifying inpatient stay
    within 30 days of discharge.

    Filters to reduce synthetic chaining artifacts:
      - ENCOUNTERCLASS == inpatient (if present)
      - length of stay >= 2 days
      - gap between discharge and next admission between 2 and 30 days
    """
    enc = encounters.copy()

    if "ENCOUNTERCLASS" in enc.columns:
        enc["ENCOUNTERCLASS"] = enc["ENCOUNTERCLASS"].astype(str).str.lower()
        enc = enc[enc["ENCOUNTERCLASS"] == "inpatient"].copy()

    enc["START"] = _to_dt(enc.get("START"))
    enc["STOP"] = _to_dt(enc.get("STOP"))
    enc = enc.dropna(subset=["patient_token", "START", "STOP"])

    enc["los_days"] = (enc["STOP"] - enc["START"]).dt.total_seconds() / 86400.0
    enc = enc[enc["los_days"] >= 2.0].copy()

    enc = enc.sort_values(["patient_token", "START"])
    enc["NEXT_START"] = enc.groupby("patient_token")["START"].shift(-1)
    enc["gap_days"] = (enc["NEXT_START"] - enc["STOP"]).dt.total_seconds() / 86400.0

    enc["readmit_30d_event"] = enc["gap_days"].between(2, 30, inclusive="both")

    labels = (
        enc.groupby("patient_token")["readmit_30d_event"]
        .any()
        .astype(int)
        .reset_index()
        .rename(columns={"readmit_30d_event": "readmit_30d"})
    )
    return labels


def build_patient_features(masked_root: str, as_of: str | None, out_root: str) -> None:
    masked_root_p = Path(masked_root)
    chosen_as_of = as_of or latest_partition(masked_root)

    # Optional: show available masked partitions <= as_of (useful for debugging)
    available = list_partitions(masked_root, as_of=chosen_as_of)
    if chosen_as_of not in available:
        raise FileNotFoundError(f"Masked partition {chosen_as_of} not found under {masked_root}")

    in_dir = masked_root_p / chosen_as_of
    out_dir = Path(out_root) / chosen_as_of
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(in_dir / "patients_masked.csv")
    encounters = pd.read_csv(in_dir / "encounters_masked.csv")
    conditions = pd.read_csv(in_dir / "conditions_masked.csv")

    # --- Patient features ---
    patients["BIRTHDATE"] = _to_dt(patients["BIRTHDATE"])
    today = pd.Timestamp(datetime.utcnow().date())
    patients["age_years"] = ((today - patients["BIRTHDATE"]).dt.days / 365.25).round(1)

    patient_feats = patients[
        [
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
        ]
    ].copy()

    # --- Encounter aggregates ---
    encounters["START"] = _to_dt(encounters.get("START"))
    encounters["STOP"] = _to_dt(encounters.get("STOP"))
    encounters["enc_duration_days"] = (encounters["STOP"] - encounters["START"]).dt.total_seconds() / 86400.0

    enc_agg = (
        encounters.groupby("patient_token")
        .agg(
            encounter_count=("START", "count"),
            avg_enc_duration_days=("enc_duration_days", "mean"),
            first_encounter=("START", "min"),
            last_encounter=("START", "max"),
        )
        .reset_index()
    )

    enc_agg["active_span_days"] = (enc_agg["last_encounter"] - enc_agg["first_encounter"]).dt.total_seconds() / 86400.0

    # --- Condition aggregates ---
    cond_agg = (
        conditions.groupby("patient_token")
        .agg(condition_count=("patient_token", "count"))
        .reset_index()
    )

    # --- Label ---
    labels = build_readmit_label(encounters)

    # --- Join ---
    df = patient_feats.merge(enc_agg, on="patient_token", how="left")
    df = df.merge(cond_agg, on="patient_token", how="left")
    df = df.merge(labels, on="patient_token", how="left")

    # Fill missing
    df["encounter_count"] = df["encounter_count"].fillna(0).astype(int)
    df["avg_enc_duration_days"] = df["avg_enc_duration_days"].fillna(0.0)
    df["active_span_days"] = df["active_span_days"].fillna(0.0)
    df["condition_count"] = df["condition_count"].fillna(0).astype(int)
    df["readmit_30d"] = df["readmit_30d"].fillna(0).astype(int)

    # Guardrail: require both classes (or at least warn)
    vc = df["readmit_30d"].value_counts(dropna=False)
    print("Label distribution:\n", vc)
    if df["readmit_30d"].nunique() < 2:
        print("WARNING: readmit_30d has only one class for this as-of dataset.")

    out_path = out_dir / "patient_features.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote features: {out_path}")
    print("Shape:", df.shape)
    print("as_of:", chosen_as_of)
    print("masked_partitions_available_up_to_as_of:", available)


def main() -> None:
    args = parse_args()
    build_patient_features(masked_root=args.masked_root, as_of=args.as_of, out_root=args.out_root)


if __name__ == "__main__":
    main()