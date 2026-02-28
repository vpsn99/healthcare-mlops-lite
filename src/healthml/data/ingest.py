import os

import pandas as pd
from dotenv import load_dotenv

from healthml.privacy.date_shift import shift_dates
from healthml.privacy.masking import mask_patients, pseudonymize_id

load_dotenv()

RAW_DIR = "data/raw/v1"
MASKED_DIR = "data/masked/v1"


def run_masking_pipeline():
    os.makedirs(MASKED_DIR, exist_ok=True)

    secret = os.getenv("HEALTHML_HMAC_SECRET")
    if not secret:
        raise ValueError("HEALTHML_HMAC_SECRET not set")

    # Load raw data
    patients = pd.read_csv(f"{RAW_DIR}/patients.csv")
    encounters = pd.read_csv(f"{RAW_DIR}/encounters.csv")
    conditions = pd.read_csv(f"{RAW_DIR}/conditions.csv")
    observations = pd.read_csv(f"{RAW_DIR}/observations.csv")

    # Build token map (Id -> patient_token) from raw patients
    token_map = patients[["Id"]].copy()
    token_map["patient_token"] = token_map["Id"].astype(str).apply(
        lambda x: pseudonymize_id(x, secret)
    )

    # Mask patients (this should drop Id and direct identifiers)
    patients_masked = mask_patients(patients)

    # Merge token into other tables
    encounters = encounters.merge(token_map, left_on="PATIENT", right_on="Id", how="left")
    conditions = conditions.merge(token_map, left_on="PATIENT", right_on="Id", how="left")
    observations = observations.merge(token_map, left_on="PATIENT", right_on="Id", how="left")

    # Drop raw patient identifiers from other tables after token merge
    encounters = encounters.drop(columns=["PATIENT", "Id"], errors="ignore")
    conditions = conditions.drop(columns=["PATIENT", "Id"], errors="ignore")
    observations = observations.drop(columns=["PATIENT", "Id"], errors="ignore")

    # Date shifting
    patients_masked = shift_dates(patients_masked, "patient_token", ["BIRTHDATE"])
    encounters = shift_dates(encounters, "patient_token", ["START", "STOP"])

    # Save masked datasets
    patients_masked.to_csv(f"{MASKED_DIR}/patients_masked.csv", index=False)
    encounters.to_csv(f"{MASKED_DIR}/encounters_masked.csv", index=False)
    conditions.to_csv(f"{MASKED_DIR}/conditions_masked.csv", index=False)
    observations.to_csv(f"{MASKED_DIR}/observations_masked.csv", index=False)

    print("Masking pipeline completed successfully.")


if __name__ == "__main__":
    run_masking_pipeline()