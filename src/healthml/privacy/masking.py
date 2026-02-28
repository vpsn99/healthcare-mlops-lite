import hashlib
import hmac
import os

import pandas as pd


def pseudonymize_id(patient_id: str, secret: str) -> str:
    return hmac.new(
        secret.encode(),
        patient_id.encode(),
        hashlib.sha256
    ).hexdigest()


def mask_patients(df: pd.DataFrame) -> pd.DataFrame:
    secret = os.getenv("HEALTHML_HMAC_SECRET")
    if not secret:
        raise ValueError("HEALTHML_HMAC_SECRET not set")

    df = df.copy()

    # Create pseudonymized patient token
    df["patient_token"] = df["Id"].apply(lambda x: pseudonymize_id(str(x), secret))

    # Drop direct identifiers
    drop_cols = [
        # Unique IDs / direct identifiers
        "Id", "SSN",

        # Name-related
        "FIRST", "LAST", "PREFIX", "MIDDLE", "SUFFIX", "MAIDEN",

        # Government IDs
        "DRIVERS", "PASSPORT",

        # Contact / address
        "ADDRESS", "PHONE", "ZIP",

        # Precise location (remove or generalize)
        "LAT", "LON",

        # Optional: if you want stricter de-id, drop birthplace too
        "BIRTHPLACE",
    ]

    drop_cols += ["CITY", "COUNTY", "FIPS"]

    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)

    return df