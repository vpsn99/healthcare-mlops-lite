import pandas as pd
from datetime import timedelta


def generate_shift_days(patient_token: str, max_days: int = 365) -> int:
    return abs(hash(patient_token)) % max_days


def shift_dates(df: pd.DataFrame, patient_col: str, date_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for idx, row in df.iterrows():
        token = row[patient_col]
        shift_days = generate_shift_days(token)

        for col in date_cols:
            if col in df.columns and pd.notnull(row[col]):
                df.at[idx, col] = row[col] + timedelta(days=shift_days)

    return df