import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from healthml.data.partitions import list_partitions, latest_partition
from healthml.privacy.date_shift import shift_dates
from healthml.privacy.masking import mask_patients, pseudonymize_id

load_dotenv()


REQUIRED_TABLES = ["patients", "encounters", "conditions", "observations"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mask raw healthcare data from monthly partitions (YYYYMM).")
    ap.add_argument("--raw-root", default="data/raw", help="Root folder containing YYYYMM partitions")
    ap.add_argument("--as-of", default=None, help="YYYYMM cutoff (include partitions <= as-of). Default: latest")
    ap.add_argument("--out-root", default="data/masked", help="Root output folder for masked data")
    return ap.parse_args()


def read_concat(raw_root: Path, partitions: list[str], table: str) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for part in partitions:
        fp = raw_root / part / f"{table}.csv"
        if fp.exists():
            dfs.append(pd.read_csv(fp))
    if not dfs:
        raise FileNotFoundError(f"No files found for table={table} in partitions={partitions}")
    return pd.concat(dfs, ignore_index=True)


def run_masking_pipeline(raw_root: str, as_of: str | None, out_root: str) -> None:
    secret = os.getenv("HEALTHML_HMAC_SECRET")
    if not secret:
        raise ValueError("HEALTHML_HMAC_SECRET not set")

    raw_root_p = Path(raw_root)
    chosen_as_of = as_of or latest_partition(raw_root)
    partitions = list_partitions(raw_root, as_of=chosen_as_of)

    if not partitions:
        raise FileNotFoundError(f"No partitions found up to as_of={chosen_as_of} under {raw_root}")

    out_dir = Path(out_root) / chosen_as_of
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + concat across partitions
    patients = read_concat(raw_root_p, partitions, "patients")
    encounters = read_concat(raw_root_p, partitions, "encounters")
    conditions = read_concat(raw_root_p, partitions, "conditions")
    observations = read_concat(raw_root_p, partitions, "observations")

    # Token map from raw patients for joining into other tables
    token_map = patients[["Id"]].copy()
    token_map["patient_token"] = token_map["Id"].astype(str).apply(lambda x: pseudonymize_id(x, secret))

    # Mask patients (drops Id and PHI, keeps patient_token)
    patients_masked = mask_patients(patients)

    # Join token into other tables
    encounters = encounters.merge(token_map, left_on="PATIENT", right_on="Id", how="left")
    conditions = conditions.merge(token_map, left_on="PATIENT", right_on="Id", how="left")
    observations = observations.merge(token_map, left_on="PATIENT", right_on="Id", how="left")

    # Drop raw identifiers after merge
    encounters = encounters.drop(columns=["PATIENT", "Id"], errors="ignore")
    conditions = conditions.drop(columns=["PATIENT", "Id"], errors="ignore")
    observations = observations.drop(columns=["PATIENT", "Id"], errors="ignore")

    # Date shift (deterministic per patient_token)
    patients_masked = shift_dates(patients_masked, "patient_token", ["BIRTHDATE"])
    encounters = shift_dates(encounters, "patient_token", ["START", "STOP"])

    # Write masked outputs for this as_of
    patients_masked.to_csv(out_dir / "patients_masked.csv", index=False)
    encounters.to_csv(out_dir / "encounters_masked.csv", index=False)
    conditions.to_csv(out_dir / "conditions_masked.csv", index=False)
    observations.to_csv(out_dir / "observations_masked.csv", index=False)

    print(f"Masking complete. as_of={chosen_as_of}")
    print(f"Partitions used: {partitions}")
    print(f"Output folder: {out_dir}")


def main() -> None:
    args = parse_args()
    run_masking_pipeline(raw_root=args.raw_root, as_of=args.as_of, out_root=args.out_root)


if __name__ == "__main__":
    main()