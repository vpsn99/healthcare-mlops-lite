import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_TABLES = [
    "patients.csv",
    "encounters.csv",
    "conditions.csv",
    "observations.csv",
]


def validate_month(month: str) -> None:
    if len(month) != 6 or not month.isdigit():
        raise ValueError("month must be YYYYMM (6 digits), e.g., 202602")


def month_has_required_files(month_dir: Path) -> bool:
    return all((month_dir / f).exists() for f in REQUIRED_TABLES)


def run_synthea(synthea_root: Path, patients: int) -> None:
    bat = synthea_root / "run_synthea.bat"
    if not bat.exists():
        raise FileNotFoundError(f"run_synthea.bat not found at: {bat}")

    # Run in synthea root so output/ is created in expected place
    cmd = [str(bat), "-p", str(patients)]
    print(f"Running: {' '.join(cmd)} (cwd={synthea_root})")
    result = subprocess.run(cmd, cwd=str(synthea_root), capture_output=True, text=True)

    if result.returncode != 0:
        print("Synthea failed.\n--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        raise RuntimeError(f"Synthea returned non-zero exit code: {result.returncode}")

    # Optional: print last few lines to show progress
    tail = "\n".join(result.stdout.splitlines()[-20:])
    if tail.strip():
        print("--- Synthea output (tail) ---")
        print(tail)


def write_outputs(
    synthea_root: Path,
    target_dir: Path,
    mode: str,   # "overwrite" or "append"
) -> None:
    csv_dir = synthea_root / "output" / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"Synthea CSV output folder not found: {csv_dir}")

    import pandas as pd  # local import to keep startup fast

    for fname in REQUIRED_TABLES:
        src = csv_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Expected output missing: {src}")

        dst = target_dir / fname

        new_df = pd.read_csv(src)

        if mode == "overwrite" or not dst.exists():
            new_df.to_csv(dst, index=False)
            continue

        if mode == "append":
            old_df = pd.read_csv(dst)

            combined = pd.concat([old_df, new_df], ignore_index=True)

            # De-dup if there is an Id column (Synthea usually has it)
            if "Id" in combined.columns:
                combined = combined.drop_duplicates(subset=["Id"], keep="first")

            combined.to_csv(dst, index=False)
            continue

        raise ValueError(f"Unknown mode: {mode}")


def archive_output(synthea_root: Path, archive_dir: Path) -> None:
    src = synthea_root / "output"
    if not src.exists():
        print("No synthea output folder to archive (output/ missing). Skipping archive.")
        return

    if archive_dir.exists():
        shutil.rmtree(archive_dir)

    shutil.copytree(src, archive_dir)
    print(f"Archived full Synthea output to: {archive_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic healthcare raw data for a month (YYYYMM) and store under data/raw/YYYYMM/"
    )
    p.add_argument("--month", required=True, help="Month partition in YYYYMM format, e.g. 202602")
    p.add_argument("--patients", type=int, default=200, help="Number of patients to generate (default: 200)")
    p.add_argument(
        "--synthea-root",
        default=r"D:\healthcare_project\synthea",
        help="Path to Synthea folder containing run_synthea.bat",
    )
    p.add_argument(
        "--project-root",
        default=r"D:\healthcare_project\healthcare-mlops-lite",
        help="Project root containing data/ folder",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing month files if they already exist",
    )
    p.add_argument(
        "--archive-output",
        action="store_true",
        help="Archive the full Synthea output folder under archives/synthea_output_<YYYYMM>",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append new rows to an existing month (dedup by Id when available)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.append and args.overwrite:
        raise ValueError("Use only one of --append or --overwrite (not both).")

    validate_month(args.month)

    synthea_root = Path(args.synthea_root)
    project_root = Path(args.project_root)

    month_dir = project_root / "data" / "raw" / args.month
    month_dir.mkdir(parents=True, exist_ok=True)

    exists = month_has_required_files(month_dir)

    if exists and not args.overwrite and not args.append:
        print(
            f"Month folder already has required files: {month_dir}\n"
            f"Use --overwrite to replace them or --append to add more rows."
        )
        return 2

    # Run synthea and copy required CSVs
    run_synthea(synthea_root, args.patients)
    mode = "overwrite" if args.overwrite else ("append" if args.append else "overwrite")
    write_outputs(synthea_root, month_dir, mode=mode)

    # Note: write_outputs will either overwrite or append based on the value of mode

    if args.archive_output:
        arch_dir = project_root / "archives" / f"synthea_output_{args.month}"
        arch_dir.parent.mkdir(parents=True, exist_ok=True)
        archive_output(synthea_root, arch_dir)

    print(f"Done. Created/updated: {month_dir}")
    print("Files:")
    for f in REQUIRED_TABLES:
        fp = month_dir / f
        print(f"  - {fp} ({fp.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())