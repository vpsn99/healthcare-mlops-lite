import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True, help="List of YYYYMM months to run as-of backtests for")
    ap.add_argument("--config", default="configs/train.yaml", help="Train config path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    for m in args.months:
        run([sys.executable, "-m", "healthml.data.ingest", "--as-of", m])
        run([sys.executable, "-m", "healthml.data.feature_builder", "--as-of", m])
        run([sys.executable, "-m", "healthml.train.train", "--config", args.config, "--as-of", m])

    print("\nBacktest complete for months:", args.months)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())