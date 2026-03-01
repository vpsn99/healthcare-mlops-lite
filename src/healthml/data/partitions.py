from pathlib import Path
from typing import List, Optional


def list_partitions(raw_root: str, as_of: Optional[str] = None) -> List[str]:
    """
    raw_root contains subfolders named YYYYMM.
    If as_of is provided, return only partitions <= as_of.
    """
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    parts = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 6
    )

    if as_of:
        parts = [p for p in parts if p <= as_of]

    return parts


def latest_partition(raw_root: str) -> str:
    parts = list_partitions(raw_root)
    if not parts:
        raise FileNotFoundError(f"No YYYYMM partitions found under {raw_root}")
    return parts[-1]