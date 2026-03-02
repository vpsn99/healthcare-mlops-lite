import json
import shutil
from pathlib import Path

import mlflow


def _read_registered_metrics(registered_dir: Path) -> dict:
    meta_path = registered_dir / "model_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_run_metrics(run_id: str) -> dict:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return dict(run.data.metrics), dict(run.data.params)


def should_promote(
    candidate_metrics: dict,
    current_metrics: dict,
    metric: str = "pr_auc",
    min_delta: float = 0.0,
) -> bool:
    cand = candidate_metrics.get(metric)
    if cand is None:
        return False

    # reject NaN
    if cand != cand:
        return False

    curr = current_metrics.get(metric)

    # No current -> promote
    if curr is None or curr != curr:
        return True

    return cand >= (curr + min_delta)


def promote_run_to_registered(run_dir: Path, registered_dir: Path) -> None:
    """
    Copy run artifacts (model + metadata + confusion matrix) to registered directory.
    """
    registered_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["model.joblib", "model_metadata.json", "confusion_matrix.png"]:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, registered_dir / fname)


def maybe_promote(
    run_id: str,
    mlflow_tracking_uri: str,
    promote_metric: str = "pr_auc",
    min_delta: float = 0.0,
    models_root: str = "models",
) -> bool:
    """
    Returns True if promoted, else False.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    models_root = Path(models_root)
    run_dir = models_root / "runs" / run_id
    registered_dir = models_root / "registered"

    # Candidate metrics from MLflow
    run_metrics, _run_params = _get_run_metrics(run_id)

    # Current metrics from registered metadata (if any)
    current_meta = _read_registered_metrics(registered_dir)
    current_run_id = current_meta.get("run_id")
    current_metrics = {}
    if current_run_id:
        try:
            current_metrics, _ = _get_run_metrics(current_run_id)
        except Exception:
            current_metrics = {}

    if should_promote(run_metrics, current_metrics, metric=promote_metric, min_delta=min_delta):
        promote_run_to_registered(run_dir, registered_dir)
        return True

    return False