import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/train.yaml")
    return ap.parse_args()


def main():
    load_dotenv()

    args = parse_args()
    cfg = load_config(args.config)

    project_name = cfg.get("project", {}).get("name", "healthcare-mlops-lite")
    seed = int(cfg.get("project", {}).get("random_seed", 42))

    paths = cfg.get("paths", {})
    features_dir = Path(paths.get("features_dir", "data/features/v1"))
    model_dir = ensure_dir(paths.get("model_dir", "models/registered"))
    mlflow_dir = Path(paths.get("mlflow_dir", "mlruns"))

    target_name = cfg.get("training", {}).get("target", {}).get("name", "readmit_30d")
    test_size = float(cfg.get("training", {}).get("split", {}).get("test_size", 0.2))

    model_params = cfg.get("training", {}).get("model", {}).get("params", {})
    max_iter = int(model_params.get("max_iter", 500))
    class_weight = model_params.get("class_weight", "balanced")

    data_path = features_dir / "patient_features.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing features file: {data_path}. "
            f"Run: python -m healthml.data.feature_builder"
        )

    df = pd.read_csv(data_path)

    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in {data_path}")

    # Define expected columns (safe defaults for v1 feature_builder output)
    categorical_cols = ["GENDER", "RACE", "ETHNICITY", "MARITAL", "STATE"]
    numeric_cols = [
        "age_years",
        "HEALTHCARE_EXPENSES",
        "HEALTHCARE_COVERAGE",
        "INCOME",
        "encounter_count",
        "avg_enc_duration_days",
        "active_span_days",
        "condition_count",
    ]

    # Keep only columns that exist (robust to minor dataset differences)
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    feature_cols = categorical_cols + numeric_cols
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    X = df[feature_cols].copy()
    y = df[target_name].astype(int)

    # Split (stratify if possible)
    stratify = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify
    )

    # Preprocessing
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols)
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=seed
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf)
    ])

    # MLflow setup (local filesystem backend)
    mlflow.set_tracking_uri(f"file:{mlflow_dir.as_posix()}")
    mlflow.set_experiment(project_name)

    with mlflow.start_run(run_name="logreg_readmit_v1") as run:
        run_id = run.info.run_id

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        ap = average_precision_score(y_test, proba)
        mlflow.log_metric("pr_auc", ap)

        # Metrics
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, proba)
            ap = average_precision_score(y_test, proba)
            mlflow.log_metric("pr_auc", ap)
        except ValueError:
            auc = float("nan")

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, pred)
        fig_path = model_dir / "confusion_matrix.png"
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()
        for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)], cm.flatten()):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # Log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("feature_cols", json.dumps(feature_cols))

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        if auc == auc:  # not NaN
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("pr_auc", ap)

        # Log artifacts
        mlflow.log_artifact(str(fig_path), artifact_path="evaluation")

        input_example = X_train.head(5)
        signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Export model locally for serving
        model_path = model_dir / "model.joblib"
        joblib.dump(pipeline, model_path)

        meta = {
            "run_id": run_id,
            "experiment": project_name,
            "model_path": str(model_path),
            "feature_cols": feature_cols,
            "target": target_name,
        }
        with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print("Training complete.")
        print("Run ID:", run_id)
        print("Saved model:", model_path)
        print("Metrics:", {"accuracy": acc, "f1": f1, "roc_auc": auc})


if __name__ == "__main__":
    main()