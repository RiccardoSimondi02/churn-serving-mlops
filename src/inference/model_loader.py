import json
import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]

def return_model_info():
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        model_path = Path(model_path)
        model = mlflow.sklearn.load_model(model_path)
        with open(model_path / "model_meta.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        version = data["version"]
        threshold = data["threshold"]
    else:
        model_name = os.environ.get("MODEL_NAME", "churn-classifier")
        alias = os.environ.get("MODEL_ALIAS", "champion")
        uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

        mlflow.set_tracking_uri(uri)
        model_uri = f"models:/{model_name}@{alias}"
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        model = mlflow.sklearn.load_model(local_path)
        model_settings = mlflow.MlflowClient().get_model_version_by_alias(model_name, alias)
        version = model_settings.version
        threshold = model_settings.tags["threshold"]

    return model, version, threshold