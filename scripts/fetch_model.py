import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "build" / "model"

def main():
    """Download the champion model into build/model for baking into the serving image."""
    model_name = os.environ.get("MODEL_NAME", "churn-classifier")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

    mlflow.set_tracking_uri(uri)
    model_settings = mlflow.MlflowClient().get_model_version_by_alias(model_name, alias)

    model_uri = f"models:/{model_name}/{model_settings.version}"

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    for item in MODEL_PATH.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    mlflow.artifacts.download_artifacts(artifact_uri= model_uri, dst_path= MODEL_PATH)

    data = {
        "name": model_name,
        "version": model_settings.version,
        "threshold": float(model_settings.tags["threshold"]),
        "run_id": model_settings.run_id,
        "model_created_at": datetime.fromtimestamp(model_settings.creation_timestamp / 1000, tz=UTC).isoformat(),
        "build_fetched_at": datetime.now(UTC).isoformat(),
    }

    with open(MODEL_PATH / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()