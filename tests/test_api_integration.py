import json
import os
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from src.api.main import app
from src.features.columns import ALL_FEATURES

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration  # applies the marker to every test in this file


@pytest.fixture(scope="module")
def champion_sample():
    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
        )
    )
    mlflow.set_experiment("churn-model-selection")
    model_name = os.environ.get("MODEL_NAME", "churn-classifier")
    alias = os.environ.get("MODEL_ALIAS", "champion")

    version = mlflow.MlflowClient().get_model_version_by_alias(model_name, alias)
    sample = mlflow.load_table("verification_sample.json", run_ids=[version.run_id])
    return version, sample.iloc[0]


def test_service_reproduces_the_training_score_and_records_it(champion_sample):
    version, row = champion_sample

    body = json.loads(row[ALL_FEATURES].to_json())
    body["customer_id"] = str(row["customerID"])

    raw_total = body["TotalCharges"]
    body["TotalCharges"] = (
        None if raw_total is None or str(raw_total).strip() == "" else float(raw_total)
    )

    with TestClient(app) as client:  # the context manager is what runs the lifespan
        response = client.post("/predict", json=body)

    assert response.status_code == 200
    answer = response.json()

    assert answer["probability"] == pytest.approx(row["expected_score"], abs=1e-9)
    assert int(answer["model_version"]) == int(version.version)

    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        stored = (
            conn.execute(
                text("SELECT * FROM predictions WHERE request_id = :request_id"),
                {"request_id": answer["request_id"]},
            )
            .mappings()
            .one()  # fails if there is no row, or more than one
        )

    assert stored["customer_id"] == answer["customer_id"]
    assert stored["prediction"] == answer["prediction"]
    assert float(stored["probability"]) == pytest.approx(answer["probability"], abs=1e-9)
    assert float(stored["threshold"]) == pytest.approx(answer["threshold"], abs=1e-9)