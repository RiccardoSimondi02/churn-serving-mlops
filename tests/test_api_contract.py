import pytest
from fastapi.testclient import TestClient

from src.api.main import PredictRequest, app


@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def valid_payload():
    """A request the contract accepts. Every test breaks exactly one field of it."""
    return {
        "customer_id": "TEST-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }

def test_unknown_category_is_rejected(client, valid_payload):
    valid_payload["Contract"] = "Three year"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert "Contract" in response.text


def test_null_total_charges_requires_zero_tenure(client, valid_payload):
    valid_payload["TotalCharges"] = None
    valid_payload["tenure"] = 5

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422

def test_missing_mandatory_field(client, valid_payload):
    del valid_payload["customer_id"]

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422

def test_wrong_typing(client, valid_payload):
    valid_payload["tenure"] = "test"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422

def test_null_total_charges_is_accepted_for_new_customers(valid_payload):
    valid_payload["TotalCharges"] = None
    valid_payload["tenure"] = 0

    request = PredictRequest(**valid_payload) 

    assert request.TotalCharges is None