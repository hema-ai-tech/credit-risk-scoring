"""Tests for the FastAPI service. Skipped automatically if the model is not trained yet."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import MODELS_DIR

MODEL_READY = (MODELS_DIR / "best_model.joblib").exists() and (
    MODELS_DIR / "threshold.json"
).exists()
needs_model = pytest.mark.skipif(
    not MODEL_READY, reason="Model files missing. Run train.py and evaluate.py first."
)

client = TestClient(app)

HIGH_RISK = {
    "revolving_utilization": 1.2, "age": 25, "past_due_30_59": 2, "debt_ratio": 0.8,
    "monthly_income": 2000, "open_credit_lines": 4, "times_90_days_late": 2,
    "real_estate_loans": 0, "past_due_60_89": 1, "dependents": 2,
}
LOW_RISK = {
    "revolving_utilization": 0.1, "age": 52, "past_due_30_59": 0, "debt_ratio": 0.2,
    "monthly_income": 9000, "open_credit_lines": 8, "times_90_days_late": 0,
    "real_estate_loans": 1, "past_due_60_89": 0, "dependents": 1,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@needs_model
def test_predict_returns_expected_fields():
    response = client.post("/predict", json=LOW_RISK)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["decision"] in {"high_risk", "low_risk"}
    assert len(body["top_factors"]) == 3


@needs_model
def test_high_risk_scores_above_low_risk():
    high = client.post("/predict", json=HIGH_RISK).json()
    low = client.post("/predict", json=LOW_RISK).json()
    assert high["risk_score"] > low["risk_score"]


@needs_model
def test_optional_fields_can_be_omitted():
    payload = {k: v for k, v in LOW_RISK.items() if k not in {"monthly_income", "dependents"}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_underage_applicant_rejected():
    response = client.post("/predict", json={**LOW_RISK, "age": 10})
    assert response.status_code == 422


def test_missing_required_field_rejected():
    payload = {k: v for k, v in LOW_RISK.items() if k != "age"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422