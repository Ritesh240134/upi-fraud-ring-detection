"""
Tests for the serving layer.

The four endpoints that existed before the model was wired in - /health,
/rings, /explanations and /score - must keep working and keep their response
shape. The rest covers the model-backed additions.
"""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip(
    "httpx2",
    reason="Install httpx2 to run the API tests.",
)

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# ============================================================
# EXISTING ENDPOINTS MUST NOT REGRESS
# ============================================================

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_lists_endpoints(client):

    response = client.get("/")

    assert response.status_code == 200
    assert "endpoints" in response.json()


def test_rings(client):

    response = client.get("/rings")

    assert response.status_code == 200

    body = response.json()

    assert "detected_rings" in body
    assert isinstance(body["rings"], list)


def test_explanations(client):

    response = client.get("/explanations")

    assert response.status_code == 200

    body = response.json()

    assert "explanation_events" in body
    assert isinstance(body["explanations"], list)


def test_events_summary(client):

    response = client.get("/events/summary")

    assert response.status_code == 200
    assert response.json()["total_events"] > 0


def test_score_keeps_its_response_shape(client):

    response = client.post(
        "/score",
        json={
            "source_account": "ACC_00001",
            "target_account": "ACC_00002",
            "amount": 25000,
            "event_type": "TRANSFER",
        },
    )

    assert response.status_code == 200

    body = response.json()

    for field in (
        "source_account",
        "target_account",
        "amount",
        "event_type",
        "risk_score",
        "risk_level",
        "fraud_probability",
        "reasons",
        "timestamp",
        "model_status",
    ):
        assert field in body, f"missing response field: {field}"

    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_score_handles_unknown_accounts(client):
    """
    An account the model has never seen must degrade to the rule layer
    rather than error.
    """

    response = client.post(
        "/score",
        json={
            "source_account": "ACC_DOES_NOT_EXIST",
            "target_account": "ACC_ALSO_MISSING",
            "amount": 500,
            "event_type": "TRANSFER",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["source_account_score"] is None
    assert body["model_status"].startswith("rule_layer")


# ============================================================
# MODEL-BACKED ENDPOINTS
# ============================================================

def test_model_info(client):

    response = client.get("/model/info")

    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "temporal_hetero_gnn"
    assert "available" in body


def test_gnn_rings_route_is_not_shadowed(client):
    """
    /rings/gnn must resolve to the GNN ring listing, not to /rings/{ring_id}
    with ring_id="gnn".
    """

    response = client.get("/rings/gnn")

    assert response.status_code == 200

    body = response.json()

    assert body["detector"] == "temporal_hetero_gnn"
    assert isinstance(body["rings"], list)


def test_account_risk(client):

    from backend.model_service import model_service

    if not model_service.available:
        pytest.skip("Model artifacts not available.")

    account = next(iter(model_service.scores))

    response = client.get(f"/accounts/{account}/risk")

    assert response.status_code == 200

    body = response.json()

    assert body["account_id"] == account
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_unknown_account_risk_returns_404(client):

    from backend.model_service import model_service

    if not model_service.available:
        pytest.skip("Model artifacts not available.")

    response = client.get("/accounts/ACC_DOES_NOT_EXIST/risk")

    assert response.status_code == 404
