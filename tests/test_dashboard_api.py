"""
Tests for the /api/ endpoints backing the web console, and for the
console's own static assets being served.
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
# CONSOLE ASSETS
# ============================================================

def test_dashboard_page_is_served(client):

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "UPI Fraud Console" in response.text


def test_static_assets_are_served(client):

    for path in ("/static/styles.css", "/static/app.js"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert len(response.content) > 500, path


def test_root_advertises_the_dashboard(client):

    body = client.get("/").json()

    assert body["endpoints"]["dashboard"] == "/dashboard"


# ============================================================
# OVERVIEW
# ============================================================

def test_overview(client):

    response = client.get("/api/overview")

    assert response.status_code == 200

    body = response.json()

    for field in (
        "accounts",
        "fraud_accounts",
        "events",
        "rings_detected",
        "accounts_in_rings",
        "true_fraud_in_rings",
        "event_types",
        "model",
    ):
        assert field in body, f"missing field: {field}"

    assert body["accounts"] > 0
    assert body["events"] > 0
    assert isinstance(body["event_types"], dict)


def test_overview_ring_precision_is_a_fraction(client):

    body = client.get("/api/overview").json()

    precision = body["ring_precision"]

    if precision is not None:
        assert 0.0 <= precision <= 1.0
        # Accounts inside rings cannot exceed the accounts flagged.
        assert body["true_fraud_in_rings"] <= body["accounts_in_rings"]


# ============================================================
# METRICS
# ============================================================

def test_metrics(client):

    response = client.get("/api/metrics")

    assert response.status_code == 200

    body = response.json()

    assert body["models"], "no model metrics found"
    assert body["caveat"]

    for model in body["models"]:

        assert model["name"]
        assert "confusion_matrix" in model

        for key in ("accuracy", "precision", "recall", "f1"):
            value = model["test"][key]
            assert value is None or 0.0 <= value <= 1.0


def test_metrics_confusion_matrix_totals_match_the_split(client):
    """
    Every model is evaluated on the same test split, so each confusion
    matrix must sum to the same number of accounts.
    """

    body = client.get("/api/metrics").json()

    expected = body["test_accounts"]

    if expected is None:
        pytest.skip("Test split size unavailable.")

    for model in body["models"]:

        matrix = model["confusion_matrix"]

        total = (
            matrix["true_negative"]
            + matrix["false_positive"]
            + matrix["false_negative"]
            + matrix["true_positive"]
        )

        assert total == expected, model["name"]


# ============================================================
# RINGS
# ============================================================

def test_ring_list(client):

    response = client.get("/api/rings")

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == len(body["rings"])

    for ring in body["rings"]:
        assert ring["detected_ring_id"]
        assert ring["ring_size"] >= 3
        assert isinstance(ring["account_list"], list)
        assert len(ring["account_list"]) == ring["ring_size"]


def test_ring_detail(client):

    rings = client.get("/api/rings").json()["rings"]

    if not rings:
        pytest.skip("No rings detected.")

    ring_id = rings[0]["detected_ring_id"]

    response = client.get(f"/api/rings/{ring_id}")

    assert response.status_code == 200

    body = response.json()

    assert body["ring"]["detected_ring_id"] == ring_id
    assert len(body["nodes"]) == body["ring"]["ring_size"]

    accounts = {node["account"] for node in body["nodes"]}

    # Every edge must connect two accounts inside the ring.
    for edge in body["edges"]:
        assert edge["source"] in accounts
        assert edge["target"] in accounts

    for node in body["nodes"]:
        assert 0.0 <= node["score"] <= 1.0
        assert node["true_label"] in (0, 1)


def test_unknown_ring_returns_404(client):

    response = client.get("/api/rings/GNN_RING_DOES_NOT_EXIST")

    assert response.status_code == 404


# ============================================================
# ACCOUNTS
# ============================================================

def test_account_list(client):

    response = client.get("/api/accounts?limit=10")

    assert response.status_code == 200

    body = response.json()

    assert body["returned"] <= 10
    assert len(body["accounts"]) == body["returned"]

    scores = [row["fraud_probability"] for row in body["accounts"]]

    # Default sort is by descending probability.
    assert scores == sorted(scores, reverse=True)


def test_account_level_filter(client):

    body = client.get("/api/accounts?level=HIGH&limit=50").json()

    for row in body["accounts"]:
        assert row["risk_level"] == "HIGH"


def test_account_search(client):

    first = client.get("/api/accounts?limit=1").json()["accounts"]

    if not first:
        pytest.skip("No accounts scored.")

    account_id = first[0]["account"]

    body = client.get(f"/api/accounts?search={account_id}").json()

    assert body["total"] >= 1
    assert any(row["account"] == account_id for row in body["accounts"])


def test_account_pagination_does_not_repeat(client):

    page_one = client.get("/api/accounts?limit=20&offset=0").json()
    page_two = client.get("/api/accounts?limit=20&offset=20").json()

    ids_one = {row["account"] for row in page_one["accounts"]}
    ids_two = {row["account"] for row in page_two["accounts"]}

    assert not (ids_one & ids_two)


def test_accounts_summary(client):

    response = client.get("/api/accounts-summary")

    assert response.status_code == 200

    body = response.json()

    assert body["total"] > 0
    assert len(body["histogram"]) == 20

    # The histogram must account for every scored account exactly once.
    assert sum(bin["count"] for bin in body["histogram"]) == body["total"]

    assert sum(body["levels"].values()) == body["total"]


# ============================================================
# SCORING CONSISTENCY
# ============================================================

def test_risk_level_matches_the_reported_score(client):
    """
    The band must be derived from the same rounded value that is returned,
    otherwise a response can read `0.4` next to `LOW`.
    """

    cases = [
        ("ACC_UNSEEN_A", "ACC_UNSEEN_B", 120000),
        ("ACC_UNSEEN_A", "ACC_UNSEEN_B", 75000),
        ("ACC_UNSEEN_A", "ACC_UNSEEN_B", 500),
    ]

    for source, target, amount in cases:

        body = client.post(
            "/score",
            json={
                "source_account": source,
                "target_account": target,
                "amount": amount,
                "event_type": "TRANSFER",
            },
        ).json()

        score = body["risk_score"]

        expected = (
            "HIGH" if score >= 0.70
            else "MEDIUM" if score >= 0.40
            else "LOW"
        )

        assert body["risk_level"] == expected, (
            f"{score} reported as {body['risk_level']}"
        )
