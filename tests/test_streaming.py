"""
Tests for the live streaming detector.

Several of these are regression tests for bugs found while building it:

  * events whose timestamp string was formatted differently from the seeded
    data were silently coerced to NaT and dropped;
  * the rescore trigger compared a running total against a modulus, so with
    a batch size that never landed on a multiple it never fired at all;
  * time was normalised against the min/max of whatever was in the buffer,
    so ingesting any event retroactively rescaled every historical edge and
    moved unrelated accounts' scores.
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
from backend.stream_service import streaming_detector  # noqa: E402


BASE_TIME = "2026-01-01T22:20:00"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module", autouse=True)
def require_streaming(client):

    state = client.get("/api/stream/state").json()

    if not state.get("available"):
        pytest.skip(f"Streaming unavailable: {state.get('error')}")


@pytest.fixture
def reset(client):
    """
    Return to the seeded baseline before each test that mutates state.
    """

    client.post("/api/stream/reset")
    yield
    client.post("/api/stream/reset")


def event(**overrides) -> dict:

    payload = {
        "event_id": "test-event",
        "timestamp": BASE_TIME,
        "event_type": "TRANSFER",
        "source_account": "ACC_00164",
        "target_account": "ACC_00188",
        "device_id": "DEV_0001",
        "source_vpa": "VPA_00164",
        "target_vpa": "VPA_00188",
        "amount": 5000.0,
    }

    payload.update(overrides)

    return payload


# ============================================================
# STATE
# ============================================================

def test_state_reports_a_seeded_baseline(client):

    state = client.get("/api/stream/state").json()

    assert state["available"] is True
    assert state["events_total"] > 0
    assert state["accounts_known"] > 0
    assert 0.0 < state["threshold"] <= 1.0


def test_state_is_reported_even_when_unavailable(client):
    """
    /api/stream/state must answer regardless, so the console can show why
    streaming is off rather than failing to render.
    """

    response = client.get("/api/stream/state")

    assert response.status_code == 200
    assert "available" in response.json()


# ============================================================
# INGEST
# ============================================================

def test_ingest_adds_events(client, reset):

    before = client.get("/api/stream/state").json()["events_total"]

    response = client.post(
        "/api/stream/events",
        json={"events": [event()], "rescore": False},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["accepted"] == 1
    assert body["total_events"] == before + 1


def test_alternative_timestamp_format_is_not_dropped(client, reset):
    """
    Regression: a space separator instead of 'T' used to be coerced to NaT
    by the inferred single format, and the event vanished with no error.
    """

    before = client.get("/api/stream/state").json()["accounts_known"]

    response = client.post(
        "/api/stream/events",
        json={
            "events": [
                event(
                    timestamp="2026-01-01 22:21:00",
                    source_account="ACC_FMT_0001",
                    target_account="ACC_FMT_0002",
                )
            ],
            "rescore": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["accepted"] == 1
    assert not response.json()["rejected"]

    after = client.get("/api/stream/state").json()["accounts_known"]

    # Both endpoints are new, so the graph must have grown by two.
    assert after == before + 2


def test_unparseable_timestamp_is_rejected_not_silently_dropped(
    client, reset
):

    response = client.post(
        "/api/stream/events",
        json={
            "events": [event(timestamp="definitely not a date")],
            "rescore": False,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["accepted"] == 0
    assert len(body["rejected"]) == 1
    assert "timestamp" in body["rejected"][0]["reason"]


def test_rescore_fires_regardless_of_batch_size(client, reset):
    """
    Regression: the trigger tested `ingested % rescore_every == 0`. With a
    batch size that never landed on a multiple, rescoring never happened.
    """

    before = client.get("/api/stream/state").json()["rescores"]

    every = client.get("/api/stream/state").json()["rescore_every"]

    # A batch size chosen so the running total steps over, never onto,
    # multiples of `rescore_every`.
    batch = every + 1

    client.post(
        "/api/stream/events",
        json={
            "events": [
                event(event_id=f"batch-{index}")
                for index in range(batch)
            ]
        },
    )

    after = client.get("/api/stream/state").json()["rescores"]

    assert after > before, "rescore never fired"


# ============================================================
# INDUCTIVE SCORING
# ============================================================

def test_unseen_accounts_are_scored(client, reset):
    """
    The model has no node-ID embeddings, so an account it has never seen can
    still be scored once it has features and edges.
    """

    before = client.get("/api/stream/state").json()["accounts_known"]

    events = [
        event(
            event_id=f"new-{index}",
            timestamp=f"2026-01-01T22:2{index}:00",
            source_account="ACC_BRAND_NEW_1",
            target_account="ACC_BRAND_NEW_2",
            source_vpa="VPA_NEW_1",
            target_vpa="VPA_NEW_2",
        )
        for index in range(4)
    ]

    client.post(
        "/api/stream/events",
        json={"events": events, "rescore": True},
    )

    state = client.get("/api/stream/state").json()

    assert state["accounts_known"] == before + 2

    scores = streaming_detector.scores

    for account in ("ACC_BRAND_NEW_1", "ACC_BRAND_NEW_2"):
        assert account in scores, f"{account} was not scored"
        assert 0.0 <= scores[account] <= 1.0


# ============================================================
# SCORE STABILITY
# ============================================================

def test_existing_scores_are_stable_without_a_clock_advance(client, reset):
    """
    Regression: time was scaled by the min/max span of the buffer, so any
    new event stretched the span and moved every historical account's score.
    With a fixed training scale, ingesting an event at the current head of
    the stream must leave existing accounts essentially untouched.
    """

    client.post("/api/stream/rescore")

    before = dict(streaming_detector.scores)

    assert before, "no baseline scores"

    # Timestamp at the existing head of the stream: no clock advance, so no
    # legitimate recency decay either.
    import pandas as pd

    head = max(
        pd.to_datetime(
            [row["timestamp"] for row in streaming_detector.events],
            format="ISO8601",
        )
    )

    # Every entity on this event is new, so it shares no device, VPA or
    # counterparty with anything already in the graph. Any movement in
    # existing scores could then only come from time being rescaled - which
    # is exactly the regression under test. (Reusing an existing device here
    # would legitimately shift the accounts that share it.)
    client.post(
        "/api/stream/events",
        json={
            "events": [
                event(
                    event_id="stability",
                    timestamp=head.isoformat(),
                    source_account="ACC_STABLE_1",
                    target_account="ACC_STABLE_2",
                    device_id="DEV_STABLE_NEW",
                    source_vpa="VPA_STABLE_1",
                    target_vpa="VPA_STABLE_2",
                )
            ],
            "rescore": True,
        },
    )

    after = streaming_detector.scores

    shared = [key for key in before if key in after]

    assert shared

    drift = max(abs(before[key] - after[key]) for key in shared)

    assert drift < 1e-3, f"existing scores drifted by {drift}"


# ============================================================
# RINGS AND RESET
# ============================================================

def test_live_rings_have_a_sane_shape(client):

    body = client.get("/api/stream/rings").json()

    assert body["count"] == len(body["rings"])

    for ring in body["rings"]:
        assert ring["ring_size"] >= 3
        assert len(ring["accounts"]) == ring["ring_size"]
        assert 0.0 <= ring["ring_risk_score"] <= 1.0
        assert ring["time_span_minutes"] <= state_window(client)


def state_window(client) -> float:
    return client.get("/api/stream/state").json()["window_minutes"]


def test_reset_returns_to_the_seeded_baseline(client):

    baseline = client.get("/api/stream/state").json()

    client.post(
        "/api/stream/events",
        json={
            "events": [event(event_id="to-be-discarded")],
            "rescore": False,
        },
    )

    after_ingest = client.get("/api/stream/state").json()

    assert after_ingest["events_total"] == baseline["events_total"] + 1

    restored = client.post("/api/stream/reset").json()

    assert restored["events_total"] == baseline["events_total"]
    assert restored["events_ingested"] == 0
    assert restored["accounts_known"] == baseline["accounts_known"]


def test_forced_rescore_reports_timing(client):

    body = client.post("/api/stream/rescore").json()

    assert body["accounts"] > 0
    assert body["milliseconds"] >= 0
