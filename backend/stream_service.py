"""
Live streaming fraud-ring detection.

The batch pipeline scores a frozen graph. This module keeps the graph alive:
events arrive one at a time, the graph is rebuilt incrementally, accounts are
rescored with the trained model, and rings are extracted from a sliding time
window over the most recent suspicious transfers.

Why this works without retraining
---------------------------------
`TemporalHeteroGNN` has no node-ID embedding tables - every node type is a
`nn.Linear` over its feature vector, followed by neighbour aggregation. The
architecture is therefore inductive: an account the model has never seen can
be scored as soon as its behavioural features and its edges exist. The only
requirement is that live features are standardised with the *training*
statistics, which is why `data/processed/feature_scaler.json` is loaded here
rather than refitting on live traffic.

Scope
-----
Rescoring runs a full forward pass over the whole graph. At this dataset's
size (hundreds of accounts, thousands of events) that is a few hundred
milliseconds and is genuinely real-time. It does not scale to a national
payment network - see the README for what production would need instead.
"""

from collections import deque
from pathlib import Path
from typing import Optional

import json
import threading
import time as clock

import pandas as pd
import torch

from graph.prepare_hetero_graph import build_hetero_graph
from models.temporal_hetero_gnn import build_model, edge_time_dict_from


ROOT = Path(__file__).resolve().parent.parent

EVENTS_FILE = ROOT / "data" / "synthetic" / "upi_events.csv"
SCALER_FILE = ROOT / "data" / "processed" / "feature_scaler.json"
MODEL_FILE = ROOT / "models" / "artifacts" / "temporal_hetero_gnn.pt"

EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "event_type",
    "source_account",
    "target_account",
    "device_id",
    "source_vpa",
    "target_vpa",
    "merchant_id",
    "amount",
    "is_fraud",
    "typology",
    "ring_id",
]

# Ring extraction, matching detection/gnn_ring_detector.py.
WINDOW_MINUTES = 10
PAIR_THRESHOLD = 0.50
MIN_RING_SIZE = 3

MAX_ALERTS = 250
MAX_EVENTS = 40000


class StreamingDetector:
    """
    Thread-safe live detector. All mutating operations take the lock, so the
    API can ingest and read state concurrently.
    """

    def __init__(self) -> None:

        self._lock = threading.RLock()
        self._loaded = False

        self.available = False
        self.error: Optional[str] = None

        self.model = None
        self.scalers = {}
        self.threshold = 0.50

        self.events: list = []
        self.seeded_count = 0

        self.scores: dict = {}
        self.rings: list = []
        self.alerts: deque = deque(maxlen=MAX_ALERTS)

        self.ingested = 0
        self.rejected = 0
        self.dropped_events = 0
        self.since_rescore = 0
        self.rescores = 0
        self.last_rescore_ms = None
        self.last_rescore_at = None
        self.started_at = clock.time()
        self.recent_ingest_times: deque = deque(maxlen=200)

        self.rescore_every = 25

    # ========================================================
    # LOADING
    # ========================================================

    def load(self, seed: bool = True) -> None:
        """
        Load the model and scaler, and optionally warm the graph with the
        historical event file. Failures are recorded, never raised - the rest
        of the API stays up.
        """

        with self._lock:

            if self._loaded:
                return

            self._loaded = True

            try:
                self._load_unsafe(seed)
                self.available = True

            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                self.available = False
                self.error = f"{type(exc).__name__}: {exc}"

    def _load_unsafe(self, seed: bool) -> None:

        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

        if not SCALER_FILE.exists():
            raise FileNotFoundError(
                f"Feature scaler not found: {SCALER_FILE}. "
                "Run: python -m graph.prepare_hetero_graph"
            )

        self.scalers = json.loads(
            SCALER_FILE.read_text(encoding="utf-8")
        )

        checkpoint = torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=False,
        )

        self.threshold = float(
            checkpoint.get("decision_threshold", 0.50)
        )

        self.checkpoint = checkpoint

        if seed and EVENTS_FILE.exists():

            self.events = self._read_seed()
            self.seeded_count = len(self.events)

            self._rescore_unsafe()

    # ========================================================
    # GRAPH + MODEL
    # ========================================================

    def _read_seed(self) -> list:
        """
        Load the historical event file, normalising timestamps to the same
        ISO format that ingest produces.
        """

        frame = pd.read_csv(EVENTS_FILE)

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            errors="coerce",
        ).map(
            lambda value: None if pd.isna(value) else value.isoformat()
        )

        return frame.to_dict(orient="records")

    def _frame(self) -> pd.DataFrame:
        """
        Materialise the live event log.

        Timestamps are normalised to a single ISO format on ingest, so this
        can use the fast single-format parse. Anything that still fails to
        parse is counted rather than silently dropped.
        """

        frame = pd.DataFrame(self.events, columns=EVENT_COLUMNS)

        parsed = pd.to_datetime(
            frame["timestamp"],
            format="ISO8601",
            errors="coerce",
        )

        unparsed = int(parsed.isna().sum())

        if unparsed:
            self.dropped_events = unparsed

        frame["timestamp"] = parsed

        frame["amount"] = pd.to_numeric(
            frame["amount"],
            errors="coerce",
        ).fillna(0.0)

        return frame.dropna(subset=["timestamp"])

    def _rescore_unsafe(self) -> dict:
        """
        Rebuild the graph from everything seen so far and run the model.
        Returns timing information.
        """

        started = clock.perf_counter()

        frame = self._frame()

        if frame.empty:
            return {"accounts": 0, "milliseconds": 0.0}

        # Live traffic carries no labels, and the scaler comes from training.
        data, _, accounts = build_hetero_graph(
            frame,
            scalers=self.scalers,
            with_labels=False,
        )

        if self.model is None:
            self.model = build_model(
                data,
                hidden_dim=self.checkpoint["hidden_dim"],
                time_dim=self.checkpoint["time_dim"],
                dropout=self.checkpoint["dropout"],
            )
            self.model.load_state_dict(
                self.checkpoint["model_state_dict"]
            )
            self.model.eval()

        edge_time_dict = (
            edge_time_dict_from(data)
            if self.checkpoint.get("uses_edge_time", True)
            else {}
        )

        with torch.no_grad():
            probabilities = torch.sigmoid(
                self.model(
                    data.x_dict,
                    data.edge_index_dict,
                    edge_time_dict,
                )
            ).tolist()

        previous = self.scores

        self.scores = {
            str(account): float(probability)
            for account, probability in zip(accounts, probabilities)
        }

        self._raise_score_alerts(previous)

        self._detect_rings_unsafe(frame)

        elapsed = (clock.perf_counter() - started) * 1000.0

        self.rescores += 1
        self.since_rescore = 0
        self.last_rescore_ms = round(elapsed, 1)
        self.last_rescore_at = clock.time()

        return {
            "accounts": len(self.scores),
            "milliseconds": self.last_rescore_ms,
        }

    # ========================================================
    # ALERTS
    # ========================================================

    def _raise_score_alerts(self, previous: dict) -> None:
        """
        Alert when an account crosses the decision threshold it was below.
        """

        for account, score in self.scores.items():

            was = previous.get(account)

            if score < self.threshold:
                continue

            if was is not None and was >= self.threshold:
                continue

            self.alerts.appendleft(
                {
                    "kind": "account",
                    "at": clock.time(),
                    "account": account,
                    "score": round(score, 4),
                    "previous": None if was is None else round(was, 4),
                    "message": (
                        f"{account} crossed the risk threshold "
                        f"({score:.3f})"
                    ),
                }
            )

    # ========================================================
    # RING DETECTION
    # ========================================================

    def _detect_rings_unsafe(self, frame: pd.DataFrame) -> None:
        """
        Sliding-window ring extraction over recent suspicious transfers,
        using the same logic as the batch detector.
        """

        from detection.gnn_ring_detector import (
            drop_subsets,
            extract_candidates,
        )

        transfers = frame[
            frame["event_type"].astype(str).str.upper() == "TRANSFER"
        ].copy()

        if transfers.empty:
            self.rings = []
            return

        transfers = transfers.dropna(
            subset=["source_account", "target_account"]
        )

        source_scores = (
            transfers["source_account"].astype(str).map(self.scores)
        )

        target_scores = (
            transfers["target_account"].astype(str).map(self.scores)
        )

        transfers["pair_score"] = (
            source_scores.fillna(0.0) * target_scores.fillna(0.0)
        ) ** 0.5

        suspicious = transfers[
            transfers["pair_score"] >= PAIR_THRESHOLD
        ].sort_values("timestamp").reset_index(drop=True)

        if suspicious.empty:
            self.rings = []
            return

        score_frame = pd.DataFrame(
            {
                "account": list(self.scores.keys()),
                "fraud_probability": list(self.scores.values()),
            }
        )

        candidates = drop_subsets(
            extract_candidates(suspicious, score_frame)
        )

        candidates.sort(
            key=lambda item: (
                -item["ring_risk_score"],
                -item["ring_size"],
            )
        )

        known = {
            tuple(ring["accounts"]) for ring in self.rings
        }

        rings = []

        for number, candidate in enumerate(candidates, start=1):

            ring = dict(candidate)

            ring["detected_ring_id"] = f"LIVE_RING_{number}"
            ring["accounts"] = list(candidate["accounts"])
            ring["window_start"] = str(candidate["window_start"])
            ring["window_end"] = str(candidate["window_end"])
            ring["matched_ground_truth_rings"] = list(
                candidate["matched_ground_truth_rings"]
            )

            rings.append(ring)

            if tuple(candidate["accounts"]) not in known:
                self.alerts.appendleft(
                    {
                        "kind": "ring",
                        "at": clock.time(),
                        "ring_id": ring["detected_ring_id"],
                        "size": ring["ring_size"],
                        "risk": ring["ring_risk_score"],
                        "accounts": ring["accounts"],
                        "message": (
                            f"Ring of {ring['ring_size']} accounts over "
                            f"{ring['time_span_minutes']} min "
                            f"(risk {ring['ring_risk_score']:.3f})"
                        ),
                    }
                )

        self.rings = rings

    # ========================================================
    # PUBLIC API
    # ========================================================

    def ingest(self, events: list, rescore: Optional[bool] = None) -> dict:
        """
        Add events to the live stream.

        Rescoring is batched: by default it runs once every `rescore_every`
        events, because a forward pass per event would dominate the cost
        without changing the answer much.
        """

        with self._lock:

            if not self.available:
                return {
                    "accepted": 0,
                    "error": self.error or "Streaming detector unavailable",
                }

            rejected = []

            for event in events:

                row = {column: event.get(column) for column in EVENT_COLUMNS}

                # Normalise the timestamp to one ISO format at the door.
                # Without this, an event whose timestamp string is formatted
                # differently from the rest ("2026-01-02 03:00:00" against
                # "2026-01-02T03:00:00") is coerced to NaT by the single
                # inferred format and disappears with no error.
                stamp = row.get("timestamp")

                if stamp in (None, "", float("nan")):
                    row["timestamp"] = pd.Timestamp.utcnow().isoformat()

                else:
                    parsed = pd.to_datetime(stamp, errors="coerce")

                    if pd.isna(parsed):
                        rejected.append(
                            {
                                "event_id": row.get("event_id"),
                                "timestamp": str(stamp),
                                "reason": "unparseable timestamp",
                            }
                        )
                        self.rejected += 1
                        continue

                    row["timestamp"] = parsed.isoformat()

                self.events.append(row)
                self.ingested += 1
                self.since_rescore += 1
                self.recent_ingest_times.append(clock.time())

            if len(self.events) > MAX_EVENTS:
                self.events = self.events[-MAX_EVENTS:]

            # Count events since the last rescore rather than testing the
            # running total against a modulus. With a batch size that does
            # not divide the counter, `ingested % rescore_every == 0` can
            # never be true and rescoring would never fire at all.
            due = (
                rescore
                if rescore is not None
                else self.since_rescore >= self.rescore_every
            )

            timing = (
                self._rescore_unsafe()
                if due
                else {"accounts": len(self.scores), "milliseconds": None}
            )

            return {
                "accepted": len(events) - len(rejected),
                "rejected": rejected,
                "total_events": len(self.events),
                "rescored": bool(due),
                "rescore_ms": timing["milliseconds"],
                "rings": len(self.rings),
            }

    def rescore(self) -> dict:
        with self._lock:
            if not self.available:
                return {"error": self.error}
            return self._rescore_unsafe()

    def throughput(self) -> float:
        """
        Events per second over the recent ingest window.
        """

        times = list(self.recent_ingest_times)

        if len(times) < 2:
            return 0.0

        span = times[-1] - times[0]

        if span <= 0:
            return 0.0

        return round((len(times) - 1) / span, 2)

    def top_accounts(self, limit: int = 12) -> list:

        ranked = sorted(
            self.scores.items(),
            key=lambda item: -item[1],
        )

        return [
            {
                "account": account,
                "score": round(score, 4),
                "flagged": score >= self.threshold,
            }
            for account, score in ranked[:limit]
        ]

    def snapshot(self) -> dict:

        with self._lock:

            return {
                "available": self.available,
                "error": self.error,
                "events_total": len(self.events),
                "events_seeded": self.seeded_count,
                "events_ingested": self.ingested,
                "events_rejected": self.rejected,
                "accounts_known": len(self.scores),
                "flagged_accounts": sum(
                    1
                    for score in self.scores.values()
                    if score >= self.threshold
                ),
                "active_rings": len(self.rings),
                "rescores": self.rescores,
                "last_rescore_ms": self.last_rescore_ms,
                "throughput_per_second": self.throughput(),
                "threshold": self.threshold,
                "window_minutes": WINDOW_MINUTES,
                "rescore_every": self.rescore_every,
                "alerts": list(self.alerts)[:40],
                "top_accounts": self.top_accounts(),
            }

    def live_rings(self) -> list:
        with self._lock:
            return list(self.rings)

    def reset(self, seed: bool = True) -> dict:
        """
        Drop live state and return to the seeded baseline.
        """

        with self._lock:

            self.events = []
            self.scores = {}
            self.rings = []
            self.alerts.clear()
            self.ingested = 0
            self.rejected = 0
            self.dropped_events = 0
            self.since_rescore = 0
            self.rescores = 0
            self.last_rescore_ms = None
            self.recent_ingest_times.clear()

            if seed and EVENTS_FILE.exists():
                self.events = self._read_seed()
                self.seeded_count = len(self.events)
            else:
                self.seeded_count = 0

            if self.available:
                self._rescore_unsafe()

            return self.snapshot()


streaming_detector = StreamingDetector()
