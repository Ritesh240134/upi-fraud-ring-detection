"""
JSON endpoints backing the web console at /dashboard.

These live under /api/ and are additive - the endpoints that existed before
the console was built keep their paths and their response shapes.

Every CSV read is cached against the file's modification time, so re-running
the detector or a training script is picked up on the next request without a
server restart, and a page refresh does not re-parse the event stream.
"""

from pathlib import Path
from typing import Optional

import json
import math

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.model_service import model_service


ROOT = Path(__file__).resolve().parent.parent

PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "models" / "artifacts"

EVENTS_FILE = ROOT / "data" / "synthetic" / "upi_events.csv"
RINGS_FILE = PROCESSED / "gnn_detected_rings.csv"
RULE_RINGS_FILE = PROCESSED / "detected_rings.csv"
SCORES_FILE = PROCESSED / "account_risk_scores.csv"
LABELS_FILE = PROCESSED / "account_labels.csv"

METRIC_FILES = [
    (
        "GraphSAGE baseline",
        "homogeneous account graph, no time",
        ARTIFACTS / "graphsage_eval_metrics.json",
    ),
    (
        "Temporal hetero (time withheld)",
        "same architecture, timestamps withheld",
        ARTIFACTS / "temporal_hetero_eval_metrics_no_time.json",
    ),
    (
        "Temporal hetero GNN",
        "4 node types, 9 relations, time-gated messages",
        ARTIFACTS / "temporal_hetero_eval_metrics.json",
    ),
]


router = APIRouter(prefix="/api", tags=["dashboard"])


# ============================================================
# CACHED READERS
# ============================================================

_cache: dict = {}


def read_csv_cached(path: Path) -> pd.DataFrame:
    """
    Read a CSV, reusing the parsed frame until the file changes on disk.
    """

    if not path.exists():
        return pd.DataFrame()

    stamp = path.stat().st_mtime_ns

    cached = _cache.get(path)

    if cached is not None and cached[0] == stamp:
        return cached[1]

    frame = pd.read_csv(path)

    _cache[path] = (stamp, frame)

    return frame


def read_json(path: Path) -> Optional[dict]:

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def safe(value):
    """
    Convert numpy / pandas scalars into JSON-safe Python values.
    """

    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def records(frame: pd.DataFrame) -> list:
    return [
        {key: safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


# ============================================================
# OVERVIEW
# ============================================================

@router.get("/overview")
def overview():
    """
    Headline counters for the top of the console.
    """

    events = read_csv_cached(EVENTS_FILE)
    labels = read_csv_cached(LABELS_FILE)
    rings = read_csv_cached(RINGS_FILE)
    scores = read_csv_cached(SCORES_FILE)

    temporal = read_json(
        ARTIFACTS / "temporal_hetero_eval_metrics.json"
    )

    accounts_in_rings = (
        int(rings["ring_size"].sum()) if not rings.empty else 0
    )

    true_in_rings = (
        int(rings["true_fraud_accounts"].sum())
        if not rings.empty and "true_fraud_accounts" in rings.columns
        else 0
    )

    fraud_accounts = (
        int(labels["label"].sum())
        if not labels.empty and "label" in labels.columns
        else 0
    )

    return {
        "accounts": int(len(labels)) if not labels.empty else 0,
        "fraud_accounts": fraud_accounts,
        "events": int(len(events)) if not events.empty else 0,
        "fraud_events": (
            int(events["is_fraud"].sum())
            if not events.empty and "is_fraud" in events.columns
            else 0
        ),
        "rings_detected": int(len(rings)),
        "accounts_in_rings": accounts_in_rings,
        "true_fraud_in_rings": true_in_rings,
        "ring_precision": (
            round(true_in_rings / accounts_in_rings, 4)
            if accounts_in_rings
            else None
        ),
        "ring_recall": (
            round(true_in_rings / fraud_accounts, 4)
            if fraud_accounts
            else None
        ),
        "high_risk_accounts": (
            int((scores["risk_level"] == "HIGH").sum())
            if not scores.empty
            else 0
        ),
        "test_roc_auc": (
            safe(temporal["test"]["roc_auc"]) if temporal else None
        ),
        "test_f1": safe(temporal["test"]["f1"]) if temporal else None,
        "event_types": (
            {
                str(key): int(value)
                for key, value in events["event_type"]
                .value_counts()
                .items()
            }
            if not events.empty and "event_type" in events.columns
            else {}
        ),
        "typologies": (
            {
                str(key): int(value)
                for key, value in events[
                    events["typology"].astype(str) != "NORMAL"
                ]["typology"]
                .value_counts()
                .items()
            }
            if not events.empty and "typology" in events.columns
            else {}
        ),
        "model": model_service.info(),
    }


# ============================================================
# MODEL METRICS
# ============================================================

@router.get("/metrics")
def metrics():
    """
    The measured evaluation metrics, read from the artifact JSON files.
    Nothing here is computed on the fly or hard-coded.
    """

    models = []

    for name, subtitle, path in METRIC_FILES:

        payload = read_json(path)

        if payload is None:
            continue

        test = payload["test"]
        validation = payload.get("validation", {})

        models.append(
            {
                "name": name,
                "subtitle": subtitle,
                "uses_edge_time": payload.get("uses_edge_time"),
                "decision_threshold": payload.get("decision_threshold"),
                "test": {
                    key: safe(test.get(key))
                    for key in (
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "average_precision",
                    )
                },
                "validation": {
                    key: safe(validation.get(key))
                    for key in ("f1", "roc_auc")
                },
                "confusion_matrix": test.get("confusion_matrix"),
            }
        )

    temporal = read_json(
        ARTIFACTS / "temporal_hetero_eval_metrics.json"
    )

    return {
        "models": models,
        "test_accounts": (
            temporal.get("test_accounts") if temporal else None
        ),
        "test_fraud_accounts": (
            temporal.get("test_fraud_accounts") if temporal else None
        ),
        "caveat": (
            "The test split holds a small number of positives, so a single "
            "account moving between columns shifts F1 noticeably. Treat the "
            "gap between the temporal model and its ablation as suggestive, "
            "not conclusive."
        ),
    }


# ============================================================
# RINGS
# ============================================================

@router.get("/rings")
def list_rings():

    rings = read_csv_cached(RINGS_FILE)

    if rings.empty:
        return {"count": 0, "rings": []}

    frame = rings.copy()

    frame["account_list"] = (
        frame["accounts"].astype(str).str.split(",")
    )

    return {
        "count": int(len(frame)),
        "rings": records(frame),
    }


@router.get("/rings/{ring_id}")
def ring_detail(ring_id: str):
    """
    One ring, plus its members and the transfers between them - enough to
    draw the ring as a graph and as a timeline.
    """

    rings = read_csv_cached(RINGS_FILE)

    if rings.empty:
        raise HTTPException(status_code=404, detail="No rings available")

    match = rings[
        rings["detected_ring_id"].astype(str) == str(ring_id)
    ]

    if match.empty:
        raise HTTPException(status_code=404, detail="Ring not found")

    ring = match.iloc[0]

    members = [
        account
        for account in str(ring["accounts"]).split(",")
        if account
    ]

    scores = read_csv_cached(SCORES_FILE)

    score_map = {}
    truth_map = {}

    if not scores.empty:
        score_map = dict(
            zip(
                scores["account"].astype(str),
                scores["fraud_probability"],
            )
        )
        truth_map = dict(
            zip(
                scores["account"].astype(str),
                scores["true_label"],
            )
        )

    nodes = [
        {
            "account": account,
            "score": safe(score_map.get(account, 0.0)),
            "true_label": int(truth_map.get(account, 0)),
        }
        for account in members
    ]

    # Transfers with both endpoints inside the ring, within its window.
    events = read_csv_cached(EVENTS_FILE)

    edges = []
    timeline = []

    if not events.empty:

        frame = events.copy()

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            errors="coerce",
        )

        window_start = pd.to_datetime(ring["window_start"])
        window_end = pd.to_datetime(ring["window_end"])

        inside = frame[
            frame["source_account"].astype(str).isin(members)
            & frame["target_account"].astype(str).isin(members)
            & (frame["timestamp"] >= window_start)
            & (frame["timestamp"] <= window_end)
        ].sort_values("timestamp")

        for row in inside.itertuples(index=False):
            edges.append(
                {
                    "source": str(row.source_account),
                    "target": str(row.target_account),
                    "amount": safe(row.amount),
                    "timestamp": str(row.timestamp),
                    "event_type": str(row.event_type),
                    "is_fraud": int(row.is_fraud),
                    "typology": safe(row.typology),
                }
            )

        # A wider trace: any event touching a member in the window.
        touching = frame[
            (
                frame["source_account"].astype(str).isin(members)
                | frame["target_account"].astype(str).isin(members)
            )
            & (frame["timestamp"] >= window_start)
            & (frame["timestamp"] <= window_end)
        ].sort_values("timestamp")

        timeline = [
            {
                "timestamp": str(row.timestamp),
                "event_type": str(row.event_type),
                "source": str(row.source_account),
                "target": safe(row.target_account),
                "amount": safe(row.amount),
                "is_fraud": int(row.is_fraud),
                "typology": safe(row.typology),
            }
            for row in touching.head(150).itertuples(index=False)
        ]

    detail = {key: safe(value) for key, value in ring.items()}

    detail["account_list"] = members

    return {
        "ring": detail,
        "nodes": nodes,
        "edges": edges,
        "timeline": timeline,
    }


# ============================================================
# ACCOUNTS
# ============================================================

@router.get("/accounts")
def list_accounts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str = Query("", max_length=64),
    level: str = Query("", max_length=16),
    sort: str = Query("score"),
):

    scores = read_csv_cached(SCORES_FILE)

    if scores.empty:
        return {"total": 0, "accounts": [], "returned": 0}

    frame = scores.copy()

    if search:
        frame = frame[
            frame["account"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    if level and level.upper() != "ALL":
        frame = frame[
            frame["risk_level"].astype(str).str.upper() == level.upper()
        ]

    if sort == "account":
        frame = frame.sort_values("account")
    else:
        frame = frame.sort_values("fraud_probability", ascending=False)

    total = int(len(frame))

    page = frame.iloc[offset : offset + limit]

    return {
        "total": total,
        "returned": int(len(page)),
        "offset": offset,
        "accounts": records(page),
    }


@router.get("/accounts-summary")
def accounts_summary():
    """
    Risk-level counts plus a histogram of the score distribution.
    """

    scores = read_csv_cached(SCORES_FILE)

    if scores.empty:
        return {"levels": {}, "histogram": [], "total": 0}

    counts = scores["risk_level"].value_counts()

    bins = [0.0] * 20

    for value in scores["fraud_probability"]:

        if pd.isna(value):
            continue

        index = min(int(float(value) * 20), 19)

        bins[index] += 1

    return {
        "total": int(len(scores)),
        "levels": {
            str(key): int(value) for key, value in counts.items()
        },
        "histogram": [
            {
                "from": round(index / 20, 2),
                "to": round((index + 1) / 20, 2),
                "count": int(count),
            }
            for index, count in enumerate(bins)
        ],
        "flagged": int(
            (
                scores["fraud_probability"]
                >= (model_service.threshold or 0.5)
            ).sum()
        ),
        "threshold": model_service.threshold,
    }
