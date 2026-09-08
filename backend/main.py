from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

import math
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.dashboard_api import router as dashboard_router
from backend.model_service import model_service
from backend.stream_api import router as stream_router
from backend.stream_service import streaming_detector


# ============================================================
# DATA FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RINGS_FILE = BASE_DIR / "data" / "processed" / "detected_rings.csv"
GNN_RINGS_FILE = BASE_DIR / "data" / "processed" / "gnn_detected_rings.csv"
EXPLANATIONS_FILE = BASE_DIR / "data" / "processed" / "ring_explanations.csv"
EVENTS_FILE = BASE_DIR / "data" / "synthetic" / "upi_events.csv"
SCORES_FILE = BASE_DIR / "data" / "processed" / "account_risk_scores.csv"

FRONTEND_DIR = BASE_DIR / "frontend"


# ============================================================
# UPI FRAUD RING DETECTION API
# ============================================================

@asynccontextmanager
async def lifespan(_: FastAPI):

    print("=" * 60)
    print("UPI FRAUD RING DETECTION API")
    print("=" * 60)

    print(f"Rings file:         {RINGS_FILE}")
    print(f"GNN rings file:     {GNN_RINGS_FILE}")
    print(f"Explanations file:  {EXPLANATIONS_FILE}")
    print(f"Events file:        {EVENTS_FILE}")

    # Load the trained model once. Failure here is non-fatal: the /score
    # endpoint falls back to the rule layer and reports it honestly.
    model_service.load()

    if model_service.available:
        print(
            f"Model:              temporal_hetero_gnn "
            f"({model_service.account_count} accounts scored, "
            f"threshold {model_service.threshold})"
        )
    else:
        print(f"Model:              UNAVAILABLE ({model_service.error})")
        print("                    /score will use the rule layer only.")

    # Warm the live detector so the console's Live view works immediately.
    streaming_detector.load()

    if streaming_detector.available:
        print(
            f"Streaming:          ready "
            f"({streaming_detector.seeded_count} events seeded, "
            f"{len(streaming_detector.rings)} rings)"
        )
    else:
        print(f"Streaming:          UNAVAILABLE ({streaming_detector.error})")

    print("=" * 60)
    print("API startup complete")
    print("=" * 60)

    yield


app = FastAPI(
    title="UPI Fraud Ring Detection API",
    description="Real-time API for UPI fraud and fraud-ring detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(dashboard_router)
app.include_router(stream_router)


# ============================================================
# WEB CONSOLE
#
# Static single-page console served from frontend/. Mounted only when
# the directory exists so the API still starts without it.
# ============================================================

if FRONTEND_DIR.is_dir():

    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return FileResponse(FRONTEND_DIR / "index.html")


# ============================================================
# REQUEST SCHEMA
# ============================================================

class Transaction(BaseModel):
    source_account: str
    target_account: str
    amount: float
    event_type: str = "TRANSFER"
    timestamp: Optional[str] = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load CSV safely.

    Converts NaN, +Infinity and -Infinity into None-compatible
    values so FastAPI can serialize the response as valid JSON.
    """

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        # Replace infinite values
        df = df.replace(
            [float("inf"), float("-inf")],
            pd.NA
        )

        # Convert NaN / NA to None
        df = df.astype(object).where(
            pd.notna(df),
            None
        )

        return df

    except Exception as exc:
        print(f"Error reading {path}: {exc}")
        return pd.DataFrame()


def clean_value(value: Any):
    """
    Recursively convert values into JSON-safe Python values.
    """

    if value is None:
        return None

    # pandas missing values
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # Floating-point NaN / Infinity
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # pandas / numpy numbers
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            str(k): clean_value(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            clean_value(v)
            for v in value
        ]

    return value


def records_from_dataframe(df: pd.DataFrame):
    """
    Convert dataframe into JSON-safe records.
    """

    if df.empty:
        return []

    records = df.to_dict(orient="records")

    return [
        clean_value(record)
        for record in records
    ]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "Real-Time Detection of UPI Fraud Rings using Graph Neural Networks",
        "status": "running",
        "api_version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "rings": "/rings",
            "ring": "/rings/{ring_id}",
            "gnn_rings": "/rings/gnn",
            "explanations": "/explanations",
            "ring_explanation": "/explanations/{ring_id}",
            "events_summary": "/events/summary",
            "score": "/score",
            "account_risk": "/accounts/{account_id}/risk",
            "model_info": "/model/info",
            "stream": "/api/stream/state",
            "dashboard": "/dashboard",
            "docs": "/docs",
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# GET ALL DETECTED RINGS
# ============================================================

@app.get("/rings")
def get_rings():

    rings = load_csv(RINGS_FILE)

    if rings.empty:
        return {
            "detected_rings": 0,
            "rings": [],
        }

    result = records_from_dataframe(rings)

    return {
        "detected_rings": len(result),
        "rings": result,
    }


# ============================================================
# GET GNN-DETECTED RINGS
#
# Declared before /rings/{ring_id} so the literal path wins; FastAPI
# matches routes in definition order and the path parameter would
# otherwise capture "gnn".
# ============================================================

@app.get("/rings/gnn")
def get_gnn_rings():

    rings = load_csv(GNN_RINGS_FILE)

    if rings.empty:
        return {
            "detected_rings": 0,
            "rings": [],
            "detector": "temporal_hetero_gnn",
        }

    result = records_from_dataframe(rings)

    return {
        "detected_rings": len(result),
        "rings": result,
        "detector": "temporal_hetero_gnn",
    }


# ============================================================
# GET ONE RING
# ============================================================

@app.get("/rings/{ring_id}")
def get_ring(ring_id: str):

    rings = load_csv(RINGS_FILE)

    if rings.empty:
        raise HTTPException(
            status_code=404,
            detail="No detected rings found",
        )

    # Find ring ID column
    possible_columns = [
        "detected_ring_id",
        "ring_id",
        "id",
    ]

    ring_column = None

    for column in possible_columns:
        if column in rings.columns:
            ring_column = column
            break

    if ring_column is None:
        raise HTTPException(
            status_code=500,
            detail="Ring ID column not found in detected_rings.csv",
        )

    result = rings[
        rings[ring_column].astype(str) == str(ring_id)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Ring not found",
        )

    records = records_from_dataframe(result)

    return records[0]


# ============================================================
# GET ALL EXPLANATIONS
# ============================================================

@app.get("/explanations")
def get_explanations():

    explanations = load_csv(EXPLANATIONS_FILE)

    if explanations.empty:
        return {
            "explanation_events": 0,
            "explanations": [],
        }

    result = records_from_dataframe(explanations)

    return {
        "explanation_events": len(result),
        "explanations": result,
    }


# ============================================================
# GET EXPLANATION FOR ONE RING
# ============================================================

@app.get("/explanations/{ring_id}")
def get_ring_explanation(ring_id: str):

    explanations = load_csv(EXPLANATIONS_FILE)

    if explanations.empty:
        raise HTTPException(
            status_code=404,
            detail="No explanations available",
        )

    if "detected_ring_id" in explanations.columns:

        result = explanations[
            explanations["detected_ring_id"].astype(str)
            == str(ring_id)
        ]

    elif "ring_id" in explanations.columns:

        result = explanations[
            explanations["ring_id"].astype(str)
            == str(ring_id)
        ]

    else:
        raise HTTPException(
            status_code=500,
            detail="Ring ID column not found in ring_explanations.csv",
        )

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Explanation not found",
        )

    # Sort by timestamp if available
    if "timestamp" in result.columns:
        try:
            result = result.sort_values(
                by="timestamp"
            )
        except Exception:
            pass

    return {
        "ring_id": str(ring_id),
        "events": records_from_dataframe(result),
    }


# ============================================================
# EVENTS SUMMARY
# ============================================================

@app.get("/events/summary")
def event_summary():

    events = load_csv(EVENTS_FILE)

    if events.empty:
        return {
            "total_events": 0,
            "event_types": {},
        }

    summary = {
        "total_events": int(len(events)),
        "event_types": {},
    }

    if "event_type" in events.columns:

        counts = events["event_type"].value_counts(
            dropna=False
        )

        summary["event_types"] = {
            str(key): int(value)
            for key, value in counts.items()
        }

    return clean_value(summary)


# ============================================================
# SCORE TRANSACTION
# ============================================================

def rule_layer(amount: float, event_type: str):
    """
    Deterministic amount / event-type signal.

    Used on its own when the model is unavailable or has never seen the
    accounts involved, and blended with the model score otherwise.
    """

    score = 0.0
    reasons = []

    if amount >= 100000:
        score += 0.35
        reasons.append("high_transaction_amount")

    elif amount >= 50000:
        score += 0.20
        reasons.append("elevated_transaction_amount")

    elif amount >= 10000:
        score += 0.10

    if event_type in {
        "TRANSFER",
        "BENEFICIARY_ADD",
        "MERCHANT_COLLECTION",
    }:
        score += 0.05

    return min(max(score, 0.0), 1.0), reasons


@app.post("/score")
def score_transaction(transaction: Transaction):

    amount = float(transaction.amount)
    event_type = transaction.event_type.upper()

    rule_score, reasons = rule_layer(amount, event_type)

    source_score = model_service.account_score(
        transaction.source_account
    )

    target_score = model_service.account_score(
        transaction.target_account
    )

    known_scores = [
        value
        for value in (source_score, target_score)
        if value is not None
    ]

    if known_scores:

        # Both endpoints risky is the fraud-ring signal; a single risky
        # endpoint still matters, so take the stronger of the pair score
        # and the maximum single-account score.
        if len(known_scores) == 2:
            pair_score = (known_scores[0] * known_scores[1]) ** 0.5
        else:
            pair_score = 0.0

        model_score = max(pair_score, max(known_scores))

        # The model carries the decision; the rule layer can only nudge.
        score = min(1.0, model_score + 0.15 * rule_score)

        if source_score is not None and source_score >= model_service.threshold:
            reasons.append("source_account_flagged_by_gnn")

        if target_score is not None and target_score >= model_service.threshold:
            reasons.append("target_account_flagged_by_gnn")

        if len(known_scores) == 2 and pair_score >= model_service.threshold:
            reasons.append("both_endpoints_in_suspected_ring")

        model_status = "temporal_hetero_gnn"

    else:
        score = rule_score

        model_status = (
            "rule_layer_unknown_accounts"
            if model_service.available
            else "rule_layer_model_unavailable"
        )

    # Band the rounded value, not the raw float. 0.35 + 0.05 is
    # 0.39999999999999997 in binary floating point, which would report
    # LOW next to a displayed risk_score of 0.4000.
    score = round(score, 4)

    if score >= 0.70:
        risk_level = "HIGH"

    elif score >= 0.40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return clean_value(
        {
            "source_account": transaction.source_account,
            "target_account": transaction.target_account,
            "amount": amount,
            "event_type": transaction.event_type,
            "risk_score": round(score, 4),
            "risk_level": risk_level,
            "fraud_probability": round(score, 4),
            "source_account_score": (
                round(source_score, 4)
                if source_score is not None
                else None
            ),
            "target_account_score": (
                round(target_score, 4)
                if target_score is not None
                else None
            ),
            "rule_layer_score": round(rule_score, 4),
            "reasons": reasons,
            "timestamp": transaction.timestamp
            or datetime.now(timezone.utc).isoformat(),
            "model_status": model_status,
        }
    )


# ============================================================
# ACCOUNT RISK
# ============================================================

@app.get("/accounts/{account_id}/risk")
def get_account_risk(account_id: str):

    score = model_service.account_score(account_id)

    if score is None:

        if not model_service.available:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model unavailable: "
                    f"{model_service.error}"
                ),
            )

        raise HTTPException(
            status_code=404,
            detail=f"Account not found in the scored graph: {account_id}",
        )

    if score >= 0.60:
        risk_level = "HIGH"

    elif score >= 0.30:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return {
        "account_id": account_id,
        "fraud_probability": round(score, 4),
        "risk_level": risk_level,
        "decision_threshold": model_service.threshold,
        "flagged": bool(score >= model_service.threshold),
        "model": "temporal_hetero_gnn",
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model/info")
def get_model_info():

    return model_service.info()


# ============================================================
# STARTUP
#
# Handled by the `lifespan` context manager defined at the top of this
# module. The deprecated @app.on_event("startup") hook was removed.
# ============================================================