"""
HTTP surface for the live streaming detector.

Ingest is deliberately batched: `rescore` defaults to null, which lets the
detector decide (once every `rescore_every` events). Pass true to force a
rescore for a single event, which is what an interactive demo wants and what
a high-throughput producer should not do.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.stream_service import streaming_detector


router = APIRouter(prefix="/api/stream", tags=["streaming"])


class StreamEvent(BaseModel):

    event_id: Optional[str] = None
    timestamp: Optional[str] = None
    event_type: str = "TRANSFER"

    source_account: str
    target_account: Optional[str] = None

    device_id: Optional[str] = None
    source_vpa: Optional[str] = None
    target_vpa: Optional[str] = None
    merchant_id: Optional[str] = None

    amount: float = 0.0

    # Ground truth, present only when replaying labelled synthetic data.
    # It is never used for scoring - only to report detector quality.
    is_fraud: Optional[int] = 0
    typology: Optional[str] = None
    ring_id: Optional[str] = None


class IngestRequest(BaseModel):

    events: List[StreamEvent] = Field(..., min_length=1, max_length=2000)
    rescore: Optional[bool] = None


def require_available() -> None:

    if not streaming_detector.available:
        raise HTTPException(
            status_code=503,
            detail=(
                "Streaming detector unavailable: "
                f"{streaming_detector.error}"
            ),
        )


@router.post("/events")
def ingest_events(request: IngestRequest):

    require_available()

    return streaming_detector.ingest(
        [event.model_dump() for event in request.events],
        rescore=request.rescore,
    )


@router.post("/event")
def ingest_event(event: StreamEvent, rescore: Optional[bool] = None):

    require_available()

    return streaming_detector.ingest(
        [event.model_dump()],
        rescore=rescore,
    )


@router.get("/state")
def stream_state():
    return streaming_detector.snapshot()


@router.get("/rings")
def stream_rings():

    require_available()

    rings = streaming_detector.live_rings()

    return {"count": len(rings), "rings": rings}


@router.post("/rescore")
def force_rescore():

    require_available()

    return streaming_detector.rescore()


@router.post("/reset")
def reset_stream(seed: bool = True):
    """
    Drop live state and return to the seeded historical baseline.
    """

    require_available()

    return streaming_detector.reset(seed=seed)
