"""
Replay the synthetic event stream into the live detector.

The dataset spans about 22 hours of simulated activity. Replaying it at a
speed factor compresses that into minutes, so the console's Live view shows
accounts crossing the risk threshold and rings forming as the events arrive.

Usage
-----
    # replay the fraud-heavy tail at 600x (default)
    python -m streaming.replay

    # the whole stream, 3000x, resetting first
    python -m streaming.replay --from-start --speed 3000 --reset

    # a fixed number of events as fast as the API will take them
    python -m streaming.replay --limit 400 --speed 0

Options
-------
    --speed     wall-clock compression. 600 means 600 simulated seconds per
                real second. 0 replays with no delay at all.
    --reset     clear live state and reseed before replaying.
    --tail N    replay only the last N events (default 600).
    --batch N   events per request.
"""

from pathlib import Path

import argparse
import sys
import time

import pandas as pd
import urllib.error
import urllib.request

import json


ROOT = Path(__file__).resolve().parents[1]

EVENTS_FILE = ROOT / "data" / "synthetic" / "upi_events.csv"

DEFAULT_API = "http://127.0.0.1:8000"


def post(url: str, payload: dict, timeout: float = 60.0) -> dict:

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get(url: str, timeout: float = 30.0) -> dict:

    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Replay UPI events into the live detector."
    )

    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--speed", type=float, default=600.0)
    parser.add_argument("--tail", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="Replay the whole file rather than only its tail.",
    )

    return parser.parse_args()


def load_events(args: argparse.Namespace):
    """
    Split the event file into the history to warm the detector with and the
    slice to replay.

    The two must not overlap. Seeding with the whole file and then replaying
    part of it would feed those events twice, inflating account degrees and
    manufacturing rings that are an artefact of the replay rather than a
    detection.
    """

    frame = pd.read_csv(EVENTS_FILE)

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
    )

    frame = (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if args.from_start:
        return frame.iloc[:0], frame

    count = args.limit or args.tail

    split = max(0, len(frame) - count)

    return frame.iloc[:split], frame.iloc[split:].reset_index(drop=True)


def to_event(row) -> dict:

    def text(value):
        return None if pd.isna(value) else str(value)

    return {
        "event_id": str(row.event_id),
        "timestamp": row.timestamp.isoformat(),
        "event_type": str(row.event_type),
        "source_account": str(row.source_account),
        "target_account": text(row.target_account),
        "device_id": text(row.device_id),
        "source_vpa": text(row.source_vpa),
        "target_vpa": text(row.target_vpa),
        "merchant_id": text(row.merchant_id),
        "amount": 0.0 if pd.isna(row.amount) else float(row.amount),
        "is_fraud": 0 if pd.isna(row.is_fraud) else int(row.is_fraud),
        "typology": text(row.typology),
        "ring_id": text(row.ring_id),
    }


def main() -> int:

    args = parse_args()

    if not EVENTS_FILE.exists():
        print(f"Events file not found: {EVENTS_FILE}")
        return 1

    try:
        state = get(f"{args.api}/api/stream/state")
    except (urllib.error.URLError, OSError) as exc:
        print(f"Cannot reach the API at {args.api}: {exc}")
        print("Start it with: python -m uvicorn backend.main:app --port 8000")
        return 1

    if not state.get("available"):
        print(f"Streaming detector unavailable: {state.get('error')}")
        return 1

    history, events = load_events(args)

    if args.reset:
        # Reset *without* seeding, then load the history explicitly, so the
        # replayed events are the only ones the detector sees twice: never.
        print("Resetting live state...")
        post(f"{args.api}/api/stream/reset?seed=false", {})

        if len(history):
            print(f"Warming detector with {len(history):,} historical events...")

            rows = [to_event(row) for row in history.itertuples(index=False)]

            for start in range(0, len(rows), 500):
                post(
                    f"{args.api}/api/stream/events",
                    {
                        "events": rows[start : start + 500],
                        "rescore": False,
                    },
                    timeout=120.0,
                )

            warm = post(f"{args.api}/api/stream/rescore", {}, timeout=120.0)
            print(
                f"Baseline: {warm.get('accounts')} accounts "
                f"in {warm.get('milliseconds')} ms"
            )

        state = get(f"{args.api}/api/stream/state")

    print("=" * 60)
    print("REPLAYING UPI EVENT STREAM")
    print("=" * 60)
    print(f"API        : {args.api}")
    print(f"History    : {len(history):,} events")
    print(f"Replaying  : {len(events):,} events")
    print(f"Speed      : {'unthrottled' if args.speed <= 0 else str(args.speed) + 'x'}")
    print(f"Batch size : {args.batch}")
    print(f"Window     : {events['timestamp'].min()} -> {events['timestamp'].max()}")
    print("=" * 60)
    print()

    started = time.time()

    previous_stamp = None
    sent = 0
    rings_seen = state.get("active_rings", 0)

    batch: list = []

    def flush(force_rescore: bool) -> None:

        nonlocal batch, sent, rings_seen

        if not batch:
            return

        payload = {
            "events": batch,
            "rescore": True if force_rescore else None,
        }

        try:
            result = post(f"{args.api}/api/stream/events", payload)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  ingest failed: {exc}")
            batch = []
            return

        sent += result.get("accepted", 0)

        rejected = result.get("rejected") or []

        if rejected:
            print(f"  {len(rejected)} event(s) rejected: {rejected[0]}")

        rings = result.get("rings", rings_seen)

        if rings != rings_seen:
            direction = "+" if rings > rings_seen else ""
            print(
                f"  [{sent:>5}/{len(events)}] rings {rings_seen} -> {rings} "
                f"({direction}{rings - rings_seen})"
            )
            rings_seen = rings

        elif result.get("rescored"):
            print(
                f"  [{sent:>5}/{len(events)}] rescored in "
                f"{result.get('rescore_ms')} ms, {rings} rings"
            )

        batch = []

    for row in events.itertuples(index=False):

        # Sleep in proportion to the gap between simulated events.
        if args.speed > 0 and previous_stamp is not None:

            gap = (row.timestamp - previous_stamp).total_seconds()

            if gap > 0:
                time.sleep(min(gap / args.speed, 2.0))

        previous_stamp = row.timestamp

        batch.append(to_event(row))

        if len(batch) >= args.batch:
            flush(force_rescore=False)

    flush(force_rescore=True)

    # `flush` returns early on an empty batch, so force the final rescore
    # explicitly - otherwise a replay whose length divides evenly by the
    # batch size ends without ever scoring its last events.
    try:
        post(f"{args.api}/api/stream/rescore", {}, timeout=120.0)
    except (urllib.error.URLError, OSError):
        pass

    elapsed = time.time() - started

    final = get(f"{args.api}/api/stream/state")

    print()
    print("=" * 60)
    print("REPLAY COMPLETE")
    print("=" * 60)
    print(f"Events sent      : {sent:,} in {elapsed:.1f}s")
    print(f"Rate             : {sent / max(elapsed, 1e-6):.1f} events/second")
    print(f"Accounts known   : {final.get('accounts_known')}")
    print(f"Flagged accounts : {final.get('flagged_accounts')}")
    print(f"Active rings     : {final.get('active_rings')}")
    print(f"Rescores         : {final.get('rescores')}")
    print(f"Last rescore     : {final.get('last_rescore_ms')} ms")
    print(f"Rejected         : {final.get('events_rejected')}")
    print()
    print("Open http://127.0.0.1:8000/dashboard#live to see the result.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
