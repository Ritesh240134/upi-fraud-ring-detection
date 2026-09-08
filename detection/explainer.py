import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "detected_rings.csv"
EVENT_FILE = ROOT / "data" / "synthetic" / "upi_events.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "ring_explanations.csv"

MAX_NODES = 20


def load_data():
    print("Loading detected rings...")
    rings = pd.read_csv(INPUT_FILE)

    print("Loading UPI events...")
    events = pd.read_csv(EVENT_FILE)
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    return rings, events


def build_explanations(rings, events):
    explanations = []

    for _, ring in rings.iterrows():

        ring_id = ring["detected_ring_id"]

        accounts = str(ring["accounts"]).split(",")

        # Limit explanation to maximum 20 accounts
        accounts = accounts[:MAX_NODES]

        ring_events = events[
            (
                events["source_account"].astype(str).isin(accounts)
                |
                events["target_account"].astype(str).isin(accounts)
            )
        ].copy()

        ring_events = ring_events.sort_values("timestamp")

        # Keep the most relevant events
        ring_events = ring_events.head(MAX_NODES)

        for _, event in ring_events.iterrows():

            explanations.append({
                "detected_ring_id": ring_id,
                "timestamp": event["timestamp"],
                "event_type": event["event_type"],
                "source_account": event["source_account"],
                "target_account": event["target_account"],
                "amount": event["amount"],
                "is_fraud": event["is_fraud"],
                "typology": event["typology"],
                "ground_truth_ring_id": event["ring_id"]
            })

    return pd.DataFrame(explanations)


def main():

    rings, events = load_data()

    print()
    print("=" * 60)
    print("RING EXPLANATION GENERATION")
    print("=" * 60)

    if rings.empty:
        print("No detected rings available.")
        return

    result = build_explanations(rings, events)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Detected rings      : {len(rings)}")
    print(f"Explanation events  : {len(result)}")
    print(f"Maximum nodes       : {MAX_NODES}")
    print(f"Saved file          : {OUTPUT_FILE}")

    print()
    print("RING EXPLANATION COMPLETED")


if __name__ == "__main__":
    main()