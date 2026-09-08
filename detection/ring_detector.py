import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "synthetic" / "upi_events.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "detected_rings.csv"

# Detection parameters
TIME_WINDOW_MINUTES = 10
MIN_RING_SIZE = 3
MIN_FRAUD_EVENTS = 2


def load_events():
    print("Loading UPI events...")
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} events.")
    return df


def build_temporal_graph(df):
    print("Building temporal fraud graph...")

    transfers = df[df["event_type"] == "TRANSFER"].copy()

    G = nx.Graph()

    for _, row in transfers.iterrows():

        source = row["source_account"]
        target = row["target_account"]

        if pd.isna(source) or pd.isna(target):
            continue

        source = str(source)
        target = str(target)

        G.add_node(source)
        G.add_node(target)

        G.add_edge(
            source,
            target,
            timestamp=row["timestamp"],
            amount=float(row["amount"]),
            is_fraud=int(row["is_fraud"]),
            typology=row["typology"],
            ring_id=row["ring_id"]
        )

    print(f"Accounts : {G.number_of_nodes()}")
    print(f"Edges    : {G.number_of_edges()}")

    return G


def detect_fraud_rings(G):
    print("Detecting suspicious connected components...")

    suspicious_edges = []

    for u, v, data in G.edges(data=True):

        if data.get("is_fraud", 0) == 1:
            suspicious_edges.append((u, v, data))

    print(f"Fraud-related edges: {len(suspicious_edges)}")

    fraud_graph = nx.Graph()

    for u, v, data in suspicious_edges:
        fraud_graph.add_edge(u, v, **data)

    components = list(nx.connected_components(fraud_graph))

    print(f"Candidate components: {len(components)}")

    detected = []

    ring_number = 1

    for component in components:

        if len(component) < MIN_RING_SIZE:
            continue

        subgraph = fraud_graph.subgraph(component)

        fraud_edges = [
            data
            for _, _, data in subgraph.edges(data=True)
            if data.get("is_fraud", 0) == 1
        ]

        if len(fraud_edges) < MIN_FRAUD_EVENTS:
            continue

        typologies = [
            data.get("typology", "")
            for data in fraud_edges
            if data.get("typology", "")
        ]

        ring_ids = [
            data.get("ring_id", "")
            for data in fraud_edges
            if data.get("ring_id", "")
        ]

        detected.append({
            "detected_ring_id": f"DETECTED_RING_{ring_number}",
            "accounts": ",".join(sorted(component)),
            "ring_size": len(component),
            "fraud_events": len(fraud_edges),
            "typologies": ",".join(sorted(set(typologies))),
            "ground_truth_ring_ids": ",".join(
                sorted(set(map(str, ring_ids)))
            ),
            "risk_score": min(
                1.0,
                0.5 + (len(fraud_edges) / 20.0)
            )
        })

        ring_number += 1

    return detected


def save_results(rings):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = pd.DataFrame(rings)

    if result.empty:
        result = pd.DataFrame(columns=[
            "detected_ring_id",
            "accounts",
            "ring_size",
            "fraud_events",
            "typologies",
            "ground_truth_ring_ids",
            "risk_score"
        ])

    result.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 60)
    print("FRAUD RING DETECTION COMPLETED")
    print("=" * 60)

    print(f"Detected rings : {len(result)}")
    print(f"Saved file     : {OUTPUT_FILE}")

    if len(result) > 0:
        print()
        print("Top detected rings:")
        print(
            result[
                [
                    "detected_ring_id",
                    "ring_size",
                    "fraud_events",
                    "typologies",
                    "risk_score"
                ]
            ].head(10).to_string(index=False)
        )
    else:
        print("No fraud rings detected.")


def main():

    df = load_events()

    G = build_temporal_graph(df)

    rings = detect_fraud_rings(G)

    save_results(rings)


if __name__ == "__main__":
    main()