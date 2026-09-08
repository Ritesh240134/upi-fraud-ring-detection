"""
GNN-driven fraud-ring detection.

Pipeline
--------
    1. Score every account with the trained temporal heterogeneous GNN.
    2. Score each transfer edge by the geometric mean of its endpoints.
    3. Slide a time window over the suspicious edges and extract connected
       components *inside* each window.
    4. Deduplicate overlapping candidates and rank by risk.

Why the sliding window matters
------------------------------
An earlier version built one graph from every suspicious edge in the dataset
and then discarded any component spanning more than the time window. Because
all the suspicious edges collapse into a single large component spanning the
whole dataset, that check rejected every candidate and the detector always
returned zero rings. Components must be built per window, not filtered after.

Outputs:
    data/processed/account_risk_scores.csv
    data/processed/gnn_detected_rings.csv
"""

from pathlib import Path

import networkx as nx
import pandas as pd
import torch

from models.temporal_hetero_gnn import build_model, edge_time_dict_from


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "processed" / "hetero_graph_data.pt"
EVENT_FILE = ROOT / "data" / "synthetic" / "upi_events.csv"
MODEL_FILE = ROOT / "models" / "artifacts" / "temporal_hetero_gnn.pt"

RING_FILE = ROOT / "data" / "processed" / "gnn_detected_rings.csv"
SCORE_FILE = ROOT / "data" / "processed" / "account_risk_scores.csv"

TIME_WINDOW_MINUTES = 10
PAIR_THRESHOLD = 0.50
MIN_RING_SIZE = 3

RING_COLUMNS = [
    "detected_ring_id",
    "accounts",
    "ring_size",
    "suspicious_edges",
    "window_start",
    "window_end",
    "time_span_minutes",
    "ring_risk_score",
    "max_account_score",
    "mean_account_score",
    "total_amount",
    "matched_ground_truth_rings",
    "true_fraud_accounts",
    "precision_vs_ground_truth",
]


# ============================================================
# SCORING
# ============================================================

def score_accounts() -> pd.DataFrame:
    """
    Run the trained model over the whole graph and return one row per account.
    """

    print("Loading heterogeneous graph and model...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}. "
            "Run: python -m training.train_temporal_hetero"
        )

    data = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    ).to("cpu")

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model(
        data,
        hidden_dim=checkpoint["hidden_dim"],
        time_dim=checkpoint["time_dim"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    edge_time_dict = (
        edge_time_dict_from(data)
        if checkpoint.get("uses_edge_time", True)
        else {}
    )

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(
                data.x_dict,
                data.edge_index_dict,
                edge_time_dict,
            )
        ).numpy()

    # The graph builder stores the account order it used, so scores line up
    # with account ids without re-deriving the sort.
    accounts = list(data.account_ids)

    scores = pd.DataFrame(
        {
            "account": accounts,
            "fraud_probability": probabilities,
            "true_label": data["account"].y.numpy().astype(int),
        }
    )

    scores["risk_level"] = pd.cut(
        scores["fraud_probability"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)

    scores.to_csv(SCORE_FILE, index=False)

    print()
    print("=" * 60)
    print("ACCOUNT RISK SCORES")
    print("=" * 60)
    print("Accounts:", len(scores))
    print(
        "High-risk accounts (>= 0.50):",
        int((scores["fraud_probability"] >= 0.50).sum()),
    )
    print(
        "Maximum probability:",
        round(float(scores["fraud_probability"].max()), 4),
    )
    print(
        "Mean probability:",
        round(float(scores["fraud_probability"].mean()), 4),
    )
    print("Saved:", SCORE_FILE)

    return scores


# ============================================================
# SUSPICIOUS EDGES
# ============================================================

def suspicious_transfers(scores: pd.DataFrame) -> pd.DataFrame:

    events = pd.read_csv(EVENT_FILE)

    events["timestamp"] = pd.to_datetime(
        events["timestamp"],
        errors="coerce",
    )

    events = events.dropna(
        subset=["timestamp", "source_account", "target_account"]
    )

    events = events[
        events["event_type"].astype(str).str.upper() == "TRANSFER"
    ].copy()

    score_map = dict(
        zip(
            scores["account"].astype(str),
            scores["fraud_probability"],
        )
    )

    events["source_score"] = (
        events["source_account"].astype(str).map(score_map).fillna(0.0)
    )

    events["target_score"] = (
        events["target_account"].astype(str).map(score_map).fillna(0.0)
    )

    # Geometric mean: both endpoints must look risky for the edge to count.
    events["pair_score"] = (
        events["source_score"] * events["target_score"]
    ) ** 0.5

    suspicious = events[
        events["pair_score"] >= PAIR_THRESHOLD
    ].copy()

    suspicious = suspicious.sort_values("timestamp").reset_index(drop=True)

    print()
    print("Transfer events:", len(events))
    print("Suspicious GNN edges:", len(suspicious))

    return suspicious


# ============================================================
# SLIDING WINDOW RING EXTRACTION
# ============================================================

def extract_candidates(suspicious: pd.DataFrame, scores: pd.DataFrame) -> list:
    """
    Slide a TIME_WINDOW_MINUTES window across the suspicious edges, anchoring
    it at each edge, and collect connected components found inside it.
    """

    score_map = dict(
        zip(
            scores["account"].astype(str),
            scores["fraud_probability"],
        )
    )

    window = pd.Timedelta(minutes=TIME_WINDOW_MINUTES)

    timestamps = suspicious["timestamp"]

    seen = {}

    for start in range(len(suspicious)):

        window_end = timestamps.iloc[start] + window

        stop = timestamps.searchsorted(window_end, side="right")

        if stop - start < MIN_RING_SIZE - 1:
            continue

        block = suspicious.iloc[start:stop]

        graph = nx.Graph()

        for row in block.itertuples(index=False):

            source = str(row.source_account)
            target = str(row.target_account)

            if source == target:
                continue

            graph.add_edge(
                source,
                target,
                timestamp=row.timestamp,
                amount=float(row.amount),
                pair_score=float(row.pair_score),
                ring_id=getattr(row, "ring_id", None),
                is_fraud=int(getattr(row, "is_fraud", 0) or 0),
            )

        for component in nx.connected_components(graph):

            if len(component) < MIN_RING_SIZE:
                continue

            accounts = tuple(sorted(component))

            subgraph = graph.subgraph(accounts)

            edges = [
                edge_data
                for _, _, edge_data in subgraph.edges(data=True)
            ]

            if not edges:
                continue

            edge_times = sorted(edge["timestamp"] for edge in edges)

            span_minutes = (
                edge_times[-1] - edge_times[0]
            ).total_seconds() / 60.0

            pair_scores = [edge["pair_score"] for edge in edges]

            ring_score = sum(pair_scores) / len(pair_scores)

            account_scores = [
                score_map.get(account, 0.0) for account in accounts
            ]

            ground_truth = sorted(
                {
                    str(edge["ring_id"])
                    for edge in edges
                    if edge.get("ring_id")
                    and str(edge["ring_id"]).lower() != "nan"
                }
            )

            candidate = {
                "accounts": accounts,
                "ring_size": len(accounts),
                "suspicious_edges": len(edges),
                "window_start": edge_times[0],
                "window_end": edge_times[-1],
                "time_span_minutes": round(span_minutes, 2),
                "ring_risk_score": round(min(1.0, ring_score), 4),
                "max_account_score": round(max(account_scores), 4),
                "mean_account_score": round(
                    sum(account_scores) / len(account_scores), 4
                ),
                "total_amount": round(
                    sum(edge["amount"] for edge in edges), 2
                ),
                "matched_ground_truth_rings": ground_truth,
            }

            # Keep the strongest version of each distinct account set.
            existing = seen.get(accounts)

            if (
                existing is None
                or candidate["ring_risk_score"] > existing["ring_risk_score"]
            ):
                seen[accounts] = candidate

    return list(seen.values())


def drop_subsets(candidates: list) -> list:
    """
    Remove candidates whose account set is contained in a larger candidate
    covering the same activity.
    """

    ordered = sorted(
        candidates,
        key=lambda item: (-item["ring_size"], -item["ring_risk_score"]),
    )

    kept = []

    for candidate in ordered:

        account_set = set(candidate["accounts"])

        if any(
            account_set.issubset(set(other["accounts"]))
            for other in kept
        ):
            continue

        kept.append(candidate)

    return kept


# ============================================================
# GROUND-TRUTH OVERLAP (REPORTING ONLY)
# ============================================================

def annotate(candidates: list, scores: pd.DataFrame) -> list:
    """
    Attach how many accounts in each candidate are genuinely fraudulent.

    This is reporting only - it never influences detection.
    """

    truth = dict(
        zip(
            scores["account"].astype(str),
            scores["true_label"],
        )
    )

    for candidate in candidates:

        hits = sum(
            int(truth.get(account, 0)) for account in candidate["accounts"]
        )

        candidate["true_fraud_accounts"] = hits

        candidate["precision_vs_ground_truth"] = round(
            hits / candidate["ring_size"], 4
        )

    return candidates


# ============================================================
# MAIN
# ============================================================

def detect_rings(scores: pd.DataFrame) -> pd.DataFrame:

    print()
    print("=" * 60)
    print("GNN FRAUD RING DETECTION")
    print("=" * 60)

    suspicious = suspicious_transfers(scores)

    candidates = drop_subsets(
        extract_candidates(suspicious, scores)
    )

    candidates = annotate(candidates, scores)

    candidates.sort(
        key=lambda item: (
            -item["ring_risk_score"],
            -item["ring_size"],
        ),
    )

    rows = []

    for number, candidate in enumerate(candidates, start=1):

        row = dict(candidate)

        row["detected_ring_id"] = f"GNN_RING_{number}"
        row["accounts"] = ",".join(candidate["accounts"])
        row["matched_ground_truth_rings"] = ",".join(
            candidate["matched_ground_truth_rings"]
        )

        rows.append(row)

    result = pd.DataFrame(rows, columns=RING_COLUMNS)

    RING_FILE.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(RING_FILE, index=False)

    print()
    print("=" * 60)
    print("GNN RING DETECTION COMPLETED")
    print("=" * 60)
    print("Detected GNN rings:", len(result))
    print("Saved:", RING_FILE)

    if not result.empty:

        flagged = int(result["ring_size"].sum())
        correct = int(result["true_fraud_accounts"].sum())

        print(
            "Accounts inside detected rings:",
            f"{flagged} ({correct} genuinely fraudulent, "
            f"{correct / flagged:.1%})",
        )

        print()
        print(
            result[
                [
                    "detected_ring_id",
                    "ring_size",
                    "suspicious_edges",
                    "time_span_minutes",
                    "ring_risk_score",
                    "true_fraud_accounts",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    else:
        print("No GNN-based rings found.")

    return result


def main() -> None:

    scores = score_accounts()

    detect_rings(scores)


if __name__ == "__main__":
    main()
