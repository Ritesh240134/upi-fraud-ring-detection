"""
Evaluate the temporal heterogeneous GNN on the held-out test split.

The threshold is the one chosen on validation during training. It is not
re-tuned on the test split.

Usage:
    python -m training.evaluate_temporal_hetero
    python -m training.evaluate_temporal_hetero --no-time    # ablation model
"""

from pathlib import Path

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.temporal_hetero_gnn import build_model, edge_time_dict_from


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "processed" / "hetero_graph_data.pt"
ARTIFACT_DIR = ROOT / "models" / "artifacts"


def evaluate_split(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_prob)
        )
    except ValueError:
        # Only one class present in this split.
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics["confusion_matrix"] = {
        "true_negative": int(matrix[0][0]),
        "false_positive": int(matrix[0][1]),
        "false_negative": int(matrix[1][0]),
        "true_positive": int(matrix[1][1]),
    }

    return metrics


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Evaluate the temporal heterogeneous GNN."
    )

    parser.add_argument(
        "--no-time",
        action="store_true",
        help="Evaluate the time-ablated checkpoint.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    use_time = not args.no_time

    suffix = "" if use_time else "_no_time"

    model_file = ARTIFACT_DIR / f"temporal_hetero_gnn{suffix}.pt"
    output_file = (
        ARTIFACT_DIR / f"temporal_hetero_eval_metrics{suffix}.json"
    )

    print("=" * 60)
    print("TEMPORAL HETEROGENEOUS GNN EVALUATION")
    print("=" * 60)
    print("Edge times used:", use_time)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}. "
            "Run: python -m training.train_temporal_hetero"
            + (" --no-time" if not use_time else "")
        )

    data = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    ).to("cpu")

    checkpoint = torch.load(
        model_file,
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

    threshold = float(checkpoint["decision_threshold"])

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

    y_true = data["account"].y.numpy()

    val_mask = data["account"].val_mask.numpy()
    test_mask = data["account"].test_mask.numpy()

    results = {
        "model": "TemporalHeteroGNN",
        "uses_edge_time": bool(checkpoint.get("uses_edge_time", True)),
        "decision_threshold": threshold,
        "test_accounts": int(test_mask.sum()),
        "test_fraud_accounts": int(y_true[test_mask].sum()),
        "validation": evaluate_split(
            y_true[val_mask],
            probabilities[val_mask],
            threshold,
        ),
        "test": evaluate_split(
            y_true[test_mask],
            probabilities[test_mask],
            threshold,
        ),
    }

    print()
    print(f"Decision threshold: {threshold}")
    print(
        f"Test accounts: {results['test_accounts']} "
        f"({results['test_fraud_accounts']} fraud)"
    )

    for split in ("validation", "test"):

        print()
        print(f"{split.upper()} METRICS")
        print("-" * 30)

        for key, value in results[split].items():

            if key == "confusion_matrix":
                continue

            if value is None:
                print(f"{key}: N/A")
            else:
                print(f"{key}: {value:.4f}")

        matrix = results[split]["confusion_matrix"]

        print(
            "confusion_matrix: "
            f"TN={matrix['true_negative']} "
            f"FP={matrix['false_positive']} "
            f"FN={matrix['false_negative']} "
            f"TP={matrix['true_positive']}"
        )

    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("Metrics saved:", output_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
