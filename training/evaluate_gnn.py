"""
Evaluate the GraphSAGE baseline on the held-out test split.

Metrics are reported on `data.test_mask` only. The threshold is the one
selected on the validation split during training, not re-tuned here.
"""

from pathlib import Path

import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.graphsage_model import GraphSAGE


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "processed" / "graph_data.pt"
MODEL_FILE = ROOT / "models" / "artifacts" / "gnn_model.pt"
METRICS_FILE = ROOT / "models" / "artifacts" / "graphsage_eval_metrics.json"


def evaluate_split(y_true, y_prob, threshold):

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
    except ValueError:
        # Only one class present in this split.
        metrics["roc_auc"] = None

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics["confusion_matrix"] = {
        "true_negative": int(matrix[0][0]),
        "false_positive": int(matrix[0][1]),
        "false_negative": int(matrix[1][0]),
        "true_positive": int(matrix[1][1]),
    }

    return metrics


def main() -> None:

    print("=" * 60)
    print("GRAPHSAGE EVALUATION (held-out test split)")
    print("=" * 60)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}. "
            "Run: python -m training.train_gnn"
        )

    package = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )

    data = package["data"]

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    model = GraphSAGE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold = float(
        checkpoint.get("decision_threshold", 0.50)
    )

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(data.x, data.edge_index)
        ).numpy()

    y_true = data.y.numpy()

    test_mask = data.test_mask.numpy()
    val_mask = data.val_mask.numpy()

    results = {
        "model": "GraphSAGE (homogeneous baseline)",
        "decision_threshold": threshold,
        "test_nodes": int(test_mask.sum()),
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
        f"Test accounts: {results['test_nodes']} "
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

    METRICS_FILE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("Metrics saved:", METRICS_FILE)
    print("Evaluation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
