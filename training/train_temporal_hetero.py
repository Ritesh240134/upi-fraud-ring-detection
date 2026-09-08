"""
Train the temporal heterogeneous GNN.

Trains on `train_mask`, selects the checkpoint and decision threshold on
`val_mask`, and never looks at `test_mask` - that is left for
training/evaluate_temporal_hetero.py.

Usage:
    python -m training.train_temporal_hetero
    python -m training.train_temporal_hetero --no-time    # ablation

The ablation runs the identical architecture with edge times withheld, which
reduces every TemporalSAGEConv to plain mean aggregation. Comparing the two
runs shows whether the temporal gating is doing real work.
"""

from pathlib import Path

import argparse
import json
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from models.temporal_hetero_gnn import build_model, edge_time_dict_from


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "processed" / "hetero_graph_data.pt"
ARTIFACT_DIR = ROOT / "models" / "artifacts"

SEED = 42
EPOCHS = 400
PATIENCE = 80
HIDDEN_DIM = 64
TIME_DIM = 16
DROPOUT = 0.30
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple:

    best_threshold = 0.50
    best_score = -1.0

    for step in range(5, 96):

        threshold = step / 100.0

        score = f1_score(
            targets,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train the temporal heterogeneous GNN."
    )

    parser.add_argument(
        "--no-time",
        action="store_true",
        help=(
            "Ablation: withhold edge times so message passing is "
            "untimed mean aggregation."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    use_time = not args.no_time

    suffix = "" if use_time else "_no_time"

    model_file = ARTIFACT_DIR / f"temporal_hetero_gnn{suffix}.pt"
    metrics_file = ARTIFACT_DIR / f"temporal_hetero_metrics{suffix}.json"

    print("=" * 60)
    print("TEMPORAL HETEROGENEOUS GNN TRAINING")
    print("=" * 60)
    print("Edge times used:", use_time)

    set_seed(SEED)

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph not found: {GRAPH_FILE}. "
            "Run: python -m graph.prepare_hetero_graph"
        )

    data = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    ).to("cpu")

    edge_time_dict = (
        edge_time_dict_from(data) if use_time else {}
    )

    model = build_model(
        data,
        hidden_dim=HIDDEN_DIM,
        time_dim=TIME_DIM,
        dropout=DROPOUT,
    )

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    y = data["account"].y.float()

    train_mask = data["account"].train_mask
    val_mask = data["account"].val_mask

    positives = float(y[train_mask].sum())
    negatives = float(int(train_mask.sum()) - positives)

    pos_weight = torch.tensor(
        [max(1.0, negatives / positives) if positives > 0 else 1.0],
        dtype=torch.float32,
    )

    print("Train normal:", int(negatives))
    print("Train fraud:", int(positives))
    print("Positive class weight:", round(float(pos_weight), 3))

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    y_val = y[val_mask].numpy()

    best_val_auc = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    epochs_without_improvement = 0

    print()
    print("Training...")

    for epoch in range(1, args.epochs + 1):

        model.train()
        optimizer.zero_grad()

        logits = model(
            data.x_dict,
            data.edge_index_dict,
            edge_time_dict,
        )

        loss = criterion(logits[train_mask], y[train_mask])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0,
        )

        optimizer.step()

        model.eval()

        with torch.no_grad():

            val_logits = model(
                data.x_dict,
                data.edge_index_dict,
                edge_time_dict,
            )

            val_loss = criterion(val_logits[val_mask], y[val_mask])

            val_probabilities = (
                torch.sigmoid(val_logits[val_mask]).numpy()
            )

        try:
            val_auc = roc_auc_score(y_val, val_probabilities)
        except ValueError:
            val_auc = 0.0

        # Model selection on validation ROC-AUC: with a ~14% positive rate
        # it is far more stable than the weighted loss.
        if val_auc > best_val_auc:

            best_val_auc = float(val_auc)
            best_val_loss = float(val_loss.item())
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {loss.item():.4f} | "
                f"Val Loss {val_loss.item():.4f} | "
                f"Val AUC {val_auc:.4f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no validation gain for {PATIENCE} epochs)."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(
                data.x_dict,
                data.edge_index_dict,
                edge_time_dict,
            )
        ).numpy()

    threshold, val_f1 = select_threshold(
        probabilities[val_mask.numpy()],
        y_val,
    )

    print()
    print("Best epoch:", best_epoch)
    print("Best validation ROC-AUC:", round(best_val_auc, 4))
    print("Validation threshold:", threshold)
    print("Validation F1:", round(val_f1, 4))

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels_dict": {
                node_type: int(data[node_type].x.shape[1])
                for node_type in data.node_types
            },
            "hidden_dim": HIDDEN_DIM,
            "time_dim": TIME_DIM,
            "dropout": DROPOUT,
            "decision_threshold": float(threshold),
            "uses_edge_time": use_time,
        },
        model_file,
    )

    metrics = {
        "model": "TemporalHeteroGNN",
        "uses_edge_time": use_time,
        "seed": SEED,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_validation_roc_auc": best_val_auc,
        "validation_loss_at_best": best_val_loss,
        "validation_f1": float(val_f1),
        "decision_threshold": float(threshold),
        "accounts": int(y.shape[0]),
        "fraud_accounts_total": int(y.sum()),
        "train_accounts": int(train_mask.sum()),
        "val_accounts": int(val_mask.sum()),
        "test_accounts": int(data["account"].test_mask.sum()),
        "parameters": int(
            sum(p.numel() for p in model.parameters())
        ),
    }

    metrics_file.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print()
    print("Model saved:", model_file)
    print("Metrics saved:", metrics_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
