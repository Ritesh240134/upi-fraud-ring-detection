"""
Train the GraphSAGE baseline on the homogeneous account transfer graph.

The model is trained on the training mask only. The validation mask is used
for model selection (best validation loss) and for choosing the decision
threshold. The test mask is never touched here - see training/evaluate_gnn.py.
"""

from pathlib import Path

import json
import random

import numpy as np
import torch
from sklearn.metrics import f1_score

from models.graphsage_model import GraphSAGE


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "processed" / "graph_data.pt"
MODEL_DIR = ROOT / "models" / "artifacts"
MODEL_FILE = MODEL_DIR / "gnn_model.pt"
METRICS_FILE = MODEL_DIR / "training_metrics.json"

SEED = 42
EPOCHS = 300
HIDDEN_DIM = 64
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple:
    """
    Pick the decision threshold that maximises F1 on the validation split.
    """

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


def main() -> None:

    print("=" * 60)
    print("GRAPHSAGE TRAINING (baseline)")
    print("=" * 60)

    set_seed(SEED)

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_FILE}. "
            "Run: python -m graph.prepare_graph"
        )

    package = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )

    data = package["data"]

    x = data.x
    edge_index = data.edge_index
    y = data.y

    train_mask = data.train_mask
    val_mask = data.val_mask

    print("Nodes:", x.shape[0])
    print("Features:", x.shape[1])
    print("Edges:", edge_index.shape[1])
    print("Positive labels:", int(y.sum()))
    print(
        "Train / val nodes:",
        int(train_mask.sum()),
        "/",
        int(val_mask.sum()),
    )

    model = GraphSAGE(
        input_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    positives = float(y[train_mask].sum())
    negatives = float(int(train_mask.sum()) - positives)

    pos_weight = torch.tensor(
        [max(1.0, negatives / positives) if positives > 0 else 1.0],
        dtype=torch.float32,
    )

    print("Positive class weight:", round(float(pos_weight), 3))

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)

        loss = criterion(
            logits[train_mask],
            y[train_mask],
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():

            val_logits = model(x, edge_index)

            val_loss = criterion(
                val_logits[val_mask],
                y[val_mask],
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss.item()),
            }
        )

        if val_loss.item() < best_val_loss:

            best_val_loss = float(val_loss.item())
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {loss.item():.4f} | "
                f"Val Loss {val_loss.item():.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------------------------------------------------------
    # Decision threshold from the validation split
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(x, edge_index)
        ).numpy()

    threshold, val_f1 = select_threshold(
        probabilities[val_mask.numpy()],
        y[val_mask].numpy(),
    )

    print()
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print("Validation threshold:", threshold)
    print("Validation F1:", round(val_f1, 4))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "hidden_dim": HIDDEN_DIM,
            "decision_threshold": float(threshold),
            "feature_names": package.get("feature_names", []),
        },
        MODEL_FILE,
    )

    metrics = {
        "model": "GraphSAGE (homogeneous baseline)",
        "seed": SEED,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "final_train_loss": history[-1]["train_loss"],
        "validation_f1": float(val_f1),
        "decision_threshold": float(threshold),
        "nodes": int(x.shape[0]),
        "edges": int(edge_index.shape[1]),
        "features": int(x.shape[1]),
        "positive_labels": int(y.sum()),
        "train_nodes": int(train_mask.sum()),
        "val_nodes": int(val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print()
    print("TRAINING SUCCESS")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
