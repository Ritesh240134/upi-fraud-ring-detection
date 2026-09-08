"""
Model serving layer for the fraud API.

Loads the trained temporal heterogeneous GNN once and keeps a table of
per-account fraud probabilities in memory, so scoring a transaction is a
dictionary lookup rather than a forward pass.

Every failure path is soft. If torch, the graph or the checkpoint is missing,
`ModelService.available` stays False and the API falls back to the rule layer
instead of failing to start - a scoring service that refuses to boot because a
checkpoint moved is worse than one that degrades.
"""

from pathlib import Path
from typing import Optional

import threading


ROOT = Path(__file__).resolve().parent.parent

GRAPH_FILE = ROOT / "data" / "processed" / "hetero_graph_data.pt"
MODEL_FILE = ROOT / "models" / "artifacts" / "temporal_hetero_gnn.pt"

DEFAULT_THRESHOLD = 0.50


class ModelService:
    """
    Thread-safe lazy loader for the trained account risk model.
    """

    def __init__(self) -> None:

        self._lock = threading.Lock()
        self._loaded = False

        self.available = False
        self.error: Optional[str] = None

        self.scores = {}
        self.threshold = DEFAULT_THRESHOLD
        self.model_name = "temporal_hetero_gnn"
        self.uses_edge_time = True
        self.account_count = 0

    # --------------------------------------------------------

    def load(self) -> None:
        """
        Load the checkpoint and precompute account probabilities.
        Safe to call repeatedly; the work happens once.
        """

        with self._lock:

            if self._loaded:
                return

            self._loaded = True

            try:
                self._load_unsafe()
                self.available = True

            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                self.available = False
                self.error = f"{type(exc).__name__}: {exc}"

    def _load_unsafe(self) -> None:

        import torch

        from models.temporal_hetero_gnn import (
            build_model,
            edge_time_dict_from,
        )

        if not GRAPH_FILE.exists():
            raise FileNotFoundError(
                f"Graph not found: {GRAPH_FILE}"
            )

        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_FILE}"
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

        self.threshold = float(
            checkpoint.get("decision_threshold", DEFAULT_THRESHOLD)
        )

        self.uses_edge_time = bool(
            checkpoint.get("uses_edge_time", True)
        )

        edge_time_dict = (
            edge_time_dict_from(data) if self.uses_edge_time else {}
        )

        with torch.no_grad():
            probabilities = torch.sigmoid(
                model(
                    data.x_dict,
                    data.edge_index_dict,
                    edge_time_dict,
                )
            ).tolist()

        self.scores = {
            str(account): float(probability)
            for account, probability in zip(
                data.account_ids,
                probabilities,
            )
        }

        self.account_count = len(self.scores)

    # --------------------------------------------------------

    def account_score(self, account_id: Optional[str]) -> Optional[float]:
        """
        Fraud probability for a known account, or None if unseen.
        """

        if not self.available or not account_id:
            return None

        return self.scores.get(str(account_id))

    def info(self) -> dict:

        return {
            "model": self.model_name,
            "available": self.available,
            "uses_edge_time": self.uses_edge_time,
            "scored_accounts": self.account_count,
            "decision_threshold": self.threshold,
            "error": self.error,
        }


# Module-level singleton used by the API.
model_service = ModelService()
