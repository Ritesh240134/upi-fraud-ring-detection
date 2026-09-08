"""
Tests for the homogeneous GraphSAGE baseline.

Covers graph construction, feature validity, label validity, the
train/val/test split, and the model forward pass.
"""

from pathlib import Path

import pytest
import torch

from graph.prepare_graph import FEATURE_NAMES
from models.graphsage_model import GraphSAGE


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "processed" / "graph_data.pt"


pytestmark = pytest.mark.skipif(
    not GRAPH_FILE.exists(),
    reason="Run `python -m graph.prepare_graph` first.",
)


@pytest.fixture(scope="module")
def package():
    return torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )


@pytest.fixture(scope="module")
def data(package):
    return package["data"]


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def test_graph_exists():
    assert GRAPH_FILE.exists()


def test_graph_has_nodes_and_edges(data):
    assert data.x.shape[0] > 0
    assert data.x.shape[1] > 0
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] > 0


def test_edge_indices_are_in_range(data):
    num_nodes = data.x.shape[0]

    assert int(data.edge_index.min()) >= 0
    assert int(data.edge_index.max()) < num_nodes


def test_account_list_matches_node_count(package, data):
    assert len(package["accounts"]) == data.x.shape[0]
    assert len(package["account_to_id"]) == data.x.shape[0]


# ============================================================
# FEATURES
# ============================================================

def test_features_contain_no_nan_or_inf(data):
    assert torch.isfinite(data.x).all()


def test_feature_names_match_feature_count(package, data):
    assert len(package["feature_names"]) == data.x.shape[1]


def test_features_do_not_leak_the_label(package):
    """
    The label comes from `is_fraud` / `ring_id` / `typology`. No feature may
    be derived from those columns - an earlier version of the pipeline used a
    per-account fraud-event count as feature 7, which made the task trivial.
    """

    forbidden = ("fraud", "ring", "typology", "label")

    for name in package["feature_names"]:
        assert not any(word in name.lower() for word in forbidden), (
            f"Feature '{name}' looks label-derived"
        )

    assert package["feature_names"] == FEATURE_NAMES


# ============================================================
# LABELS
# ============================================================

def test_labels_are_valid(data):
    assert torch.isfinite(data.y).all()
    assert set(data.y.unique().tolist()).issubset({0.0, 1.0})
    assert data.y.shape[0] == data.x.shape[0]


def test_both_classes_are_present(data):
    assert int((data.y == 1).sum()) > 0
    assert int((data.y == 0).sum()) > 0


# ============================================================
# SPLITS
# ============================================================

def test_masks_exist_and_cover_every_node(data):

    for name in ("train_mask", "val_mask", "test_mask"):
        assert hasattr(data, name), f"missing {name}"

    total = (
        data.train_mask.sum()
        + data.val_mask.sum()
        + data.test_mask.sum()
    )

    assert int(total) == data.x.shape[0]


def test_masks_are_disjoint(data):
    assert not bool((data.train_mask & data.val_mask).any())
    assert not bool((data.train_mask & data.test_mask).any())
    assert not bool((data.val_mask & data.test_mask).any())


def test_every_split_contains_fraud_accounts(data):
    """
    The split is stratified, so a fraud-free test split would mean the
    stratification broke and ROC-AUC would be undefined.
    """

    for mask in (data.train_mask, data.val_mask, data.test_mask):
        assert int(data.y[mask].sum()) > 0


# ============================================================
# MODEL
# ============================================================

def test_model_forward_pass(data):

    model = GraphSAGE(
        input_dim=data.x.shape[1],
        hidden_dim=64,
    )

    model.eval()

    with torch.no_grad():
        output = model(data.x, data.edge_index)

    assert output.shape[0] == data.x.shape[0]
    assert output.dim() == 1
    assert torch.isfinite(output).all()


def test_model_probabilities_are_in_range(data):

    model = GraphSAGE(
        input_dim=data.x.shape[1],
        hidden_dim=64,
    )

    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(data.x, data.edge_index)
        )

    assert float(probabilities.min()) >= 0.0
    assert float(probabilities.max()) <= 1.0
