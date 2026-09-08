"""
Tests for the temporal heterogeneous graph and model.

Covers heterogeneous graph construction, node feature validity, timestamp
validity, absence of NaN/inf, the forward pass, and - most importantly - that
the model actually consumes edge time rather than merely accepting it.
"""

from pathlib import Path

import pytest
import torch

from models.temporal_hetero_gnn import (
    TimeEncoder,
    build_model,
    edge_time_dict_from,
    in_channels_from,
)


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "processed" / "hetero_graph_data.pt"

EXPECTED_NODE_TYPES = {"account", "device", "vpa", "merchant"}


pytestmark = pytest.mark.skipif(
    not GRAPH_FILE.exists(),
    reason="Run `python -m graph.prepare_hetero_graph` first.",
)


@pytest.fixture(scope="module")
def data():
    return torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    ).to("cpu")


@pytest.fixture(scope="module")
def model(data):
    torch.manual_seed(0)
    built = build_model(data, hidden_dim=32, time_dim=8, dropout=0.0)
    built.eval()
    return built


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def test_expected_node_types_are_present(data):
    assert EXPECTED_NODE_TYPES.issubset(set(data.node_types))


def test_account_relations_exist(data):

    relations = {relation[1] for relation in data.edge_types}

    for expected in ("transfers", "logs_in", "owns", "collects_at"):
        assert expected in relations, f"missing relation: {expected}"


def test_every_relation_has_edges(data):
    for relation in data.edge_types:
        assert data[relation].edge_index.shape[0] == 2
        assert data[relation].edge_index.shape[1] > 0


def test_edge_indices_point_at_real_nodes(data):

    for relation in data.edge_types:

        source_type, _, target_type = relation

        edge_index = data[relation].edge_index

        assert int(edge_index[0].min()) >= 0
        assert int(edge_index[1].min()) >= 0

        assert int(edge_index[0].max()) < data[source_type].x.shape[0]
        assert int(edge_index[1].max()) < data[target_type].x.shape[0]


# ============================================================
# NODE FEATURES
# ============================================================

def test_node_features_are_valid(data):

    for node_type in data.node_types:

        x = data[node_type].x

        assert x.shape[0] > 0, f"{node_type} has no nodes"
        assert x.shape[1] > 0, f"{node_type} has no features"
        assert torch.isfinite(x).all(), f"{node_type} has NaN or inf"


def test_account_features_are_not_constant(data):
    """
    Constant features carry no signal. The previous heterogeneous builder
    filled every device/vpa/merchant row with ones.
    """

    assert float(data["account"].x.std()) > 0.0

    for node_type in ("device", "vpa", "merchant"):
        assert float(data[node_type].x.std()) > 0.0, (
            f"{node_type} features are constant"
        )


# ============================================================
# TIMESTAMPS
# ============================================================

def test_relations_carry_edge_time(data):

    timed = edge_time_dict_from(data)

    assert timed, "no relation carries edge_time"

    for relation in data.edge_types:
        assert "edge_time" in data[relation], (
            f"{relation} has no edge_time"
        )


def test_edge_times_are_valid(data):

    for relation, edge_time in edge_time_dict_from(data).items():

        assert torch.isfinite(edge_time).all(), (
            f"{relation} has NaN or inf edge_time"
        )

        assert edge_time.shape[0] == data[relation].edge_index.shape[1], (
            f"{relation} edge_time length does not match edge count"
        )

        assert float(edge_time.min()) >= 0.0
        assert float(edge_time.max()) <= 1.0


def test_edge_times_actually_vary(data):
    """
    A constant time column would make the temporal gate a no-op.
    """

    edge_time = data["account", "transfers", "account"].edge_time

    assert float(edge_time.std()) > 0.0


# ============================================================
# LABELS AND SPLITS
# ============================================================

def test_labels_are_valid(data):

    y = data["account"].y

    assert y.shape[0] == data["account"].x.shape[0]
    assert torch.isfinite(y.float()).all()
    assert set(y.unique().tolist()).issubset({0, 1, 0.0, 1.0})
    assert int(y.sum()) > 0


def test_masks_are_disjoint_and_complete(data):

    account = data["account"]

    total = (
        account.train_mask.sum()
        + account.val_mask.sum()
        + account.test_mask.sum()
    )

    assert int(total) == account.x.shape[0]

    assert not bool((account.train_mask & account.val_mask).any())
    assert not bool((account.train_mask & account.test_mask).any())
    assert not bool((account.val_mask & account.test_mask).any())


def test_every_split_contains_fraud_accounts(data):

    account = data["account"]

    for mask in (account.train_mask, account.val_mask, account.test_mask):
        assert int(account.y[mask].sum()) > 0


# ============================================================
# TIME ENCODER
# ============================================================

def test_time_encoder_shape_and_finiteness():

    encoder = TimeEncoder(dim=8)

    t = torch.linspace(0.0, 1.0, steps=32)

    encoding = encoder(t)

    assert encoding.shape == (32, 8)
    assert torch.isfinite(encoding).all()


def test_time_encoder_separates_different_times():

    encoder = TimeEncoder(dim=8)

    early = encoder(torch.tensor([0.0]))
    late = encoder(torch.tensor([1.0]))

    assert not torch.allclose(early, late)


# ============================================================
# MODEL
# ============================================================

def test_in_channels_match_the_graph(data):

    channels = in_channels_from(data)

    for node_type in data.node_types:
        assert channels[node_type] == data[node_type].x.shape[1]


def test_forward_pass(data, model):

    with torch.no_grad():
        logits = model(
            data.x_dict,
            data.edge_index_dict,
            edge_time_dict_from(data),
        )

    assert logits.shape[0] == data["account"].x.shape[0]
    assert logits.dim() == 1
    assert torch.isfinite(logits).all()


def test_forward_pass_without_edge_time(data, model):
    """
    Withholding time must still produce a valid result - that path is the
    ablation and the fallback for relations with no timestamp.
    """

    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict, {})

    assert logits.shape[0] == data["account"].x.shape[0]
    assert torch.isfinite(logits).all()


def test_probabilities_are_in_range(data, model):

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(
                data.x_dict,
                data.edge_index_dict,
                edge_time_dict_from(data),
            )
        )

    assert float(probabilities.min()) >= 0.0
    assert float(probabilities.max()) <= 1.0


def test_edge_time_changes_the_output(data, model):
    """
    The point of this model. If passing edge times produced identical output
    to withholding them, the temporal machinery would be decorative - which
    is exactly what the previous version of this file did.
    """

    with torch.no_grad():

        timed = model(
            data.x_dict,
            data.edge_index_dict,
            edge_time_dict_from(data),
        )

        untimed = model(data.x_dict, data.edge_index_dict, {})

    assert not torch.allclose(timed, untimed, atol=1e-6)


def test_shuffling_edge_times_changes_the_output(data, model):
    """
    Stronger than the previous test: the *values* of the timestamps must
    matter, not merely whether the argument was supplied.
    """

    torch.manual_seed(0)

    original = edge_time_dict_from(data)

    shuffled = {
        relation: edge_time[torch.randperm(edge_time.shape[0])]
        for relation, edge_time in original.items()
    }

    with torch.no_grad():
        a = model(data.x_dict, data.edge_index_dict, original)
        b = model(data.x_dict, data.edge_index_dict, shuffled)

    assert not torch.allclose(a, b, atol=1e-6)
