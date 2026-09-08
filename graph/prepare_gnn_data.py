"""
SUPERSEDED - kept for reference, not part of the active pipeline.

`graph/prepare_hetero_graph.py` now produces graph, features, labels and
splits in one step. This module also derived its account features from the
node index, which leaks nothing but carries no signal either. Nothing imports
this file, and its output `data/processed/gnn_data.pt` is no longer read.

--------------------------------------------------------------------------

Prepare the heterogeneous graph for GNN training.

This script:
1. Loads the saved heterogeneous graph.
2. Loads account-level fraud labels.
3. Adds account node features.
4. Adds account labels.
5. Creates train/validation/test masks.
6. Saves the final GNN-ready graph.
"""

from pathlib import Path

import pandas as pd
import torch


# ============================================================
# PATHS
# ============================================================

GRAPH_FILE = Path("data/processed/upi_graph.pt")
LABEL_FILE = Path("data/processed/account_labels.csv")
OUTPUT_FILE = Path("data/processed/gnn_data.pt")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph not found: {GRAPH_FILE}"
        )

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Labels not found: {LABEL_FILE}"
        )

    graph = torch.load(
        GRAPH_FILE,
        weights_only=False
    )

    labels = pd.read_csv(LABEL_FILE)

    return graph, labels


# ============================================================
# PREPARE ACCOUNT FEATURES
# ============================================================

def prepare_account_data(graph, labels):

    num_accounts = graph["account"].num_nodes

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    account_labels = torch.zeros(
        num_accounts,
        dtype=torch.long
    )

    for _, row in labels.iterrows():

        account_id = str(row["account_id"])

        # Account IDs in the simulator are expected to follow
        # the format ACCOUNT_0001, ACCOUNT_0002, etc.
        try:
            index = int(
                account_id.split("_")[-1]
            ) - 1
        except ValueError:
            continue

        if 0 <= index < num_accounts:
            account_labels[index] = int(
                row["label"]
            )

    graph["account"].y = account_labels

    # --------------------------------------------------------
    # Create basic account features
    # --------------------------------------------------------

    # Feature 1: normalized node index
    node_index = torch.arange(
        num_accounts,
        dtype=torch.float32
    )

    if num_accounts > 1:
        node_index = node_index / (
            num_accounts - 1
        )

    # Feature 2: degree from transfer graph
    degree = torch.zeros(
        num_accounts,
        dtype=torch.float32
    )

    if (
        "transfers"
        in graph["account", "transfers", "account"]
    ):

        edge_index = graph[
            "account",
            "transfers",
            "account"
        ].edge_index

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        degree.scatter_add_(
            0,
            source_nodes,
            torch.ones_like(
                source_nodes,
                dtype=torch.float32
            )
        )

        degree.scatter_add_(
            0,
            target_nodes,
            torch.ones_like(
                target_nodes,
                dtype=torch.float32
            )
        )

    # Normalize degree
    max_degree = degree.max()

    if max_degree > 0:
        degree = degree / max_degree

    # Feature 3: login count
    login_count = torch.zeros(
        num_accounts,
        dtype=torch.float32
    )

    if (
        "logs_in"
        in graph["account", "logs_in", "device"]
    ):

        login_edges = graph[
            "account",
            "logs_in",
            "device"
        ].edge_index

        login_accounts = login_edges[0]

        login_count.scatter_add_(
            0,
            login_accounts,
            torch.ones_like(
                login_accounts,
                dtype=torch.float32
            )
        )

    max_login = login_count.max()

    if max_login > 0:
        login_count = login_count / max_login

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    graph["account"].x = torch.stack(
        [
            node_index,
            degree,
            login_count,
        ],
        dim=1
    )

    return graph


# ============================================================
# CREATE TRAIN / VALIDATION / TEST MASKS
# ============================================================

def create_masks(graph):

    num_accounts = graph["account"].num_nodes

    generator = torch.Generator()

    generator.manual_seed(42)

    permutation = torch.randperm(
        num_accounts,
        generator=generator
    )

    train_end = int(
        0.70 * num_accounts
    )

    val_end = int(
        0.85 * num_accounts
    )

    train_indices = permutation[
        :train_end
    ]

    val_indices = permutation[
        train_end:val_end
    ]

    test_indices = permutation[
        val_end:
    ]

    train_mask = torch.zeros(
        num_accounts,
        dtype=torch.bool
    )

    val_mask = torch.zeros(
        num_accounts,
        dtype=torch.bool
    )

    test_mask = torch.zeros(
        num_accounts,
        dtype=torch.bool
    )

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    graph["account"].train_mask = train_mask
    graph["account"].val_mask = val_mask
    graph["account"].test_mask = test_mask

    return graph


# ============================================================
# SAVE
# ============================================================

def save_data(graph):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        graph,
        OUTPUT_FILE
    )

    labels = graph["account"].y

    print()
    print("=" * 60)
    print("GNN DATA PREPARATION COMPLETED")
    print("=" * 60)

    print(
        f"Account nodes : "
        f"{graph['account'].num_nodes}"
    )

    print(
        f"Account features: "
        f"{graph['account'].x.shape}"
    )

    print(
        f"Fraud labels : "
        f"{(labels == 1).sum().item()}"
    )

    print(
        f"Normal labels: "
        f"{(labels == 0).sum().item()}"
    )

    print()
    print(
        f"Training accounts  : "
        f"{graph['account'].train_mask.sum().item()}"
    )

    print(
        f"Validation accounts: "
        f"{graph['account'].val_mask.sum().item()}"
    )

    print(
        f"Test accounts      : "
        f"{graph['account'].test_mask.sum().item()}"
    )

    print()
    print(
        f"Saved GNN data: {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading graph and labels...")

    graph, labels = load_data()

    print("Preparing account features...")

    graph = prepare_account_data(
        graph,
        labels
    )

    print("Creating train/validation/test masks...")

    graph = create_masks(graph)

    save_data(graph)


if __name__ == "__main__":
    main()