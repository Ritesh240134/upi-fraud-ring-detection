"""
SUPERSEDED - kept for reference, not part of the active pipeline.

`models/temporal_hetero_gnn.py` replaces this module. This version hard-codes
its relation list and the node counts of the device / vpa / merchant embedding
tables (120 / 500 / 80), so it silently breaks on any dataset of a different
shape, and it ignores edge times entirely. Nothing imports this file.

--------------------------------------------------------------------------

Heterogeneous GNN for UPI fraud-ring detection.

Architecture:
    Account / Device / VPA / Merchant
                ↓
    Heterogeneous GraphSAGE
                ↓
          64-d hidden layer
                ↓
    Heterogeneous GraphSAGE
                ↓
       Account risk score

Output:
    Probability that an account belongs to a fraud ring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroFraudGNN(nn.Module):
    """
    Two-layer heterogeneous GraphSAGE model.
    """

    def __init__(
        self,
        account_features=3,
        hidden_channels=64,
        dropout=0.30,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # ----------------------------------------------------
        # Account feature projection
        # ----------------------------------------------------

        self.account_encoder = nn.Linear(
            account_features,
            hidden_channels,
        )

        # ----------------------------------------------------
        # Learnable embeddings for other node types
        # ----------------------------------------------------

        self.device_embedding = nn.Embedding(
            120,
            hidden_channels,
        )

        self.vpa_embedding = nn.Embedding(
            500,
            hidden_channels,
        )

        self.merchant_embedding = nn.Embedding(
            80,
            hidden_channels,
        )

        # ----------------------------------------------------
        # First heterogeneous GraphSAGE layer
        # ----------------------------------------------------

        self.conv1 = HeteroConv(
            {
                (
                    "account",
                    "transfers",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "account",
                    "logs_in",
                    "device",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "device",
                    "rev_logs_in",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "account",
                    "adds_beneficiary",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "account",
                    "collects_at",
                    "merchant",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "merchant",
                    "rev_collects_at",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "account",
                    "owns",
                    "vpa",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "vpa",
                    "rev_owns",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),
            },
            aggr="sum",
        )

        # ----------------------------------------------------
        # Second heterogeneous GraphSAGE layer
        # ----------------------------------------------------

        self.conv2 = HeteroConv(
            {
                (
                    "account",
                    "transfers",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "account",
                    "logs_in",
                    "device",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "device",
                    "rev_logs_in",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "account",
                    "adds_beneficiary",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "account",
                    "collects_at",
                    "merchant",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "merchant",
                    "rev_collects_at",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "account",
                    "owns",
                    "vpa",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),

                (
                    "vpa",
                    "rev_owns",
                    "account",
                ): SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                ),
            },
            aggr="sum",
        )

        # ----------------------------------------------------
        # Account fraud-risk head
        # ----------------------------------------------------

        self.risk_head = nn.Sequential(
            nn.Linear(
                hidden_channels,
                32,
            ),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                32,
                1,
            ),
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x_dict,
        edge_index_dict,
    ):

        # ----------------------------------------------------
        # Encode node features
        # ----------------------------------------------------

        account_x = self.account_encoder(
            x_dict["account"]
        )

        device_count = (
            self.device_embedding.num_embeddings
        )

        vpa_count = (
            self.vpa_embedding.num_embeddings
        )

        merchant_count = (
            self.merchant_embedding.num_embeddings
        )

        device_ids = torch.arange(
            device_count,
            device=self.device_embedding.weight.device,
        )

        vpa_ids = torch.arange(
            vpa_count,
            device=self.vpa_embedding.weight.device,
        )

        merchant_ids = torch.arange(
            merchant_count,
            device=self.merchant_embedding.weight.device,
        )

        x = {
            "account": account_x,
            "device": self.device_embedding(
                device_ids
            ),
            "vpa": self.vpa_embedding(
                vpa_ids
            ),
            "merchant": self.merchant_embedding(
                merchant_ids
            ),
        }

        # ----------------------------------------------------
        # Add reverse relations
        # ----------------------------------------------------

        edges = dict(edge_index_dict)

        if (
            "account",
            "logs_in",
            "device",
        ) in edge_index_dict:

            edges[
                "device",
                "rev_logs_in",
                "account",
            ] = edge_index_dict[
                "account",
                "logs_in",
                "device",
            ].flip(0)

        if (
            "account",
            "collects_at",
            "merchant",
        ) in edge_index_dict:

            edges[
                "merchant",
                "rev_collects_at",
                "account",
            ] = edge_index_dict[
                "account",
                "collects_at",
                "merchant",
            ].flip(0)

        if (
            "account",
            "owns",
            "vpa",
        ) in edge_index_dict:

            edges[
                "vpa",
                "rev_owns",
                "account",
            ] = edge_index_dict[
                "account",
                "owns",
                "vpa",
            ].flip(0)

        # ----------------------------------------------------
        # First GNN layer
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edges,
        )

        x = {
            node_type: F.relu(
                features
            )
            for node_type, features
            in x.items()
        }

        x = {
            node_type: F.dropout(
                features,
                p=self.dropout,
                training=self.training,
            )
            for node_type, features
            in x.items()
        }

        # ----------------------------------------------------
        # Second GNN layer
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edges,
        )

        x = {
            node_type: F.relu(
                features
            )
            for node_type, features
            in x.items()
        }

        # ----------------------------------------------------
        # Account fraud probability
        # ----------------------------------------------------

        logits = self.risk_head(
            x["account"]
        ).squeeze(-1)

        return logits


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Heterogeneous Fraud GNN")
    print("=" * 60)

    model = HeteroFraudGNN()

    print(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"Total parameters: {total_parameters:,}"
    )