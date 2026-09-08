"""
Temporal heterogeneous GNN for UPI fraud-ring detection.

The previous version of this file was a plain heterogeneous GraphSAGE: it
accepted a graph carrying `edge_time` but never read it, so a login from three
weeks ago and a login from three seconds ago contributed identically.

This version makes the time dimension real. Every relation uses
`TemporalSAGEConv`, which:

    1. encodes each edge's recency with a Time2Vec-style encoder, and
    2. turns that encoding into a per-edge gate in (0, 1) that scales the
       neighbour's message before aggregation.

Recent events therefore dominate an account's representation, which is what
makes burst typologies (mule farms, smurfing) separable from ordinary
long-running activity.

CPU-only by design - no CUDA-specific ops anywhere.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HeteroConv, MessagePassing


# ============================================================
# TIME ENCODING
# ============================================================

class TimeEncoder(nn.Module):
    """
    Time2Vec-style encoding.

    Component 0 is linear in time, the remaining components are periodic with
    learned frequency and phase. This lets the model express both "how long
    ago" and cyclic patterns such as nightly bursts.
    """

    def __init__(self, dim: int = 16):
        super().__init__()

        if dim < 2:
            raise ValueError("TimeEncoder dim must be at least 2")

        self.dim = dim

        self.frequency = nn.Parameter(torch.empty(dim))
        self.phase = nn.Parameter(torch.zeros(dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Spread initial frequencies over several orders of magnitude so the
        # encoder can resolve both minutes and whole days.
        with torch.no_grad():
            self.frequency.copy_(
                torch.logspace(0.0, 2.0, self.dim)
            )
            self.phase.zero_()

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: [E] edge times, normalised to [0, 1].

        Returns:
            [E, dim] time encoding.
        """

        projected = t.unsqueeze(-1) * self.frequency + self.phase

        return torch.cat(
            [
                projected[..., :1],
                torch.cos(projected[..., 1:]),
            ],
            dim=-1,
        )


# ============================================================
# TEMPORAL CONVOLUTION
# ============================================================

class TemporalSAGEConv(MessagePassing):
    """
    GraphSAGE-style convolution whose messages are gated by edge recency.

        out_i = W_self * x_i + W_neigh * mean_j( gate(dt_ij) * x_j )

    where dt_ij is how long before the most recent edge in the batch the
    interaction happened, and gate is a learned function of its time encoding.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        time_dim: int = 16,
    ):
        super().__init__(aggr="mean")

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_neighbour = nn.Linear(
            in_channels[0],
            out_channels,
            bias=False,
        )

        self.lin_self = nn.Linear(
            in_channels[1],
            out_channels,
            bias=True,
        )

        self.time_encoder = TimeEncoder(time_dim)

        self.time_gate = nn.Linear(time_dim, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_neighbour.reset_parameters()
        self.lin_self.reset_parameters()
        self.time_encoder.reset_parameters()
        self.time_gate.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_time: Optional[Tensor] = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        x_src, x_dst = x

        if edge_time is None:
            # No temporal information for this relation: fall back to an
            # ungated mean, which is exactly plain GraphSAGE.
            gate = torch.ones(
                edge_index.shape[1],
                1,
                device=x_src.device,
                dtype=x_src.dtype,
            )
        else:
            # Recency: 0 for the most recent edge of this relation, larger
            # the further back the event sits.
            recency = edge_time.max() - edge_time

            gate = torch.sigmoid(
                self.time_gate(
                    self.time_encoder(recency)
                )
            )

        aggregated = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            gate=gate,
            size=(x_src.shape[0], x_dst.shape[0]),
        )

        return self.lin_self(x_dst) + self.lin_neighbour(aggregated)

    def message(self, x_j: Tensor, gate: Tensor) -> Tensor:
        return x_j * gate

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.in_channels}, {self.out_channels})"
        )


# ============================================================
# MODEL
# ============================================================

class TemporalHeteroGNN(nn.Module):
    """
    Two-layer temporal heterogeneous GNN producing a fraud logit per account.
    """

    def __init__(
        self,
        metadata,
        in_channels_dict: Dict[str, int],
        hidden_dim: int = 64,
        time_dim: int = 16,
        dropout: float = 0.30,
    ):
        super().__init__()

        node_types, edge_types = metadata

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Project every node type into a shared hidden space.
        self.input_layers = nn.ModuleDict(
            {
                node_type: nn.Linear(
                    in_channels_dict[node_type],
                    hidden_dim,
                )
                for node_type in node_types
            }
        )

        self.conv1 = HeteroConv(
            {
                edge_type: TemporalSAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    time_dim=time_dim,
                )
                for edge_type in edge_types
            },
            aggr="mean",
        )

        self.conv2 = HeteroConv(
            {
                edge_type: TemporalSAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    time_dim=time_dim,
                )
                for edge_type in edge_types
            },
            aggr="mean",
        )

        self.norm1 = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in node_types
            }
        )

        self.norm2 = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in node_types
            }
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    # --------------------------------------------------------

    def encode(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[tuple, Tensor],
        edge_time_dict: Optional[Dict[tuple, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Run the two temporal message-passing layers and return the hidden
        representation of every node type.
        """

        if edge_time_dict is None:
            edge_time_dict = {}

        hidden = {
            node_type: F.relu(self.input_layers[node_type](x))
            for node_type, x in x_dict.items()
        }

        # --- layer 1 ------------------------------------------
        out = self.conv1(
            hidden,
            edge_index_dict,
            edge_time_dict=edge_time_dict,
        )

        hidden = {
            node_type: F.dropout(
                self.norm1[node_type](
                    F.relu(value) + hidden[node_type]
                ),
                p=self.dropout,
                training=self.training,
            )
            for node_type, value in out.items()
        }

        # --- layer 2 ------------------------------------------
        out = self.conv2(
            hidden,
            edge_index_dict,
            edge_time_dict=edge_time_dict,
        )

        hidden = {
            node_type: self.norm2[node_type](
                F.relu(value) + hidden[node_type]
            )
            for node_type, value in out.items()
        }

        return hidden

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[tuple, Tensor],
        edge_time_dict: Optional[Dict[tuple, Tensor]] = None,
    ) -> Tensor:
        """
        Returns:
            [num_accounts] fraud logits. Apply sigmoid for probabilities.
        """

        hidden = self.encode(
            x_dict,
            edge_index_dict,
            edge_time_dict,
        )

        return self.classifier(hidden["account"]).squeeze(-1)


# ============================================================
# HELPERS
# ============================================================

def in_channels_from(data) -> Dict[str, int]:
    """
    Read the per-node-type input width straight off a HeteroData object.
    """

    return {
        node_type: data[node_type].x.shape[1]
        for node_type in data.node_types
    }


def edge_time_dict_from(data) -> Dict[tuple, Tensor]:
    """
    Collect `edge_time` for every relation that carries it.
    """

    return {
        relation: data[relation].edge_time
        for relation in data.edge_types
        if "edge_time" in data[relation]
    }


def build_model(data, **kwargs) -> TemporalHeteroGNN:
    """
    Construct a model matching the shapes of the supplied graph.
    """

    return TemporalHeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels_from(data),
        **kwargs,
    )


if __name__ == "__main__":

    print("=" * 60)
    print("Temporal Heterogeneous Fraud GNN")
    print("=" * 60)

    from pathlib import Path

    graph_file = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "hetero_graph_data.pt"
    )

    graph = torch.load(graph_file, weights_only=False)

    model = build_model(graph)

    print(model)

    print()
    print(
        "Total parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
