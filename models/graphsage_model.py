
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
class GraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        dropout=0.2
    ):
        super().__init__()
        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )
        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )
        self.classifier = nn.Linear(
            hidden_dim,
            1
        )
        self.dropout = dropout
    def forward(
        self,
        x,
        edge_index
    ):
        x = self.conv1(
            x,
            edge_index
        )
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )
        x = self.conv2(
            x,
            edge_index
        )
        x = F.relu(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)
