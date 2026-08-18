"""
src/fraud/model.py

Graph Attention Network for fraud detection, per the math in
notebooks/math_derivations.ipynb (section 3, GAT attention mechanism).

This file only defines the architecture and is testable standalone with
synthetic graph data -- it does not depend on graph.py or dataset.py.
Wiring it to the real IEEE-CIS transaction graph happens in train.py,
once graph.py exists.

The temporal extension described in the math notebook (section 4) is not
implemented yet -- this is the static GAT first, temporal component comes
once graph.py can produce time-windowed snapshots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class FraudGAT(nn.Module):
    """Multi-layer GAT for binary fraud classification on a transaction graph.

    Architecture: `num_layers` GATConv layers with multi-head attention,
    concatenated on hidden layers and averaged on the final layer (standard
    GAT design from Velickovic et al. 2018), ELU nonlinearity, dropout for
    regularization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert num_layers >= 2, "need at least an input and output GAT layer"

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # input layer: in_channels -> hidden_channels, heads concatenated
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )

        # middle layers: (hidden_channels * heads) -> hidden_channels, concatenated
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )

        # output layer: averaged across heads (concat=False), so output dim
        # is out_channels regardless of head count
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    @torch.no_grad()
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Returns (edge_index, attention_weights) from the first GAT layer --
        the raw material GNNExplainer / regulatory reporting will consume
        later to explain why a given transaction was flagged."""
        self.eval()
        _, (attn_edge_index, attn_weights) = self.convs[0](
            x, edge_index, return_attention_weights=True
        )
        return attn_edge_index, attn_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Synthetic graph: 200 nodes, 32 input features, ~800 directed edges --
    # just enough to prove the architecture runs end-to-end on GPU before
    # we wire it to real data.
    num_nodes, in_channels, num_edges = 200, 32, 800
    x = torch.randn(num_nodes, in_channels, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    model = FraudGAT(in_channels=in_channels, hidden_channels=64, out_channels=2, heads=4).to(device)
    print(model)

    out = model(x, edge_index)
    print(f"\nOutput shape: {out.shape}  (expected: [{num_nodes}, 2])")
    assert out.shape == (num_nodes, 2), "unexpected output shape"

    # Confirm gradients flow -- a fake loss + backward pass
    fake_labels = torch.randint(0, 2, (num_nodes,), device=device)
    loss = F.cross_entropy(out, fake_labels)
    loss.backward()
    print(f"Fake training loss: {loss.item():.4f}")
    print("Backward pass OK -- gradients computed successfully")

    attn_edge_index, attn_weights = model.get_attention_weights(x, edge_index)
    print(f"\nAttention weights shape: {attn_weights.shape}  (per-edge, per-head)")

    print("\nMODEL.PY CHECKS PASSED")