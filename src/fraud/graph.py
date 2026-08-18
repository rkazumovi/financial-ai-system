"""
src/fraud/graph.py

Builds a transaction graph from the IEEE-CIS tabular data for GAT input.

Design note: IEEE-CIS has no persistent "account ID" -- card1/card2 (and
to a lesser extent addr1) are the closest available proxies for a
recurring actor. Rather than treating those values as a separate node
type, this builds a HOMOGENEOUS graph where nodes = transactions and
edges connect transactions from the same actor that are temporally
close, within a sliding window (not just the immediate next transaction).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.preprocessing import StandardScaler

from dataset import (
    load_raw,
    get_numeric_feature_columns,
    get_categorical_feature_columns,
    encode_categorical_features,
    TARGET_COL,
)
from model import FraudGAT


def build_windowed_edges(df: pd.DataFrame, key_col: str, window: int = 3) -> np.ndarray:
    """For each group of transactions sharing `key_col`, sort by
    TransactionDT and connect each transaction to the next `window`
    transactions in the same group -- a generalization of a simple chain
    (window=1). O(n * window) edges, far short of an O(k^2) full clique.
    """
    edges = []
    for _, group in df.groupby(key_col, sort=False):  # NaN keys auto-dropped
        if len(group) < 2:
            continue
        ordered = group.sort_values("TransactionDT")
        idx = ordered.index.to_numpy()  # == row positions, since df has a RangeIndex
        n = len(idx)
        max_offset = min(window, n - 1)
        for offset in range(1, max_offset + 1):
            edges.append(np.stack([idx[:-offset], idx[offset:]]))

    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.concatenate(edges, axis=1).astype(np.int64)


def build_graph(
    max_transactions: int = None,
    key_cols: tuple = ("card1", "addr1"),
    window: int = 2,
) -> Data:
    """Build a PyG Data object from the IEEE-CIS transactions.

    max_transactions: if set, uses only the first N transactions by time.
    key_cols: identifier columns used to build "same actor" edges.
    window: how many nearby-in-time transactions per key each node links to.

    Returns a Data object with an extra `feature_names` attribute (a plain
    list, same length/order as the columns of x) so downstream code like
    explainer.py can report feature importance by name, not column index.
    """
    df = load_raw()
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    if max_transactions is not None:
        df = df.iloc[:max_transactions].reset_index(drop=True)

    print(f"Building graph on {len(df):,} transactions...")

    numeric_cols = get_numeric_feature_columns(df)
    numeric_features = df[numeric_cols].fillna(df[numeric_cols].median())
    numeric_scaled = StandardScaler().fit_transform(numeric_features.values)

    categorical_cols = get_categorical_feature_columns(df)
    categorical_features = encode_categorical_features(df, categorical_cols)
    print(f"  features: {len(numeric_cols)} numeric + {categorical_features.shape[1]} encoded categorical")

    x_combined = np.concatenate([numeric_scaled, categorical_features.values], axis=1)
    x = torch.tensor(x_combined, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COL].values, dtype=torch.long)
    feature_names = list(numeric_cols) + list(categorical_features.columns)

    edge_parts = []
    for key_col in key_cols:
        if key_col not in df.columns:
            continue
        edges = build_windowed_edges(df, key_col=key_col, window=window)
        print(f"  {key_col}: {edges.shape[1]:,} edges")
        edge_parts.append(edges)

    edge_index = np.concatenate(edge_parts, axis=1) if edge_parts else np.empty((2, 0), dtype=np.int64)
    if edge_index.shape[1] > 0:
        edge_index = np.unique(edge_index, axis=1)  # dedupe pairs shared by multiple key_cols

    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index_t, y=y)
    data.feature_names = feature_names
    return data


if __name__ == "__main__":
    data = build_graph(max_transactions=50_000)

    print("\n" + "=" * 50)
    print("GRAPH SUMMARY")
    print("=" * 50)
    print(data)
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")
    print(f"Avg degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Fraud rate among nodes: {data.y.float().mean():.4%}")
    print(f"Feature names tracked: {len(data.feature_names)}")

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes) + degree(data.edge_index[1], num_nodes=data.num_nodes)
    isolated = int((deg == 0).sum().item())
    print(f"Isolated nodes (no edges): {isolated:,} ({isolated / data.num_nodes:.2%})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = FraudGAT(in_channels=data.num_node_features, hidden_channels=64, out_channels=2, heads=4).to(device)
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y)
    print(f"\nForward pass on real graph OK -- output shape {out.shape}, loss {loss.item():.4f}")

    print("\nGRAPH.PY CHECKS PASSED")