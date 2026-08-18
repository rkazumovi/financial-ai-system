"""
Reusable graph-construction and community-detection backbone, generalized
from src/fraud/graph.py, src/fraud/community.py, and the k-hop subgraph
extraction pattern from src/fraud/explainer.py. Not fraud-specific: any
component building a relationship graph from tabular data (e.g. a future
correlation-network view of the quant universe) can reuse this directly.
"""
import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx


def build_windowed_edges(df, key_col, window=3, id_col="row_id"):
    """
    Connects rows that share the same value in key_col and are within
    `window` positions of each other after sorting by that shared key --
    e.g. transactions sharing a card number within a few positions of each
    other in time, or (more generally) any entity-linkage-by-proximity
    pattern. Returns a list of (i, j) row-index edge tuples.
    """
    edges = []
    for _, group in df.groupby(key_col):
        indices = group[id_col].tolist()
        for i in range(len(indices)):
            for j in range(i + 1, min(i + 1 + window, len(indices))):
                edges.append((indices[i], indices[j]))
    return edges


def build_graph_from_edges(num_nodes, edges, x=None, y=None, feature_names=None):
    """Assembles a PyG Data object from an edge list, making the graph
    undirected (both directions present) since most GNN layers assume
    this unless explicitly told otherwise."""
    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edges_arr = np.array(edges, dtype=np.int64).T  # (2, n_edges)
        edge_index = np.concatenate([edges_arr, edges_arr[[1, 0]]], axis=1)  # undirected

    import torch
    data = Data(
        x=x, y=y,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=num_nodes,
    )
    if feature_names is not None:
        data.feature_names = feature_names
    return data


def extract_k_hop_subgraph(node_idx, edge_index, num_hops, num_nodes):
    """
    Thin wrapper around torch_geometric.utils.k_hop_subgraph, extracting
    only the num_hops-neighborhood of a single node BEFORE running anything
    expensive (e.g. GNNExplainer) on it. This is the fix that took
    explainer.py from a full-graph CUDA OOM crash to sub-second runtime: a
    k-layer GNN's prediction for one node only ever depends on that node's
    k-hop neighborhood, so there's no need to keep the rest of the graph
    in the computation at all.
    """
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes,
    )
    return {
        "subset": subset,          # original node indices included in the subgraph
        "edge_index": sub_edge_index,
        "mapping": mapping,        # index of the original node_idx within `subset`
        "edge_mask": edge_mask,
    }


def to_undirected_networkx(edge_index, num_nodes):
    """Strips a PyG graph down to bare (edge_index, num_nodes) before
    exporting to networkx -- avoids attribute-export issues when the Data
    object carries extra fields like feature_names that networkx doesn't
    know how to serialize."""
    stripped = Data(edge_index=edge_index, num_nodes=num_nodes)
    return to_networkx(stripped, to_undirected=True)


def detect_communities(g, seed=42):
    """Louvain community detection -- groups of nodes more densely
    connected to each other than to the rest of the graph."""
    return nx.community.louvain_communities(g, seed=seed)


def analyze_communities(communities, y, base_rate, min_size=5, min_fraud_multiplier=3.0, id_to_idx=None):
    """
    Flags communities whose positive-label (e.g. fraud) rate is
    min_fraud_multiplier times the overall base rate, restricted to
    communities of at least min_size nodes (small communities give noisy
    rate estimates and shouldn't be flagged on a handful of nodes).

    y: array-like of labels indexed the same way node ids in `communities`
       are indexed (via id_to_idx if node ids and array positions differ).
    """
    y = np.asarray(y)
    flagged = []
    for community in communities:
        if len(community) < min_size:
            continue
        idxs = [id_to_idx[n] if id_to_idx is not None else n for n in community]
        idxs = [i for i in idxs if 0 <= i < len(y)]
        if not idxs:
            continue
        rate = float(y[idxs].mean())
        if rate >= min_fraud_multiplier * base_rate:
            flagged.append({"size": len(community), "positive_rate": rate,
                             "multiplier": rate / base_rate if base_rate > 0 else float("inf"),
                             "node_ids": list(community)})
    flagged.sort(key=lambda c: c["positive_rate"], reverse=True)
    return flagged


if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("SELF-TEST: graph_utils.py on a synthetic planted-community graph")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_clean, n_ring = 300, 20
    num_nodes = n_clean + n_ring

    edges = []
    for _ in range(n_clean * 2):
        i, j = rng.integers(0, n_clean, size=2)
        if i != j:
            edges.append((int(i), int(j)))

    ring_ids = list(range(n_clean, n_clean + n_ring))
    for i in ring_ids:
        for j in rng.choice(ring_ids, size=6, replace=False):
            if i != j:
                edges.append((i, int(j)))

    y = np.zeros(num_nodes)
    y[n_clean:] = (rng.random(n_ring) < 0.7).astype(float)
    base_rate = float(y.mean())

    edges_arr = np.array(edges, dtype=np.int64).T
    edge_index_full = np.concatenate([edges_arr, edges_arr[[1, 0]]], axis=1)
    edge_index = torch.tensor(edge_index_full, dtype=torch.long)

    g = to_undirected_networkx(edge_index, num_nodes)
    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    communities = detect_communities(g)
    print(f"found {len(communities)} communities")

    flagged = analyze_communities(communities, y, base_rate, min_size=5, min_fraud_multiplier=2.0)
    print(f"base rate: {base_rate:.4f}")
    print(f"flagged {len(flagged)} communities:")
    for c in flagged[:5]:
        print(f"  size={c['size']}, positive_rate={c['positive_rate']:.4f}, multiplier={c['multiplier']:.2f}x")

    ring_found = any(set(range(n_clean, num_nodes)).issubset(set(c["node_ids"])) or
                      len(set(c["node_ids"]) & set(range(n_clean, num_nodes))) >= n_ring * 0.7
                      for c in flagged)
    print(f"\nplanted ring recovered as a flagged community: {ring_found}")

    print("\n" + "=" * 60)
    print("SELF-TEST: k-hop subgraph extraction")
    print("=" * 60)
    sub = extract_k_hop_subgraph(node_idx=n_clean, edge_index=edge_index, num_hops=2, num_nodes=num_nodes)
    print(f"2-hop subgraph around node {n_clean}: {sub['subset'].shape[0]} nodes, "
          f"{sub['edge_index'].shape[1]} edges (vs {num_nodes} nodes in full graph)")
    subgraph_smaller = sub["subset"].shape[0] < num_nodes
    print(f"subgraph strictly smaller than full graph: {subgraph_smaller}")

    print("\n" + "=" * 60)
    print("SELF-TEST: build_windowed_edges")
    print("=" * 60)
    df = pd.DataFrame({
        "row_id": range(10),
        "key": ["A", "A", "A", "B", "B", "A", "A", "B", "B", "B"],
    })
    we_edges = build_windowed_edges(df, key_col="key", window=2, id_col="row_id")
    print(f"windowed edges: {we_edges}")
    expected_pairs_exist = (0, 1) in we_edges and (0, 2) in we_edges and (0, 5) not in we_edges
    print(f"expected adjacency pattern (window=2, no cross-window edges): {expected_pairs_exist}")

    all_passed = ring_found and subgraph_smaller and expected_pairs_exist
    print("\nRESULT:", "ALL CHECKS PASSED" if all_passed else "one or more checks FAILED")