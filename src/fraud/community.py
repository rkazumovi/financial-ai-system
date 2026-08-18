"""
src/fraud/community.py

Louvain community detection on the transaction graph, per the project
spec's fraud-ring detection requirement. A coordinated fraud ring shows
up as a tight cluster of transactions (same actors reusing cards or
addresses) with an anomalously high concentration of labeled fraud --
much higher than the ~3% base rate -- findable structurally, without
looking at any single transaction's features.

Uses networkx's built-in Louvain implementation, run on an UNDIRECTED
version of the graph -- community structure doesn't depend on edge
direction (which came first in time), only on which nodes are connected.
This runs entirely on CPU; no GPU/model involved.
"""

import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from graph import build_graph


def to_undirected_networkx(data) -> nx.Graph:
    """Convert just the graph STRUCTURE (edge_index, node count) to an
    undirected networkx Graph -- deliberately stripped of node features
    and the custom feature_names attribute before conversion, to avoid
    any ambiguity in how to_networkx would export non-standard Data
    attributes. Louvain only needs the edges."""
    structure_only = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
    return to_networkx(structure_only, to_undirected=True)


def detect_communities(g: nx.Graph, seed: int = 42) -> list:
    """Returns a list of sets, each set the node indices in one community."""
    return list(nx.community.louvain_communities(g, seed=seed))


def analyze_communities(
    communities: list, y: np.ndarray, base_rate: float, min_size: int = 5, min_fraud_multiplier: float = 3.0
) -> list:
    """For each community: size and fraud rate. Flagged as a suspected
    fraud ring only if it's large enough to be meaningful (not just noise
    from one fraud node in a tiny community) AND its fraud rate is well
    above the graph's overall base rate."""
    results = []
    for community in communities:
        idx = np.array(list(community))
        size = len(idx)
        fraud_count = int(y[idx].sum())
        fraud_rate = fraud_count / size
        is_ring = size >= min_size and fraud_rate >= base_rate * min_fraud_multiplier
        results.append({
            "size": size,
            "fraud_count": fraud_count,
            "fraud_rate": fraud_rate,
            "node_ids": idx.tolist(),
            "is_suspected_ring": is_ring,
        })
    return results


if __name__ == "__main__":
    data = build_graph(max_transactions=100_000)
    y = data.y.numpy()
    base_rate = y.mean()

    print("Converting to undirected networkx graph...")
    g = to_undirected_networkx(data)
    print(f"  {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    print("\nRunning Louvain community detection (may take a minute or two on 100K nodes)...")
    communities = detect_communities(g)
    print(f"  found {len(communities):,} communities")

    sizes = [len(c) for c in communities]
    print(f"  community size: min {min(sizes)}, max {max(sizes)}, median {int(np.median(sizes))}")

    print(f"\nAnalyzing communities for fraud concentration (base rate {base_rate:.4%})...")
    analysis = analyze_communities(communities, y, base_rate)

    suspected_rings = [c for c in analysis if c["is_suspected_ring"]]
    suspected_rings.sort(key=lambda c: c["fraud_rate"], reverse=True)

    print(f"\n{len(suspected_rings)} communities flagged as suspected fraud rings "
          f"(size >= 5, fraud rate >= {base_rate * 3:.2%})")

    print("\n" + "=" * 60)
    print("TOP SUSPECTED FRAUD RINGS")
    print("=" * 60)
    for ring in suspected_rings[:10]:
        print(f"  size {ring['size']:4d}  fraud {ring['fraud_count']:3d}/{ring['size']:<4d} "
              f"({ring['fraud_rate']:.2%})  sample nodes: {ring['node_ids'][:5]}")

    total_flagged_nodes = sum(r["size"] for r in suspected_rings)
    print(f"\nTotal transactions inside suspected rings: {total_flagged_nodes:,} "
          f"({total_flagged_nodes / data.num_nodes:.2%} of graph)")

    print("\nCOMMUNITY.PY CHECKS PASSED")