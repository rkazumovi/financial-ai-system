"""
src/fraud/explainer.py

GNNExplainer-based interpretability for the tuned FraudGAT, per the
project spec's regulatory-compliance requirement: for transactions the
model flags as high-risk, produce a human-readable explanation of WHICH
input features and WHICH connected transactions drove that decision.

Extracts the num_layers-hop subgraph around each explained node BEFORE
running GNNExplainer, instead of masking the entire graph -- a GAT with
num_layers layers can only ever look num_layers hops away for any given
prediction, so explaining over the full 100K-node graph is both wasteful
and (on an 8GB laptop GPU, with this model's size) ran out of memory.
The k-hop subgraph is mathematically equivalent for that node's
explanation and far smaller.
"""

import json

import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from graph import build_graph
from train import make_node_splits
from ensemble import load_tuned_gnn, BEST_CONFIG_PATH


def explain_node(explainer, data, node_idx: int, num_hops: int) -> dict:
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=True
    )
    sub_x = data.x[subset]
    sub_target_idx = int(mapping.item())

    explanation = explainer(sub_x, sub_edge_index, index=sub_target_idx)

    feature_importance = explanation.node_mask[sub_target_idx].detach().cpu().numpy()
    top_feature_idx = feature_importance.argsort()[::-1][:10]
    feature_names = data.feature_names
    top_features = [(feature_names[i], float(feature_importance[i])) for i in top_feature_idx]

    edge_importance = explanation.edge_mask.detach().cpu().numpy()
    sub_edge_np = sub_edge_index.cpu().numpy()
    touching = (sub_edge_np[0] == sub_target_idx) | (sub_edge_np[1] == sub_target_idx)
    touching_idx = touching.nonzero()[0]
    touching_scores = edge_importance[touching_idx]
    order = touching_scores.argsort()[::-1][:5]
    # map local subgraph node ids back to original graph node ids via `subset`
    top_edges = [
        (int(subset[sub_edge_np[0][touching_idx[i]]]), int(subset[sub_edge_np[1][touching_idx[i]]]), float(touching_scores[i]))
        for i in order
    ]

    return {
        "node_idx": node_idx,
        "subgraph_size": int(subset.shape[0]),
        "top_features": top_features,
        "top_connected_transactions": top_edges,
    }


def print_explanation(exp: dict, true_label: int, predicted_prob: float):
    print(f"\nTransaction node {exp['node_idx']}  "
          f"(true label: {'FRAUD' if true_label == 1 else 'not fraud'}, "
          f"predicted fraud probability: {predicted_prob:.4f}, "
          f"explained over a {exp['subgraph_size']}-node local subgraph)")
    print("  Top contributing features:")
    for name, score in exp["top_features"]:
        print(f"    {name:<30} importance {score:.4f}")
    print("  Most influential connected transactions:")
    for src, dst, score in exp["top_connected_transactions"]:
        print(f"    edge ({src} -> {dst})  importance {score:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = build_graph(max_transactions=100_000)
    train_mask, val_mask, test_mask = make_node_splits(data.y)
    data = data.to(device)
    test_mask = test_mask.to(device)

    model = load_tuned_gnn(data, device)

    with open(BEST_CONFIG_PATH) as f:
        num_hops = json.load(f)["num_layers"]

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]

    y = data.y
    correct_fraud_mask = test_mask & (y == 1) & (probs > 0.5)
    candidate_idx = correct_fraud_mask.nonzero(as_tuple=True)[0]
    if len(candidate_idx) == 0:
        raise RuntimeError("No correctly-flagged fraud nodes found on test split to explain")

    top_k = min(3, len(candidate_idx))
    ranked = candidate_idx[probs[candidate_idx].argsort(descending=True)][:top_k]

    print(f"\nExplaining top {top_k} highest-confidence correctly-flagged fraud transactions "
          f"(each over its own {num_hops}-hop local subgraph, not the full 100K-node graph)...")

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    for node_idx in ranked.tolist():
        exp = explain_node(explainer, data, node_idx, num_hops=num_hops)
        print_explanation(exp, true_label=int(y[node_idx].item()), predicted_prob=float(probs[node_idx].item()))

    print("\nEXPLAINER.PY CHECKS PASSED")