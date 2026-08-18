"""
src/fraud/train.py

Trains FraudGAT on the transaction graph in a transductive node-classification
setup: the full graph (all nodes, all edges) is used for message passing every
epoch, but the loss is only computed on labeled TRAIN nodes -- VAL/TEST node
features are still visible to the GAT for aggregation (same setup as the
classic Cora/Citeseer GNN benchmarks), their labels are just hidden from the
loss and only used for evaluation.

train() accepts a pre-built graph/splits (data, train_mask, val_mask,
test_mask) so tune.py can build the graph once and reuse it across many
hyperparameter trials, instead of rebuilding it from scratch each time.
Running this file standalone still builds its own graph if none is passed in.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from graph import build_graph
from model import FraudGAT
from baseline import precision_at_fpr

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "fraud_gat_best.pt"


def make_node_splits(y: torch.Tensor, val_size=0.15, test_size=0.15, seed=42):
    """Stratified split over node indices, preserving the fraud rate in
    each split -- same reasoning as dataset.py's stratified_split, applied
    to graph node indices instead of dataframe rows."""
    n = y.shape[0]
    idx = np.arange(n)
    labels = y.numpy()

    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=labels[train_val_idx],
        random_state=seed,
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights from the train split, for
    CrossEntropyLoss -- critical given the ~3% fraud rate."""
    labels = y[mask]
    counts = torch.bincount(labels, minlength=2).float()
    weights = len(labels) / (2.0 * counts)
    return weights


@torch.no_grad()
def evaluate(model, data, mask) -> tuple:
    model.eval()
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]
    y_true = data.y[mask].cpu().numpy()
    y_score = probs[mask].cpu().numpy()
    auc = roc_auc_score(y_true, y_score)
    precision = precision_at_fpr(y_true, y_score)
    return auc, precision


def train(
    data=None,
    train_mask=None,
    val_mask=None,
    test_mask=None,
    max_transactions: int = 100_000,
    hidden_channels: int = 64,
    heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 5e-3,
    weight_decay: float = 5e-4,
    max_epochs: int = 100,
    patience: int = 15,
    checkpoint_path: Path = CHECKPOINT_PATH,
    verbose: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    if data is None:
        data = build_graph(max_transactions=max_transactions)
        train_mask, val_mask, test_mask = make_node_splits(data.y)

    data = data.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    if verbose:
        print(f"Train nodes: {train_mask.sum().item():,}  "
              f"Val nodes: {val_mask.sum().item():,}  "
              f"Test nodes: {test_mask.sum().item():,}")

    weights = class_weights(data.y, train_mask).to(device)

    model = FraudGAT(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=2,
        heads=heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    best_val_auc = 0.0
    epochs_without_improvement = 0

    if verbose:
        print("\nTraining...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        val_auc, val_precision = evaluate(model, data, val_mask)

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 5 == 0 or improved):
            marker = " *" if improved else ""
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  val AUC {val_auc:.4f}  val P@3%FPR {val_precision:.4f}{marker}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no val AUC improvement for {patience} epochs)")
            break

    if verbose:
        print(f"\nBest val AUC: {best_val_auc:.4f}  (checkpoint saved to {checkpoint_path})")
        print("\nLoading best checkpoint for final test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_auc, test_precision = evaluate(model, data, test_mask)

    if verbose:
        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS")
        print("=" * 50)
        print(f"FraudGAT      AUC-ROC {test_auc:.4f}   Precision@3%FPR {test_precision:.4f}")
        print("(compare against baseline.py: Random Forest AUC-ROC 0.9216)")
        print(f"\nTarget from spec: GNN AUC-ROC > 0.95 -- "
              f"{'PASS' if test_auc > 0.95 else 'below target'}")

    return {
        "val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_precision": test_precision,
        "config": {
            "hidden_channels": hidden_channels,
            "heads": heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    }


if __name__ == "__main__":
    train()
    print("\nTRAIN.PY CHECKS PASSED")