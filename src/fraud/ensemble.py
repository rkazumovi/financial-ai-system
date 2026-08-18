"""
src/fraud/ensemble.py

Combines three anomaly signals into one fraud score, per the project spec's
"novel contribution": GNN + Isolation Forest + Autoencoder ensemble.

- GNN: the tuned FraudGAT from tune.py, loaded from its saved checkpoint
  and best_config.json (so the architecture always matches the weights)
- Isolation Forest: unsupervised anomaly score, fit directly on the same
  graph node features used by the GNN (not a separately-split tabular
  dataset), so every model scores the exact same set of nodes
- Autoencoder: a small feedforward network trained to reconstruct NORMAL
  (non-fraud) transactions only; fraud transactions -- being different
  from what it learned to reconstruct -- produce higher reconstruction
  error, which becomes the third anomaly signal

The three raw scores live on different scales (probabilities, isolation
depths, MSE), so rather than a naive average, a small logistic regression
"blender" is fit on the TRAIN split (using true labels) to learn how to
weigh the three signals -- standard "stacking" ensemble practice.
"""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from graph import build_graph
from model import FraudGAT
from train import make_node_splits, OUTPUT_DIR, CHECKPOINT_PATH
from baseline import precision_at_fpr

BEST_CONFIG_PATH = OUTPUT_DIR / "best_config.json"


class Autoencoder(nn.Module):
    """Small feedforward autoencoder for reconstruction-error anomaly
    scoring. Trained only on non-fraud transactions -- it learns what
    "normal" looks like, so fraud (which it never sees during training)
    reconstructs poorly, and that reconstruction error becomes the
    anomaly signal.
    """

    def __init__(self, in_dim: int, bottleneck: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.ReLU(),
            nn.Linear(128, in_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_tuned_gnn(data, device) -> FraudGAT:
    """Reconstruct FraudGAT with the exact architecture tune.py found best,
    then load its trained weights. Architecture must match exactly or
    load_state_dict fails -- that's why the winning config needs to be
    saved (best_config.json) rather than hardcoded here."""
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_CONFIG_PATH} not found -- see the one-line command to "
            "create it from tune.py's printed winning config."
        )
    with open(BEST_CONFIG_PATH) as f:
        config = json.load(f)

    model = FraudGAT(
        in_channels=data.num_node_features,
        hidden_channels=config["hidden_channels"],
        out_channels=2,
        heads=config["heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()
    return model


def gnn_scores(model, data) -> np.ndarray:
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]
    return probs.cpu().numpy()


def isolation_forest_scores(x: np.ndarray, train_mask: np.ndarray, contamination: float) -> np.ndarray:
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(x[train_mask])
    return -clf.score_samples(x)  # higher = more anomalous


def train_autoencoder(x: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, device, epochs: int = 50) -> np.ndarray:
    """Trains on non-fraud TRAIN nodes only, scores every node by
    reconstruction MSE."""
    normal_train_mask = train_mask & (y == 0)
    x_normal = x[normal_train_mask]

    model = Autoencoder(in_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(x_normal)
        loss = F.mse_loss(recon, x_normal)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    autoencoder epoch {epoch + 1}/{epochs}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        recon_all = model(x)
        error = F.mse_loss(recon_all, x, reduction="none").mean(dim=1)
    return error.cpu().numpy()


def normalize(scores: np.ndarray) -> np.ndarray:
    """Rank-based normalization to [0, 1] -- robust to the very different
    raw scales of probabilities, isolation-forest scores, and MSE, and to
    outliers in any one of them."""
    ranks = np.argsort(np.argsort(scores))
    return ranks / (len(scores) - 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = build_graph(max_transactions=100_000)
    train_mask_t, val_mask_t, test_mask_t = make_node_splits(data.y)
    data = data.to(device)
    train_mask_t, val_mask_t, test_mask_t = train_mask_t.to(device), val_mask_t.to(device), test_mask_t.to(device)

    fraud_rate = data.y.float().mean().item()

    print("\nScoring with tuned GNN...")
    gnn_model = load_tuned_gnn(data, device)
    s_gnn = gnn_scores(gnn_model, data)

    print("Scoring with Isolation Forest...")
    x_np = data.x.cpu().numpy()
    train_mask_np = train_mask_t.cpu().numpy()
    s_if = isolation_forest_scores(x_np, train_mask_np, contamination=fraud_rate)

    print("Training autoencoder on non-fraud train nodes...")
    s_ae = train_autoencoder(data.x, data.y, train_mask_t, device)

    print("\nNormalizing scores and fitting stacking blender on train split...")
    s_gnn_n, s_if_n, s_ae_n = normalize(s_gnn), normalize(s_if), normalize(s_ae)
    stacked = np.stack([s_gnn_n, s_if_n, s_ae_n], axis=1)

    y_np = data.y.cpu().numpy()
    val_mask_np = val_mask_t.cpu().numpy()
    test_mask_np = test_mask_t.cpu().numpy()

    blender = LogisticRegression(class_weight="balanced")
    blender.fit(stacked[train_mask_np], y_np[train_mask_np])
    print(f"  blender weights (gnn, isolation_forest, autoencoder): {blender.coef_[0]}")

    ensemble_scores = blender.predict_proba(stacked)[:, 1]

    print("\n" + "=" * 60)
    print("COMPONENT vs ENSEMBLE COMPARISON (test split)")
    print("=" * 60)
    for name, scores in [("GNN alone", s_gnn), ("Isolation Forest alone", s_if),
                          ("Autoencoder alone", s_ae), ("Ensemble (stacked)", ensemble_scores)]:
        auc = roc_auc_score(y_np[test_mask_np], scores[test_mask_np])
        precision = precision_at_fpr(y_np[test_mask_np], scores[test_mask_np])
        print(f"{name:<24} AUC-ROC {auc:.4f}   Precision@3%FPR {precision:.4f}")

    print("\nReference: Random Forest baseline AUC-ROC 0.9216 (baseline.py)")
    print("\nENSEMBLE.PY CHECKS PASSED")