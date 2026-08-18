"""
Reusable anomaly-scoring backbone, generalized from the scoring/ensembling
logic originally built in src/fraud/ensemble.py and src/fraud/baseline.py.
Not fraud-specific: any component needing "score how anomalous is this row"
(e.g. a future quant regime-detection module) can use this directly instead
of re-deriving it.

Three building blocks:
  1. Individual anomaly scorers (Isolation Forest, a simple Autoencoder).
  2. rank_normalize -- puts heterogeneous scorers on a common [0,1] scale.
  3. StackedEnsemble -- a Logistic Regression blender combining several
     rank-normalized scores into one, with a precision_at_fpr evaluation
     metric appropriate for heavily imbalanced anomaly-detection tasks
     (where plain accuracy is meaningless).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression


def rank_normalize(scores):
    """Converts raw scores (arbitrary scale/direction) to [0,1] via rank,
    so heterogeneous scorers (e.g. Isolation Forest's path-length score and
    an autoencoder's reconstruction error) can be combined on equal footing."""
    scores = np.asarray(scores)
    ranks = scores.argsort().argsort()
    return ranks / max(len(scores) - 1, 1)


def precision_at_fpr(y_true, scores, fpr_target=0.03):
    """Precision achieved at the score threshold that yields a target false
    positive rate -- more meaningful than plain precision/recall for
    heavily imbalanced anomaly-detection tasks, since it answers the
    operational question 'if we can only tolerate flagging fpr_target of
    the negatives, how precise are the positives we do flag'."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    negatives = scores[y_true == 0]
    if len(negatives) == 0:
        return float("nan")
    threshold = np.percentile(negatives, 100 * (1 - fpr_target))
    flagged = scores >= threshold
    if flagged.sum() == 0:
        return 0.0
    return float(y_true[flagged].mean())


def run_isolation_forest(X, contamination="auto", random_state=42, n_jobs=1):
    """Fits Isolation Forest and returns rank-normalized anomaly scores
    (higher = more anomalous). n_jobs=1 by default -- nested parallelism
    with an outer parallel process (e.g. GridSearchCV) has previously
    caused memory crashes in this project; only raise it in a standalone
    context."""
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=n_jobs)
    model.fit(X)
    raw_scores = -model.score_samples(X)  # higher = more anomalous
    return rank_normalize(raw_scores), model


class Autoencoder(nn.Module):
    """Simple bottleneck autoencoder; reconstruction error on held-out rows
    is used as an anomaly score (rows the network struggles to reconstruct
    are the ones least like the "normal" data it was trained on)."""
    def __init__(self, input_dim, bottleneck_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_normal, input_dim, epochs=50, batch_size=256, lr=1e-3, device="cpu"):
    """Trains on X_normal ONLY -- the anomaly-detection premise is that the
    network learns to reconstruct 'normal' rows well and 'anomalous' rows
    poorly precisely because it never saw anomalous patterns during
    training. Training on the full (mixed) dataset would defeat this."""
    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_tensor = torch.tensor(X_normal, dtype=torch.float32, device=device)
    n = len(X_tensor)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch = X_tensor[perm[i:i + batch_size]]
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
    return model


def autoencoder_scores(model, X, device="cpu"):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
    return rank_normalize(errors)


class StackedEnsemble:
    """Combines several rank-normalized anomaly scores into one via a
    Logistic Regression blender, learning how much to trust each scorer
    rather than averaging them blindly."""
    def __init__(self, class_weight="balanced"):
        self.model = LogisticRegression(class_weight=class_weight)
        self.score_names = None

    def fit(self, score_dict, y):
        self.score_names = list(score_dict.keys())
        X = np.column_stack([score_dict[name] for name in self.score_names])
        self.model.fit(X, y)
        return self

    def predict_scores(self, score_dict):
        X = np.column_stack([score_dict[name] for name in self.score_names])
        return self.model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    print("=" * 60)
    print("SELF-TEST: anomaly_core.py on synthetic data")
    print("=" * 60)
    rng = np.random.default_rng(42)
    n_normal, n_anomaly, n_features = 2000, 100, 10
    X_normal = rng.normal(0, 1, size=(n_normal, n_features))
    X_anomaly = rng.normal(4, 1, size=(n_anomaly, n_features))  # shifted -- should be "anomalous"
    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    shuffle = rng.permutation(len(X))
    X, y = X[shuffle], y[shuffle]
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("\nIsolation Forest:")
    if_scores_train, if_model = run_isolation_forest(X_train)
    if_scores_test = rank_normalize(-if_model.score_samples(X_test))
    if_p = precision_at_fpr(y_test, if_scores_test, fpr_target=0.05)
    print(f"  precision@5%FPR: {if_p:.4f}")

    print("\nAutoencoder (trained on normal-labeled train rows only):")
    ae_model = train_autoencoder(X_train[y_train == 0], input_dim=n_features, epochs=30)
    ae_scores_test = autoencoder_scores(ae_model, X_test)
    ae_p = precision_at_fpr(y_test, ae_scores_test, fpr_target=0.05)
    print(f"  precision@5%FPR: {ae_p:.4f}")

    print("\nStacked ensemble:")
    if_scores_train_full = rank_normalize(-if_model.score_samples(X_train))
    ae_scores_train_full = autoencoder_scores(ae_model, X_train)
    ensemble = StackedEnsemble().fit(
        {"iso_forest": if_scores_train_full, "autoencoder": ae_scores_train_full}, y_train)
    ensemble_scores_test = ensemble.predict_scores({"iso_forest": if_scores_test, "autoencoder": ae_scores_test})
    ens_p = precision_at_fpr(y_test, ensemble_scores_test, fpr_target=0.05)
    print(f"  precision@5%FPR: {ens_p:.4f}")

    checks_passed = if_p > 0.3 and ae_p > 0.3 and ens_p >= min(if_p, ae_p) - 0.05
    print(f"\nRESULT: {'PASSED' if checks_passed else 'FAILED'} "
          f"(scorers found the shifted anomaly cluster with reasonable precision)")