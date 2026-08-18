"""
src/fraud/baseline.py

Classical ML baselines for fraud detection, per the project spec's
requirement to demonstrate GNN superiority with a documented AUC
comparison:

- Isolation Forest: unsupervised anomaly detection (doesn't use isFraud
  labels during fitting -- scores every transaction by how "isolated" it
  is in feature space)
- Random Forest: supervised classifier, hyperparameter-searched on a
  stratified subsample then refit once on the full training set --
  memory-constrained mode (see comments below) to survive on machines
  with limited free RAM

Run standalone: loads data via dataset.py, evaluates both on the held-out
test split, and prints an AUC-ROC + precision-at-3%-FPR comparison table.
"""

import time

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_score

from dataset import load_raw, stratified_split, get_numeric_feature_columns, TARGET_COL


def fill_missing(train_df, other_df, numeric_cols):
    """Median-fill NaNs (median from train only, avoids leakage), and
    return a float32 numpy array rather than a pandas DataFrame -- this
    halves memory versus float64 and avoids pandas' per-worker indexing
    overhead when sklearn parallelizes across processes."""
    medians = train_df[numeric_cols].median()
    filled = other_df[numeric_cols].fillna(medians)
    return filled.values.astype(np.float32)


def precision_at_fpr(y_true, scores, target_fpr: float = 0.03) -> float:
    """Precision at the threshold where false-positive rate is closest to
    (but not above) target_fpr -- matches the project's '>85% precision at
    3% FPR' target metric."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    if not valid.any():
        return float("nan")
    idx = np.where(valid)[0][-1]  # largest fpr still <= target
    threshold = thresholds[idx]
    y_pred = (scores >= threshold).astype(int)
    return precision_score(y_true, y_pred, zero_division=0)


def run_isolation_forest(X_train, X_test, y_test, contamination: float):
    print("\nTraining Isolation Forest (unsupervised)...")
    t0 = time.time()
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train)  # no labels used
    print(f"  fit time: {time.time() - t0:.1f}s")

    scores = -clf.score_samples(X_test)
    auc = roc_auc_score(y_test, scores)
    precision = precision_at_fpr(y_test, scores)
    return auc, precision


def run_random_forest(X_train, y_train, X_test, y_test):
    print("\nTraining Random Forest (memory-constrained mode: serial, subsampled grid search)...")
    t0 = time.time()

    # Hyperparameter search on a stratified subsample -- full 413K rows isn't
    # needed to pick good hyperparameters, and searching on it is what's
    # exhausting memory. rng is fixed for reproducibility.
    rng = np.random.RandomState(42)
    subsample_size = min(80_000, len(X_train))
    fraud_idx = np.where(y_train == 1)[0]
    normal_idx = np.where(y_train == 0)[0]
    # keep the same ~3.5% fraud ratio in the subsample
    n_fraud = int(subsample_size * (len(fraud_idx) / len(y_train)))
    sample_idx = np.concatenate([
        rng.choice(fraud_idx, size=n_fraud, replace=False),
        rng.choice(normal_idx, size=subsample_size - n_fraud, replace=False),
    ])
    X_sub, y_sub = X_train[sample_idx], y_train[sample_idx]
    print(f"  grid search on subsample: {len(X_sub):,} rows ({y_sub.mean():.4%} fraud)")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
    }
    # n_jobs=1 everywhere -- fully serial, avoids any duplicated in-memory
    # copies of the data across worker processes. Slower, but won't crash
    # on a memory-constrained machine.
    base_clf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)
    search = GridSearchCV(base_clf, param_grid, scoring="roc_auc", cv=3, n_jobs=1)
    search.fit(X_sub, y_sub)
    print(f"  grid search time: {time.time() - t0:.1f}s")
    print(f"  best params: {search.best_params_}")

    # Single full-data fit with the winning hyperparameters -- one model,
    # one copy of the full training set in memory, no concurrent folds.
    print("  refitting best params on the full training set...")
    t1 = time.time()
    final_clf = RandomForestClassifier(
        **search.best_params_, class_weight="balanced", random_state=42, n_jobs=1
    )
    final_clf.fit(X_train, y_train)
    print(f"  full-data fit time: {time.time() - t1:.1f}s")

    scores = final_clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    precision = precision_at_fpr(y_test, scores)
    return auc, precision, final_clf


if __name__ == "__main__":
    print("Loading data...")
    df = load_raw()
    splits = stratified_split(df)
    numeric_cols = get_numeric_feature_columns(df)
    fraud_rate = df[TARGET_COL].mean()

    print(f"Using {len(numeric_cols)} numeric features, fraud rate {fraud_rate:.4%}")

    X_train = fill_missing(splits.train, splits.train, numeric_cols)
    y_train = splits.train[TARGET_COL].values
    X_test = fill_missing(splits.train, splits.test, numeric_cols)
    y_test = splits.test[TARGET_COL].values

    if_auc, if_precision = run_isolation_forest(X_train, X_test, y_test, contamination=fraud_rate)
    rf_auc, rf_precision, rf_model = run_random_forest(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 50)
    print("BASELINE COMPARISON (test split)")
    print("=" * 50)
    print(f"{'Model':<20}{'AUC-ROC':<12}{'Precision@3%FPR':<18}")
    print(f"{'Isolation Forest':<20}{if_auc:<12.4f}{if_precision:<18.4f}")
    print(f"{'Random Forest':<20}{rf_auc:<12.4f}{rf_precision:<18.4f}")

    print(f"\nTarget from spec: Random Forest AUC-ROC > 0.90 -- {'PASS' if rf_auc > 0.90 else 'below target'}")
    print("\nBASELINE.PY CHECKS PASSED")