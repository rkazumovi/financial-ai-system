"""
src/fraud/dataset.py

Loads and prepares the IEEE-CIS Fraud Detection dataset:
- merges train_transaction.csv with train_identity.csv
- reports class imbalance (~3.5% fraud) and missing-value structure
- provides a stratified train/val/test split so the imbalance is preserved
  in every split (a plain random split can starve val/test of positive
  examples given how rare fraud is)
- provides a scikit-learn Pipeline (StandardScaler + PCA) for numeric
  transaction features, used by the classical baselines later
- provides categorical feature encoding (one-hot for low-cardinality
  columns, frequency encoding for high-cardinality ones) for graph.py

Graph construction (accounts as nodes, transactions as edges) lives in
graph.py, not here -- this module only owns the raw tabular data.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ieee-cis"
TARGET_COL = "isFraud"
ID_COL = "TransactionID"


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and merge train_transaction.csv + train_identity.csv on TransactionID.

    Identity data (device/browser fingerprint) is missing for most
    transactions -- a left join keeps every transaction and leaves identity
    columns as NaN where absent. That absence is itself a signal worth
    keeping, not a reason to drop the row.
    """
    tx_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found -- did the Kaggle download from Step 2 complete?"
        )

    transactions = pd.read_csv(tx_path)
    identity = pd.read_csv(id_path)
    df = transactions.merge(identity, on=ID_COL, how="left")
    return df


def basic_report(df: pd.DataFrame) -> dict:
    """Print and return a summary: shape, fraud rate, missingness, dtypes."""
    n_rows, n_cols = df.shape
    fraud_rate = df[TARGET_COL].mean()
    missing_frac = df.isna().mean().sort_values(ascending=False)
    n_numeric = df.select_dtypes(include=[np.number]).shape[1]
    n_categorical = df.select_dtypes(exclude=[np.number]).shape[1]

    print(f"Rows: {n_rows:,}  Columns: {n_cols}")
    print(f"Fraud rate: {fraud_rate:.4%}  ({int(df[TARGET_COL].sum()):,} fraud / {n_rows:,} total)")
    print(f"Numeric columns: {n_numeric}  Categorical columns: {n_categorical}")
    print("Top 10 columns by missingness:")
    print(missing_frac.head(10).to_string())

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "fraud_rate": fraud_rate,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
    }


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """Split preserving the ~3.5% fraud rate in every split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[TARGET_COL],
        random_state=random_state,
    )
    return DatasetSplits(train=train, val=val, test=test)


def get_numeric_feature_columns(df: pd.DataFrame) -> list:
    """Numeric columns excluding identifiers and the label itself."""
    exclude = {ID_COL, TARGET_COL}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in exclude]


def get_categorical_feature_columns(df: pd.DataFrame) -> list:
    """Categorical (non-numeric) columns -- ProductCD, card4/card6,
    DeviceType, email domains, M1-M9 flags, and the identity id_*
    categorical fields. Previously unused by graph.py -- this is what
    feeds encode_categorical_features below."""
    return list(df.select_dtypes(exclude=[np.number]).columns)


def encode_categorical_features(
    df: pd.DataFrame, categorical_cols: list, max_onehot_cardinality: int = 10
) -> pd.DataFrame:
    """Encode categorical columns as numeric features.

    - Low-cardinality columns (<= max_onehot_cardinality unique values,
      including a category for missing) are one-hot encoded.
    - High-cardinality columns (e.g. DeviceInfo, email domains, which can
      have hundreds to thousands of distinct values) are frequency-encoded
      instead -- one-hot would blow up dimensionality; frequency (how
      common is this value) is a reasonable fraud signal on its own, since
      rare devices/domains tend to correlate with fraud.

    NOTE: like the StandardScaler fit in graph.py's build_graph, this fits
    vocabularies and frequencies on the whole (sub)graph rather than
    train-split-only -- an acceptable simplification for this exploratory
    stage, to be revisited once train.py has proper train-only-fit
    preprocessing for every feature, not just the numeric ones.
    """
    onehot_parts = []
    freq_parts = []

    for col in categorical_cols:
        values = df[col].astype("object").fillna("__missing__")
        n_unique = values.nunique()

        if n_unique <= max_onehot_cardinality:
            dummies = pd.get_dummies(values, prefix=col)
            onehot_parts.append(dummies)
        else:
            freq_map = values.value_counts(normalize=True)
            freq_parts.append(values.map(freq_map).rename(f"{col}_freq"))

    parts = onehot_parts + freq_parts
    if not parts:
        return pd.DataFrame(index=df.index)
    encoded = pd.concat(parts, axis=1).astype(np.float32)
    encoded.index = df.index
    return encoded


def build_preprocessing_pipeline(n_components: int = 50) -> Pipeline:
    """StandardScaler + PCA on numeric transaction features.

    Used by the classical baselines (Isolation Forest / Random Forest) and
    as an optional dimensionality-reduced input to the GNN's node features.
    n_components=50 is a starting point -- revisit after checking the
    explained-variance curve.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])


if __name__ == "__main__":
    print("Loading IEEE-CIS Fraud Detection dataset...")
    df = load_raw()

    print("\n" + "=" * 50)
    print("DATASET REPORT")
    print("=" * 50)
    basic_report(df)

    print("\nSplitting (stratified 70/15/15 train/val/test)...")
    splits = stratified_split(df)
    for name, split_df in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        print(f"  {name}: {len(split_df):,} rows, fraud rate {split_df[TARGET_COL].mean():.4%}")

    print("\nFitting StandardScaler + PCA on numeric features (train split only)...")
    numeric_cols = get_numeric_feature_columns(df)
    print(f"  {len(numeric_cols)} numeric feature columns found")

    train_numeric = splits.train[numeric_cols].fillna(splits.train[numeric_cols].median())

    pipeline = build_preprocessing_pipeline(n_components=50)
    pipeline.fit(train_numeric)
    explained = pipeline.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"  PCA(50 components) explains {explained:.2%} of variance on train split")

    print("\nDATASET.PY CHECKS PASSED")