"""
Sequence models for next-day return prediction: LSTM and Transformer.

Input: SEQUENCE_LENGTH days of engineered features (from features.py) per
ticker. Target: next day's simple return. Evaluated on RMSE, directional
accuracy, and Information Coefficient (Pearson correlation between predicted
and actual returns -- the standard metric for whether a signal has any
tradable value, since low RMSE alone doesn't establish that).

Train/val/test split is done by DATE, not randomly -- shuffling time series
data would leak future information into training, which is the single most
common mistake in quant ML pipelines.

Note: near-50% directional accuracy and small IC (0.01-0.05) on raw daily
returns is the textbook-expected result, not a bug -- daily returns are
close to a random walk (market efficiency). This isn't chased further with
extra tuning rounds; it's documented as-is.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "market"
FEATURES_PATH = DATA_DIR / "features.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

SEQUENCE_LENGTH = 30
TARGET_COL = "return"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_features():
    df = pd.read_csv(FEATURES_PATH)
    date_col = "date" if "date" in df.columns else "Date"
    ticker_col = "ticker" if "ticker" in df.columns else "Ticker"
    close_col = "Close" if "Close" in df.columns else "close"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {FEATURES_PATH}. "
                          f"Available columns: {list(df.columns)}")

    # Raw price/volume levels (Close, High, Low, Open, Volume, raw sma_10/50)
    # carry no legitimate return-predictive signal on their own -- a stock's
    # dollar price level is an arbitrary artifact of share count, not
    # information. Pooled across 5 tickers with very different price levels,
    # feeding these in raw lets the model shortcut on "which ticker is this"
    # instead of learning anything genuinely predictive. Convert the moving
    # averages to scale-invariant ratios instead of dropping their signal
    # entirely.
    if "sma_10" in df.columns:
        df["sma_10_ratio"] = df[close_col] / df["sma_10"] - 1.0
    if "sma_50" in df.columns:
        df["sma_50_ratio"] = df[close_col] / df["sma_50"] - 1.0

    raw_price_cols = {"Close", "close", "High", "high", "Low", "low",
                       "Open", "open", "Volume", "volume", "sma_10", "sma_50",
                       "Adj Close", "adj_close"}
    exclude = {date_col, ticker_col, TARGET_COL} | raw_price_cols
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    print(f"auto-detected feature columns ({len(feature_cols)}): {feature_cols}")

    df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    return df, feature_cols, date_col, ticker_col


def make_sequences(df, feature_cols, date_col, ticker_col, seq_len=SEQUENCE_LENGTH):
    """
    For each ticker independently, builds sliding windows of seq_len days of
    features, with the target being the return on the day AFTER the window
    ends (next-day prediction, using only past information).
    """
    X_list, y_list, date_list = [], [], []
    for ticker, group in df.groupby(ticker_col):
        group = group.sort_values(date_col).reset_index(drop=True)
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        target = group[TARGET_COL].to_numpy(dtype=np.float32)
        dates = group[date_col].to_numpy()
        for i in range(seq_len, len(group) - 1):
            X_list.append(feats[i - seq_len:i])
            y_list.append(target[i + 1])
            date_list.append(dates[i])
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    dates = np.array(date_list)
    return X, y, dates


def time_based_split(X, y, dates, train_frac=0.7, val_frac=0.15):
    order = np.argsort(dates)
    X, y, dates = X[order], y[order], dates[order]
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test": (X[n_train + n_val:], y[n_train + n_val:]),
    }
    return splits


def scale_features(splits, n_features):
    """Fits a StandardScaler on TRAIN ONLY, applies to all splits -- fitting
    on val/test would leak their distribution into preprocessing."""
    X_train = splits["train"][0]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    scaled = {}
    for name, (X, y) in splits.items():
        shape = X.shape
        X_scaled = scaler.transform(X.reshape(-1, n_features)).reshape(shape).astype(np.float32)
        scaled[name] = (X_scaled, y)
    return scaled, scaler


class LSTMPricePredictor(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerPricePredictor(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        out = self.encoder(x)
        last_token = out[:, -1, :]
        return self.head(last_token).squeeze(-1)


def evaluate(model, X, y, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)

    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    directional_acc = float(np.mean(np.sign(preds) == np.sign(y)))
    ic = float(np.corrcoef(preds, y)[0, 1]) if np.std(preds) > 1e-8 else 0.0
    return {"rmse": rmse, "directional_accuracy": directional_acc, "information_coefficient": ic}


def train_model(model, splits, epochs=50, batch_size=256, lr=1e-3, patience=8, verbose=True):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0

    n_train = len(X_train)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.tensor(X_train[idx], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(y_train[idx], dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n_train

        val_metrics = evaluate(model, X_val, y_val)
        if verbose:
            print(f"epoch {epoch:3d} | train_mse={train_loss:.6f} | val_rmse={val_metrics['rmse']:.6f} "
                  f"| val_dir_acc={val_metrics['directional_accuracy']:.4f} "
                  f"| val_ic={val_metrics['information_coefficient']:.4f}")

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("LOADING FEATURES")
    print("=" * 60)
    df, feature_cols, date_col, ticker_col = load_features()
    print(f"rows: {len(df)}, feature columns used: {feature_cols}")

    print("\n" + "=" * 60)
    print("BUILDING SEQUENCES")
    print("=" * 60)
    X, y, dates = make_sequences(df, feature_cols, date_col, ticker_col)
    print(f"sequences: {X.shape}, targets: {y.shape}")

    splits = time_based_split(X, y, dates)
    for name, (Xs, ys) in splits.items():
        print(f"  {name}: {len(Xs)} sequences")

    splits, scaler = scale_features(splits, n_features=len(feature_cols))

    results = {}
    for name, ModelClass in [("LSTM", LSTMPricePredictor), ("Transformer", TransformerPricePredictor)]:
        print("\n" + "=" * 60)
        print(f"TRAINING {name}")
        print("=" * 60)
        model = ModelClass(n_features=len(feature_cols))
        model = train_model(model, splits)
        test_metrics = evaluate(model, *splits["test"])
        print(f"{name} TEST METRICS: {test_metrics}")
        results[name] = test_metrics

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"  {name}: RMSE={metrics['rmse']:.6f}, "
              f"DirAcc={metrics['directional_accuracy']:.4f}, "
              f"IC={metrics['information_coefficient']:.4f}")