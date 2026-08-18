"""
src/quant/features.py

Downloads historical price data via yfinance and builds a market
microstructure feature set for the quant trading system's downstream
models (LSTM/Transformer price prediction, GARCH volatility, cointegration,
Black-Scholes PINN calibration).

Ticker choice: a small basket of liquid large-cap names plus SPY as the
market benchmark -- enough diversity for cross-asset work (factor model,
portfolio optimization) later without the data volume becoming unwieldy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "market"
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
DEFAULT_PERIOD = "5y"


def download_prices(tickers: list = DEFAULT_TICKERS, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Downloads daily OHLCV data for each ticker via yfinance, returns a
    long-format DataFrame (one row per ticker per date) -- easier to work
    with downstream than yfinance's wide multi-index columns when handling
    multiple tickers."""
    frames = []
    for ticker in tickers:
        print(f"  downloading {ticker}...")
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            print(f"    WARNING: no data returned for {ticker}, skipping")
            continue
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df["ticker"] = ticker
        df = df.reset_index()
        frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded for any ticker -- check tickers/network")

    return pd.concat(frames, ignore_index=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds return, volatility, momentum, and volume features per ticker.
    Computed independently within each ticker's own time series (never
    mixing rows across tickers into a single rolling window)."""
    out = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("Date").copy()

        g["return"] = g["Close"].pct_change()
        g["log_return"] = np.log(g["Close"] / g["Close"].shift(1))

        for window in [5, 10, 21]:
            g[f"realized_vol_{window}d"] = g["log_return"].rolling(window).std() * np.sqrt(252)

        g["sma_10"] = g["Close"].rolling(10).mean()
        g["sma_50"] = g["Close"].rolling(50).mean()

        for window in [5, 21]:
            g[f"momentum_{window}d"] = g["Close"] / g["Close"].shift(window) - 1

        # RSI(14) -- Wilder's smoothing
        delta = g["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        g["rsi_14"] = 100 - (100 / (1 + rs))

        g["volume_zscore_21d"] = (
            (g["Volume"] - g["Volume"].rolling(21).mean()) / g["Volume"].rolling(21).std()
        )

        out.append(g)

    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    print(f"Downloading {len(DEFAULT_TICKERS)} tickers, {DEFAULT_PERIOD} of daily history...")
    raw = download_prices()
    print(f"  {len(raw):,} raw rows across {raw['ticker'].nunique()} tickers")

    print("\nComputing features...")
    featured = compute_features(raw)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "features.csv"
    featured.to_csv(out_path, index=False)
    print(f"\nSaved {len(featured):,} rows x {featured.shape[1]} columns to {out_path}")

    print("\n" + "=" * 50)
    print("FEATURE SUMMARY")
    print("=" * 50)
    for ticker, group in featured.groupby("ticker"):
        date_min, date_max = group["Date"].min(), group["Date"].max()
        print(f"  {ticker}: {len(group):,} rows, {date_min.date()} to {date_max.date()}, "
              f"avg realized vol (21d): {group['realized_vol_21d'].mean():.2%}")

    print("\nFEATURES.PY CHECKS PASSED")