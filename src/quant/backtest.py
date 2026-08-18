"""
Vectorized backtesting engine, tying together factor_model.py -> portfolio.py
-> risk.py into a walk-forward strategy test.

The single most important correctness property of any backtest is avoiding
LOOKAHEAD BIAS: the position held on day t must be decided using only
information available strictly before day t, then applied to day t's
realized return. Get the lag wrong and a backtest can show fantastic, fake
returns (this is the most common way backtests lie to people). This is
verified explicitly in the __main__ sanity check below, not just asserted.

Two strategies are demonstrated:
  1. A simple momentum signal (rules-based, no lookahead by construction).
  2. A walk-forward, monthly-rebalanced max-Sharpe portfolio using
     portfolio.py's optimizer refit only on trailing data available at each
     rebalance date (never the full sample) -- this is what makes it a
     legitimate walk-forward test rather than an in-sample fit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from factor_model import load_returns_matrix
from portfolio import (get_factor_covariance_annualized, expected_returns_annualized,
                        max_sharpe_portfolio_numerical)
from risk import risk_report

TRANSACTION_COST_BPS = 5  # 5 basis points per unit of turnover


def vectorized_backtest(returns_df, weights_df, transaction_cost_bps=TRANSACTION_COST_BPS):
    """
    returns_df, weights_df: same DatetimeIndex and columns (tickers).
    weights_df[t] is the TARGET weight decided using information available
    as of day t; it is applied to returns_df[t+1] (shifted forward by one
    day) so that no future information leaks into the position that earns
    that day's return.
    """
    weights_df = weights_df.reindex(returns_df.index).ffill().fillna(0.0)
    applied_weights = weights_df.shift(1).fillna(0.0)  # the lookahead-avoidance step

    gross_returns = (applied_weights * returns_df).sum(axis=1)
    turnover = applied_weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (transaction_cost_bps / 10_000.0)
    net_returns = gross_returns - costs

    equity_curve = (1 + net_returns).cumprod()
    return {
        "net_returns": net_returns,
        "gross_returns": gross_returns,
        "turnover": turnover,
        "equity_curve": equity_curve,
        "applied_weights": applied_weights,
    }


def momentum_signal(returns_df, lookback=21):
    """Equal-weight the assets with positive trailing lookback-day momentum
    (inclusive of day t itself -- 'what I'd compute right after today's
    close'), zero weight otherwise. Deliberately NOT shifted here -- lagging
    happens in exactly one place, vectorized_backtest's shift(1), so weights
    for day t only ever get applied to day t+1's return. Two independent
    shifts (here and in the engine) would silently double-lag the signal."""
    trailing_return = returns_df.rolling(lookback).sum()
    is_positive = trailing_return > 0
    n_positive = is_positive.sum(axis=1).replace(0, np.nan)
    weights = is_positive.div(n_positive, axis=0).fillna(0.0)
    return weights


def walk_forward_max_sharpe(returns_df, rebalance_freq="ME", lookback_days=126, r_f=0.02):
    """
    At each rebalance date, refits the factor-model covariance and max-Sharpe
    weights using ONLY the trailing lookback_days of returns up to (and
    including) that date -- never future data. Weights are held constant
    between rebalances.
    """
    rebalance_dates = returns_df.resample(rebalance_freq).last().index
    rebalance_dates = [d for d in rebalance_dates if d in returns_df.index or
                        returns_df.index[returns_df.index <= d].shape[0] > 0]

    weight_rows = []
    tickers = returns_df.columns.tolist()
    for d in rebalance_dates:
        trailing = returns_df.loc[returns_df.index <= d].tail(lookback_days)
        if len(trailing) < lookback_days // 2:
            continue
        try:
            Sigma = get_factor_covariance_annualized(trailing, n_factors=min(2, len(tickers) - 1))
            mu = expected_returns_annualized(trailing)
            w = max_sharpe_portfolio_numerical(mu, Sigma, r_f=r_f, long_only=True)
        except Exception:
            w = np.ones(len(tickers)) / len(tickers)
        weight_rows.append(pd.Series(w, index=tickers, name=d))

    weights_df = pd.DataFrame(weight_rows)
    return weights_df


def _lookahead_bug_demo(returns_df):
    """Deliberately WRONG version -- applies day-t's weights (computed using
    data through and including day t) directly to day t's OWN return,
    instead of day t+1's. This is genuine lookahead: it uses day t's return
    twice -- once inside the signal (today's momentum) and once as the P&L
    that return supposedly earned, which is impossible to achieve live."""
    weights_df = momentum_signal(returns_df).reindex(returns_df.index).ffill().fillna(0.0)
    cheating_returns = (weights_df * returns_df).sum(axis=1)  # NOT shifted -- genuine lookahead
    return cheating_returns


if __name__ == "__main__":
    print("=" * 60)
    print("SANITY CHECK: lookahead-bias lag actually matters")
    print("=" * 60)
    returns_df = load_returns_matrix()

    mom_weights = momentum_signal(returns_df)
    correct = vectorized_backtest(returns_df, mom_weights)
    cheating_returns = _lookahead_bug_demo(returns_df)

    correct_sharpe = risk_report(correct["net_returns"].dropna())["sharpe"]
    cheating_sharpe = risk_report(cheating_returns.dropna())["sharpe"]
    print(f"correctly-lagged momentum strategy Sharpe: {correct_sharpe:.4f}")
    print(f"same-day (buggy, lookahead) Sharpe:        {cheating_sharpe:.4f}")
    lookahead_check = cheating_sharpe > correct_sharpe
    print(f"lookahead version shows inflated Sharpe (expected): {lookahead_check}")
    print("RESULT:", "PASSED" if lookahead_check else "FAILED (lag isn't doing anything -- investigate)")

    print("\n" + "=" * 60)
    print("SANITY CHECK: constant equal weights == buy-and-hold-equal-weight")
    print("=" * 60)
    tickers = returns_df.columns.tolist()
    equal_weights_df = pd.DataFrame(
        np.ones((len(returns_df), len(tickers))) / len(tickers),
        index=returns_df.index, columns=tickers,
    )
    bh_result = vectorized_backtest(returns_df, equal_weights_df, transaction_cost_bps=0)
    direct_equal_weight_return = (returns_df.mean(axis=1))
    diff = (bh_result["net_returns"].iloc[5:] - direct_equal_weight_return.iloc[5:]).abs().max()
    print(f"max diff vs direct equal-weight return calc: {diff:.10f}")
    bh_check = diff < 1e-9
    print("RESULT:", "PASSED" if bh_check else "FAILED")

    print("\n" + "=" * 60)
    print("STRATEGY 1: Momentum (rules-based)")
    print("=" * 60)
    mom_report = risk_report(correct["net_returns"].dropna())
    for k, v in mom_report.items():
        print(f"  {k}: {v:.6f}")
    print(f"  final equity (from $1): {correct['equity_curve'].iloc[-1]:.4f}")
    print(f"  avg daily turnover: {correct['turnover'].mean():.4f}")

    print("\n" + "=" * 60)
    print("STRATEGY 2: Walk-forward monthly max-Sharpe (portfolio.py, no lookahead)")
    print("=" * 60)
    wf_weights = walk_forward_max_sharpe(returns_df)
    wf_result = vectorized_backtest(returns_df, wf_weights)
    wf_returns = wf_result["net_returns"].dropna()
    wf_report = risk_report(wf_returns) if len(wf_returns) > 30 else None
    if wf_report:
        for k, v in wf_report.items():
            print(f"  {k}: {v:.6f}")
        print(f"  final equity (from $1): {wf_result['equity_curve'].dropna().iloc[-1]:.4f}")
    else:
        print("  not enough data for a full walk-forward report")

    print("\n" + "=" * 60)
    print("BENCHMARK: equal-weight buy-and-hold")
    print("=" * 60)
    bh_report = risk_report(bh_result["net_returns"].dropna())
    for k, v in bh_report.items():
        print(f"  {k}: {v:.6f}")
    print(f"  final equity (from $1): {bh_result['equity_curve'].iloc[-1]:.4f}")

    all_passed = lookahead_check and bh_check
    print("\nRESULT: ALL CHECKS PASSED" if all_passed else "\nRESULT: one or more checks FAILED")