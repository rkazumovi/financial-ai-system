"""
Markowitz mean-variance portfolio optimization, using the factor-model
covariance estimate from factor_model.py.

min_w  w^T Sigma w   subject to   w^T mu = target_return,  sum(w) = 1

Unconstrained (shorting allowed), this has a closed-form solution via
Lagrange multipliers (the "two-fund separation theorem" -- every efficient
portfolio is a combination of the global minimum-variance portfolio and the
tangency/max-Sharpe portfolio). Long-only (no shorting) has no closed form
and is solved numerically instead. The unconstrained numerical solver is
checked against the closed-form solution as a correctness test before
trusting the long-only results.

Note: the max-Sharpe portfolio can come out heavily concentrated in one or
two assets -- this is the well-known sensitivity of mean-variance
optimization to estimation error in expected returns ("Markowitz's curse"),
not a bug. It's part of why the factor-model covariance from factor_model.py
is used here instead of the raw sample covariance.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from factor_model import (load_returns_matrix, fit_pca_factors, fit_ridge_betas,
                           reconstruct_covariance)

TRADING_DAYS = 252


def expected_returns_annualized(returns_matrix):
    return returns_matrix.mean().values * TRADING_DAYS


def get_factor_covariance_annualized(returns_matrix, n_factors=2):
    pca_result = fit_pca_factors(returns_matrix, n_factors=n_factors)
    betas, residual_var = fit_ridge_betas(returns_matrix, pca_result["factor_scores"], alpha=1.0)
    tickers = returns_matrix.columns.tolist()
    Sigma_daily = reconstruct_covariance(betas, pca_result["factor_scores"], residual_var, tickers)
    return Sigma_daily * TRADING_DAYS


def min_variance_portfolio_closed_form(Sigma):
    """w = Sigma^-1 * 1 / (1^T * Sigma^-1 * 1) -- the unconstrained
    global minimum-variance portfolio, no target return constraint."""
    n = Sigma.shape[0]
    ones = np.ones(n)
    Sigma_inv_ones = np.linalg.solve(Sigma, ones)
    w = Sigma_inv_ones / (ones @ Sigma_inv_ones)
    return w


def min_variance_portfolio_numerical(Sigma, long_only=False):
    n = Sigma.shape[0]
    w0 = np.ones(n) / n

    def objective(w):
        return w @ Sigma @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else None  # None = truly unbounded

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"Optimizer failed to converge: {result.message}")
    return result.x


def max_sharpe_portfolio_closed_form(mu, Sigma, r_f=0.02):
    """w proportional to Sigma^-1 * (mu - r_f * 1), normalized to sum to 1 --
    the unconstrained tangency portfolio."""
    excess = mu - r_f
    Sigma_inv_excess = np.linalg.solve(Sigma, excess)
    w = Sigma_inv_excess / np.sum(Sigma_inv_excess)
    return w


def max_sharpe_portfolio_numerical(mu, Sigma, r_f=0.02, long_only=True):
    n = Sigma.shape[0]
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ Sigma @ w)
        return -(port_return - r_f) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else None  # None = truly unbounded

    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"Optimizer failed to converge: {result.message}")
    return result.x


def portfolio_stats(w, mu, Sigma, r_f=0.02):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    sharpe = (ret - r_f) / vol
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def efficient_frontier(mu, Sigma, n_points=20):
    """Sweeps target returns between the min-variance portfolio's return and
    the max-return single asset, solving the constrained QP at each point."""
    n = len(mu)
    min_ret = float(mu @ min_variance_portfolio_closed_form(Sigma))
    max_ret = float(np.max(mu))
    targets = np.linspace(min_ret, max_ret * 0.95, n_points)

    frontier = []
    for target in targets:
        w0 = np.ones(n) / n

        def objective(w):
            return w @ Sigma @ w

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        bounds = [(0.0, 1.0)] * n
        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-12, "maxiter": 1000})
        if result.success:
            vol = float(np.sqrt(result.x @ Sigma @ result.x))
            frontier.append({"target_return": target, "volatility": vol, "weights": result.x})
    return frontier


if __name__ == "__main__":
    print("=" * 60)
    print("SANITY CHECK: closed-form vs numerical min-variance portfolio")
    print("=" * 60)
    returns_matrix = load_returns_matrix()
    tickers = returns_matrix.columns.tolist()
    Sigma = get_factor_covariance_annualized(returns_matrix)
    mu = expected_returns_annualized(returns_matrix)

    w_closed = min_variance_portfolio_closed_form(Sigma)
    w_numerical = min_variance_portfolio_numerical(Sigma, long_only=False)
    max_diff = float(np.max(np.abs(w_closed - w_numerical)))
    print(f"tickers: {tickers}")
    print(f"closed-form weights:   {np.round(w_closed, 4)}")
    print(f"numerical weights:     {np.round(w_numerical, 4)}")
    print(f"max abs diff: {max_diff:.8f}")
    check_passed = max_diff < 1e-4
    print("RESULT:", "PASSED" if check_passed else "FAILED")

    print("\n" + "=" * 60)
    print("PORTFOLIOS (long-only)")
    print("=" * 60)
    w_minvar = min_variance_portfolio_numerical(Sigma, long_only=True)
    w_maxsharpe = max_sharpe_portfolio_numerical(mu, Sigma, r_f=0.02, long_only=True)

    print("Min-variance portfolio:")
    for t, w in zip(tickers, w_minvar):
        print(f"  {t}: {w:.4f}")
    print(f"  stats: {portfolio_stats(w_minvar, mu, Sigma)}")

    print("\nMax-Sharpe portfolio:")
    for t, w in zip(tickers, w_maxsharpe):
        print(f"  {t}: {w:.4f}")
    print(f"  stats: {portfolio_stats(w_maxsharpe, mu, Sigma)}")

    print("\nEqual-weight baseline:")
    w_equal = np.ones(len(tickers)) / len(tickers)
    print(f"  stats: {portfolio_stats(w_equal, mu, Sigma)}")

    print("\n" + "=" * 60)
    print("EFFICIENT FRONTIER (long-only, 10 points)")
    print("=" * 60)
    frontier = efficient_frontier(mu, Sigma, n_points=10)
    for point in frontier:
        print(f"  target_return={point['target_return']:.4f} | volatility={point['volatility']:.4f}")

    beats_equal_weight_sharpe = (portfolio_stats(w_maxsharpe, mu, Sigma)["sharpe"]
                                  >= portfolio_stats(w_equal, mu, Sigma)["sharpe"])
    all_frontier_points_valid = len(frontier) >= 8

    print("\nRESULT: ALL CHECKS PASSED" if (check_passed and beats_equal_weight_sharpe and all_frontier_points_valid)
          else "\nRESULT: one or more checks FAILED -- see output above")