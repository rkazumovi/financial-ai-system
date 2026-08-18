"""
Risk-adjusted performance metrics: Sharpe, Sortino, max drawdown, VaR, CVaR.

Sharpe penalizes upside and downside volatility equally, which double-counts
"risk" that investors don't actually mind (a stellar up day looks the same
as a bad down day to the Sharpe ratio's denominator). Sortino only penalizes
downside deviation relative to a minimum acceptable return (MAR).

VaR (Value at Risk) at confidence level alpha is the loss threshold not
expected to be exceeded on (1-alpha) of days -- the (1-alpha)-th percentile
of the loss distribution. CVaR (Conditional VaR / Expected Shortfall) is the
average loss GIVEN that the loss exceeds VaR -- strictly more informative
about tail risk, and always >= VaR by construction (it's an average over a
subset of the losses that VaR itself is the boundary of).
"""
import numpy as np
from scipy.stats import norm


def sharpe_ratio(returns, r_f=0.02, periods_per_year=252):
    returns = np.asarray(returns)
    excess = returns.mean() * periods_per_year - r_f
    vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return float(excess / vol) if vol > 0 else float("nan")


def sortino_ratio(returns, mar=0.0, r_f=0.02, periods_per_year=252):
    returns = np.asarray(returns)
    excess = returns.mean() * periods_per_year - r_f
    downside = returns[returns < mar] - mar
    if len(downside) == 0:
        return float("inf")
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)
    return float(excess / downside_dev) if downside_dev > 0 else float("nan")


def max_drawdown(cumulative_returns):
    """cumulative_returns: array of portfolio value (or cumulative return
    index) over time, NOT period returns. Returns (max_drawdown, drawdown_series)."""
    cumulative_returns = np.asarray(cumulative_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return float(drawdown.min()), drawdown


def historical_var(returns, confidence=0.95):
    """Empirical VaR: the (1-confidence) percentile of the return
    distribution, reported as a positive loss number."""
    returns = np.asarray(returns)
    return float(-np.percentile(returns, (1 - confidence) * 100))


def historical_cvar(returns, confidence=0.95):
    """Empirical CVaR: mean of returns at or below the VaR threshold."""
    returns = np.asarray(returns)
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= var_threshold]
    return float(-tail.mean()) if len(tail) > 0 else historical_var(returns, confidence)


def parametric_var(mu, sigma, confidence=0.95):
    """Closed-form VaR assuming Gaussian returns: VaR = -(mu + sigma*z)
    where z = Phi^-1(1 - confidence) (a negative number)."""
    z = norm.ppf(1 - confidence)
    return float(-(mu + sigma * z))


def parametric_cvar(mu, sigma, confidence=0.95):
    """Closed-form CVaR (Expected Shortfall) assuming Gaussian returns:
    CVaR = -(mu - sigma * phi(z) / (1 - confidence)), z = Phi^-1(1-confidence)."""
    z = norm.ppf(1 - confidence)
    return float(-(mu - sigma * norm.pdf(z) / (1 - confidence)))


def risk_report(returns, r_f=0.02, periods_per_year=252, confidence=0.95):
    cumulative = np.cumprod(1 + np.asarray(returns))
    mdd, _ = max_drawdown(cumulative)
    return {
        "sharpe": sharpe_ratio(returns, r_f, periods_per_year),
        "sortino": sortino_ratio(returns, 0.0, r_f, periods_per_year),
        "max_drawdown": mdd,
        "historical_var_95": historical_var(returns, confidence),
        "historical_cvar_95": historical_cvar(returns, confidence),
        "annualized_return": float(np.mean(returns) * periods_per_year),
        "annualized_vol": float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SANITY CHECK: historical vs parametric VaR/CVaR on simulated Gaussian returns")
    print("=" * 60)
    rng = np.random.default_rng(42)
    mu, sigma = 0.0005, 0.02
    simulated_returns = rng.normal(mu, sigma, size=100_000)

    hist_var = historical_var(simulated_returns, 0.95)
    param_var = parametric_var(mu, sigma, 0.95)
    hist_cvar = historical_cvar(simulated_returns, 0.95)
    param_cvar = parametric_cvar(mu, sigma, 0.95)

    print(f"historical VaR(95%):   {hist_var:.6f}")
    print(f"parametric VaR(95%):   {param_var:.6f}")
    print(f"historical CVaR(95%):  {hist_cvar:.6f}")
    print(f"parametric CVaR(95%):  {param_cvar:.6f}")

    var_check = abs(hist_var - param_var) < 0.001
    cvar_check = abs(hist_cvar - param_cvar) < 0.001
    cvar_geq_var_check = hist_cvar >= hist_var and param_cvar >= param_var
    print(f"\nVaR match (hist vs parametric): {var_check}")
    print(f"CVaR match (hist vs parametric): {cvar_check}")
    print(f"CVaR >= VaR (must always hold): {cvar_geq_var_check}")

    all_passed = var_check and cvar_check and cvar_geq_var_check
    print("\nRESULT:", "PASSED" if all_passed else "FAILED")

    print("\n" + "=" * 60)
    print("FULL RISK REPORT (same simulated returns)")
    print("=" * 60)
    report = risk_report(simulated_returns)
    for k, v in report.items():
        print(f"  {k}: {v:.6f}")