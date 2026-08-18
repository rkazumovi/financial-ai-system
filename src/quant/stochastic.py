"""
Heston stochastic volatility model: simulation, Monte Carlo option pricing,
and comparison to Black-Scholes.

dS_t = r*S_t*dt + sqrt(v_t)*S_t*dW_t^S
dv_t = kappa*(theta - v_t)*dt + sigma_v*sqrt(v_t)*dW_t^v
corr(dW^S, dW^v) = rho * dt

Unlike Black-Scholes, variance v_t is itself a random mean-reverting process,
which lets the model reproduce the volatility smile/skew markets actually
show (implied vol varying by strike) instead of assuming one constant sigma.
There's no closed form for the European option price under Heston (unlike
Black-Scholes), so it's priced by Monte Carlo simulation instead.

See notebooks/math_derivations.ipynb for the full derivation and the
Ito's-lemma background this builds on.
"""
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

SEED = 42
S0 = 100.0
RATE = 0.05
MATURITY = 1.0

# Heston parameters (typical equity-index-like calibration)
V0 = 0.04         # initial variance (sqrt(0.04) = 20% initial vol)
KAPPA = 2.0        # mean-reversion speed of variance
THETA = 0.04        # long-run variance
SIGMA_V = 0.3         # vol-of-vol
RHO = -0.7             # correlation between price and variance shocks (equity skew)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


def black_scholes_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T,
                           n_steps=252, n_paths=200_000, seed=SEED):
    """
    Euler-Maruyama discretization with the 'full truncation' scheme
    (Lord, Koekkoek & Van Dijk, 2010): variance is floored at 0 wherever it's
    used (as a rate or under a sqrt), since the discretized process can dip
    negative even though the true continuous-time CIR-type process for v_t
    stays non-negative given standard parameter conditions.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, S0, dtype=np.float64)
    v = np.full(n_paths, v0, dtype=np.float64)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z_v = z1
        z_s = rho * z1 + np.sqrt(1 - rho ** 2) * z2

        v_pos = np.maximum(v, 0.0)
        S = S * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_s)
        v = v + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos * dt) * z_v

    return S


def mc_call_price(S_T, K, r, T):
    payoff = np.maximum(S_T - K, 0.0)
    price = np.exp(-r * T) * np.mean(payoff)
    se = np.exp(-r * T) * np.std(payoff) / np.sqrt(len(payoff))
    return price, se


def implied_vol(price, S0, K, T, r):
    def diff(sigma):
        return black_scholes_price(S0, K, T, r, sigma) - price
    try:
        return brentq(diff, 1e-4, 3.0)
    except ValueError:
        return float("nan")


def sanity_check_reduces_to_black_scholes(n_paths=200_000):
    """
    With sigma_v -> 0 and v0 = theta, variance never meaningfully moves, so
    Heston must reduce to plain Black-Scholes with sigma = sqrt(theta). If
    this doesn't hold (within Monte Carlo error), the simulation has a bug.
    """
    sigma_flat = 0.2
    theta = sigma_flat ** 2
    K = 100.0

    bs_price = black_scholes_price(S0, K, MATURITY, RATE, sigma_flat)
    S_T = simulate_heston_paths(S0, v0=theta, kappa=2.0, theta=theta, sigma_v=1e-4,
                                 rho=-0.7, r=RATE, T=MATURITY, n_paths=n_paths, seed=SEED)
    heston_price, se = mc_call_price(S_T, K, RATE, MATURITY)

    diff = abs(bs_price - heston_price)
    passed = diff < 3 * se
    return {"bs_price": bs_price, "heston_price": heston_price, "mc_se": se,
            "diff": diff, "passed": passed}


def volatility_smile(strikes, n_paths=200_000):
    S_T = simulate_heston_paths(S0, V0, KAPPA, THETA, SIGMA_V, RHO, RATE, MATURITY,
                                 n_paths=n_paths, seed=SEED)
    results = []
    for K in strikes:
        price, se = mc_call_price(S_T, K, RATE, MATURITY)
        iv = implied_vol(price, S0, K, MATURITY, RATE)
        results.append({"strike": K, "price": price, "mc_se": se, "implied_vol": iv})
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SANITY CHECK: Heston -> Black-Scholes as sigma_v -> 0")
    print("=" * 60)
    check = sanity_check_reduces_to_black_scholes()
    for k, v in check.items():
        print(f"  {k}: {v}")
    print("RESULT:", "PASSED" if check["passed"] else "FAILED")

    print("\n" + "=" * 60)
    print("VOLATILITY SMILE / SKEW")
    print("=" * 60)
    strikes = [70, 80, 90, 100, 110, 120, 130]
    smile = volatility_smile(strikes)
    for row in smile:
        print(f"  K={row['strike']:>5.0f} | price={row['price']:.4f} "
              f"(+/- {2*row['mc_se']:.4f}) | implied_vol={row['implied_vol']:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    ks = [r["strike"] for r in smile]
    ivs = [r["implied_vol"] for r in smile]
    plt.plot(ks, ivs, marker="o", label="Heston implied vol")
    plt.axhline(0.2, color="gray", linestyle="--", label="Black-Scholes constant vol (0.20)")
    plt.axvline(S0, color="cyan", linestyle=":", label=f"S0 = {S0:.0f}")
    plt.xlabel("Strike K")
    plt.ylabel("Black-Scholes implied volatility")
    plt.title("Heston-implied volatility skew vs. Black-Scholes' flat assumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heston_vol_smile.png", dpi=150)
    print(f"\nSaved smile plot to {OUTPUT_DIR / 'heston_vol_smile.png'}")

    if check["passed"] and all(not np.isnan(r["implied_vol"]) for r in smile):
        print("\nRESULT: ALL CHECKS PASSED")
    else:
        print("\nRESULT: one or more checks failed -- see output above")