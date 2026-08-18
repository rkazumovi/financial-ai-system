"""
One-time script: appends the System 2 math sections built since the last
notebook update to notebooks/math_derivations.ipynb -- the PINN debugging
case study (loss-weighting and the hard-constraint tradeoff), the Heston
model, Markowitz portfolio theory, the PCA/Ridge factor model, and the risk
metrics used in risk.py. Run once; safe to re-run (just appends again, so
don't run twice without checking cell count first).
"""
from pathlib import Path
import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent / "notebooks" / "math_derivations.ipynb"

nb = nbf.read(NOTEBOOK_PATH, as_version=4)
print(f"Loaded notebook with {len(nb.cells)} existing cells.")

new_cells = []

new_cells.append(nbf.v4.new_markdown_cell(r"""
## 9. Case study: diagnosing a Black-Scholes PINN's accuracy plateau

This section documents the actual debugging path for `pinn_bs.py`, kept as a
case study because each fix was individually correct and still didn't
produce the naive expectation -- a useful lesson in verifying end results,
not just the reasoning behind a fix.

**Problem 1 -- input saturation.** Feeding raw $S \in [0, 300]$ directly into
a `Tanh`-activated network saturates the activation (Tanh is nearly flat for
$|x| \gg 1$), killing gradient flow. Fix: normalize inputs,
$\hat S = S / S_{\max}$, $\hat t = t / T$, both in $[0,1]$.

**Problem 2 -- loss-term scale imbalance.** Even after input normalization,
the network still predicted $V$ directly in dollars (up to ~$200). At
initialization, the three loss terms lived on wildly different scales:

$$
\mathcal{L}_{\text{pde}} \sim 10^{-5}, \qquad
\mathcal{L}_{\text{terminal}} \sim 10^{4}, \qquad
\mathcal{L}_{\text{boundary}} \sim 10^{4}
$$

Whichever term is numerically larger dominates the combined gradient signal
through Adam's shared parameters, even though Adam's per-parameter adaptive
scaling partially compensates. Fix: **output non-dimensionalization** -- have
the network predict a dimensionless price $v = V/K$ instead of raw dollars,
and rescale outside the network:

$$
V(S, t) = K \cdot v\!\left(\frac{S}{S_{\max}}, \frac{t}{T}\right)
$$

Because the Black-Scholes PDE is **linear and homogeneous in $V$**, dividing
the whole PDE residual by $K$ gives an equally valid, better-scaled version
of the same constraint:

$$
\frac{1}{K}\left(\frac{\partial V}{\partial t} + \tfrac12 \sigma^2 S^2
\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV\right) = 0
$$

This alone took the mean relative error from 19.8% down to 8.85%, and after
also fixing the loss-term normalization, to 6.33%.

**Problem 3 -- the terminal kink.** The payoff $\max(S-K,0)$ has a kink at
$S=K$: the second derivative is singular there in the continuum limit, and
this is exactly the region uniform random collocation sampling under-covers.
Diagnosed via a per-region error breakdown showing error concentrated near
$(S,t) \approx (K, T)$.

**Attempted fix -- hard-constraining the terminal condition.** Instead of
learning the terminal condition via a loss penalty, bake it into the
network's output by construction (a "trial solution" ansatz, going back to
Lagaris et al. 1998):

$$
V(S, t) = \max(S - K, 0) + (T - t) \cdot K \cdot \text{correction}(S, t)
$$

At $t = T$, $(T-t) = 0$, so $V(S,T) = \max(S-K,0)$ **exactly**, for every
$S$, with zero approximation error -- no learning needed for the terminal
condition at all.

**The tradeoff this introduces.** The $(T-t)$ factor is *largest* at $t=0$
(today's price, the value that actually matters for real pricing) and
vanishes at $t=T$. Any absolute error $\delta$ in the learned `correction`
function produces a price error of roughly $(T-t) \cdot K \cdot \delta$ --
maximal at $t=0$, minimal near $t=T$. Diagnosed via the same per-region error
breakdown: the worst error moved from $t \approx T$ (the kink) to
$t = 0$ (maximum distance from the one anchor point), with max absolute
error actually increasing from ~$1.3 to ~$7. This is a general feature of
solving a **backward** parabolic PDE from a single terminal anchor: small
per-point PDE residual errors compound over the "distance" (here, time)
from that anchor, similar to how a numerical PDE solver accumulates global
error the further it marches from known boundary data.

A causal / curriculum training scheme (train near $t=T$ first, then widen
the sampled window backward toward $t=0$, following Wang et al. 2022,
*"Respecting causality is all you need for training physics-informed
neural networks"*) was tried and did not resolve this -- confirming the
issue is the ansatz's structural error amplification, not training order.

**Resolution.** The soft-terminal-loss version (Problem 1 + 2 fixes only,
without the hard constraint) had **no** such amplification pathology and
remained the best result across every variant tried:

| Variant | Mean relative error |
|---|---|
| Unnormalized inputs | 19.8% |
| + input normalization only | 8.85% |
| + output normalization (final) | **4.77%** |
| + hard-constrained terminal | 7.0-13% (new $t=0$ pathology) |
| + hard-constraint + causal curriculum | 7.0% (unchanged) |
| + hard-constraint + log-price coordinates | 13.5% (worse) |

The lesson: a change that correctly fixes a diagnosed problem can introduce
a different, comparably-sized problem elsewhere -- the only way to know is
to re-measure the actual end-to-end metric, not just confirm the original
symptom is gone.
"""))

new_cells.append(nbf.v4.new_markdown_cell(r"""
## 10. The Heston stochastic volatility model

Black-Scholes assumes constant volatility $\sigma$, which cannot reproduce
the volatility smile/skew real option markets show (implied volatility
varying systematically by strike). Heston (1993) fixes this by making
variance itself a random, mean-reverting process:

$$
dS_t = r S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^S
$$
$$
dv_t = \kappa(\theta - v_t)\, dt + \sigma_v \sqrt{v_t}\, dW_t^v
$$
$$
\text{corr}(dW_t^S, dW_t^v) = \rho\, dt
$$

where $\kappa$ is the speed of mean reversion, $\theta$ the long-run
variance, $\sigma_v$ the "vol-of-vol", and $\rho$ (typically negative for
equities) captures the well-documented leverage effect: volatility tends to
spike when prices fall.

The variance process is a **CIR (Cox-Ingersoll-Ross) process**; under the
Feller condition $2\kappa\theta \geq \sigma_v^2$ it stays strictly positive
in continuous time. Discretized (Euler-Maruyama) it can still dip negative
due to the discrete step size, so the implementation uses the **full
truncation scheme** (Lord, Koekkoek & Van Dijk, 2010): wherever $v_t$
appears as a rate or under a square root, it's floored at 0.

**Correlated shocks.** Given independent standard normals $Z_1, Z_2$, the
correlated pair is constructed via a Cholesky-style decomposition of the
$2\times 2$ correlation matrix:

$$
Z^v = Z_1, \qquad Z^S = \rho Z_1 + \sqrt{1-\rho^2}\, Z_2
$$

**No closed form (here).** Heston does have a semi-analytical price via the
characteristic function and Fourier inversion (the original 1993 approach),
but this implementation prices by **Monte Carlo simulation** instead, for
simplicity and because it generalizes trivially to path-dependent payoffs
that the Fourier approach doesn't handle as cleanly.

**Sanity check used in `stochastic.py`.** As $\sigma_v \to 0$ with
$v_0 = \theta$, variance never meaningfully moves, so Heston must reduce
exactly to Black-Scholes with $\sigma = \sqrt{\theta}$ -- verified
numerically to within Monte Carlo standard error before trusting the full
model.
"""))

new_cells.append(nbf.v4.new_markdown_cell(r"""
## 11. Markowitz mean-variance portfolio theory

**Problem.** Find portfolio weights $w$ minimizing variance for a given
target expected return, fully invested:

$$
\min_w \; w^\top \Sigma w \quad \text{s.t.} \quad w^\top \mu = R,\;\; \mathbf{1}^\top w = 1
$$

**Lagrangian:**

$$
\mathcal{L} = w^\top \Sigma w - \lambda_1(w^\top \mu - R) - \lambda_2(\mathbf{1}^\top w - 1)
$$

Setting $\partial \mathcal{L}/\partial w = 0$:

$$
2\Sigma w = \lambda_1 \mu + \lambda_2 \mathbf{1}
\quad\Longrightarrow\quad
w = \tfrac12 \Sigma^{-1}(\lambda_1 \mu + \lambda_2 \mathbf{1})
$$

Substituting back into the two constraints and solving for
$\lambda_1, \lambda_2$ (using $A = \mu^\top\Sigma^{-1}\mu$,
$B = \mu^\top\Sigma^{-1}\mathbf{1}$, $C = \mathbf{1}^\top\Sigma^{-1}\mathbf{1}$,
the standard Merton 1972 notation) gives the efficient frontier in closed
form -- every point on it is a linear combination of exactly two portfolios,
the **two-fund separation theorem**.

**Special case 1 -- global minimum-variance** (no return target):

$$
w_{\min\text{var}} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}
$$

**Special case 2 -- maximum Sharpe (tangency portfolio):**

$$
w_{\max\text{Sharpe}} \propto \Sigma^{-1}(\mu - r_f \mathbf{1})
$$

normalized so weights sum to 1.

**Long-only constraint.** Both closed forms allow shorting (unconstrained).
Adding $w_i \geq 0$ turns this into a constrained quadratic program with no
closed form, solved numerically instead (`scipy.optimize.minimize`,
SLSQP). `portfolio.py` verifies the numerical solver against the closed
form in the unconstrained case as a correctness check before trusting the
long-only results -- both must find the identical optimum since they solve
the same problem two different ways.

**A known pathology.** Unconstrained (or even long-only) mean-variance
optimization is notoriously sensitive to estimation error in $\mu$,
routinely producing extreme, concentrated, or leveraged "corner solution"
portfolios rather than diversified ones (observed directly in
`portfolio.py`'s results: an unconstrained min-variance weight of 1.59 on
one asset, financed by shorting the others). This sensitivity is a large
part of *why* `factor_model.py`'s regularized covariance estimate is used
here instead of the raw sample covariance.
"""))

new_cells.append(nbf.v4.new_markdown_cell(r"""
## 12. Statistical factor model: PCA + Ridge regression

**Motivation.** The sample covariance matrix $\hat\Sigma$ is a noisy
estimate, especially when the number of assets is large relative to the
number of observations -- small estimation errors, especially among
correlated assets, get amplified by the matrix inversion inside Markowitz
optimization into unstable portfolio weights.

**PCA factor extraction.** Standardize returns, then take the top-$k$
eigenvectors of the sample correlation matrix as latent common factors. The
factor scores are the returns projected onto those eigenvectors; the
fraction of total variance each explains is its eigenvalue divided by the
sum of all eigenvalues.

**Ridge factor loadings.** For each asset $i$, estimate loadings on the
extracted factors via $L_2$-penalized regression instead of plain OLS:

$$
\hat\beta_i = \arg\min_\beta \; \|r_i - F\beta\|^2 + \alpha \|\beta\|^2
\;=\; (F^\top F + \alpha I)^{-1} F^\top r_i
$$

The penalty keeps loadings stable when factor scores are themselves
correlated, at the cost of some bias -- a standard bias-variance tradeoff.

**Reconstructed covariance.** Under the factor model assumption that
idiosyncratic residuals $\varepsilon_i$ are uncorrelated across assets and
uncorrelated with the factors:

$$
\Sigma = B\, \Omega\, B^\top + D
$$

where $B$ is the loadings matrix, $\Omega$ the factor covariance, and $D$
the diagonal matrix of idiosyncratic (residual) variances. This is positive
semi-definite by construction (verified numerically via eigenvalues in
`factor_model.py`) -- a hard requirement for Markowitz optimization to be
well-posed at all.

**Honest caveat.** With only 5 assets and far more time observations than
assets, the sample covariance here is already well-estimated, so the
factor model's noise-reduction benefit doesn't show up as an improved
condition number in this specific example -- that benefit matters most with
many more assets than observations. The decomposition methodology itself
is unaffected by this and was verified correct via the real data showing
~80% of variance explained by 2 factors (the expected result for equities,
which share a strong common market factor).
"""))

new_cells.append(nbf.v4.new_markdown_cell(r"""
## 13. Risk-adjusted performance metrics

**Sharpe ratio** -- excess return per unit of *total* volatility:

$$
\text{Sharpe} = \frac{\bar{r} \cdot p - r_f}{\sigma_r \sqrt{p}}
$$

for $p$ periods per year. Penalizes upside and downside volatility equally.

**Sortino ratio** -- excess return per unit of *downside* volatility only,
relative to a minimum acceptable return (MAR, here 0):

$$
\text{Sortino} = \frac{\bar r \cdot p - r_f}{\sigma_{\text{downside}}\sqrt p},
\qquad
\sigma_{\text{downside}} = \sqrt{\mathbb{E}\big[\min(r - \text{MAR}, 0)^2\big]}
$$

**Value at Risk (VaR)** at confidence $\alpha$ -- the loss threshold not
expected to be exceeded on $(1-\alpha)$ of periods, i.e. the
$(1-\alpha)$-quantile of the loss distribution. Under a Gaussian assumption
with mean $\mu$, std $\sigma$:

$$
\text{VaR}_\alpha = -\big(\mu + \sigma \, \Phi^{-1}(1-\alpha)\big)
$$

**Conditional VaR / Expected Shortfall (CVaR)** -- the *average* loss given
that it exceeds VaR, strictly more informative about tail risk than VaR
alone. Under the same Gaussian assumption, using the standard truncated-
normal expectation:

$$
\text{CVaR}_\alpha = -\left(\mu - \sigma \, \frac{\phi(\Phi^{-1}(1-\alpha))}{1-\alpha}\right)
$$

where $\phi$ is the standard normal density. By construction,
$\text{CVaR}_\alpha \geq \text{VaR}_\alpha$ always (CVaR averages over the
tail region that VaR is merely the boundary of) -- verified numerically in
`risk.py` on simulated Gaussian data, alongside agreement between this
closed form and the empirical (historical-percentile) estimator.
"""))

nb.cells.extend(new_cells)
nbf.write(nb, NOTEBOOK_PATH)
print(f"Appended {len(new_cells)} cells. Notebook now has {len(nb.cells)} cells total.")