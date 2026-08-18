"""
append_quant_math.py

One-time script: appends the Black-Scholes PDE derivation to the existing
notebooks/math_derivations.ipynb (built earlier for the fraud-detection
math) rather than creating a second notebook -- matches the project spec's
single math_derivations.ipynb.
"""

import nbformat as nbf

PATH = "notebooks/math_derivations.ipynb"

nb = nbf.read(PATH, as_version=4)

new_cells = [
    nbf.v4.new_markdown_cell(
        "---\n\n# Part 2: Quantitative Trading -- Black-Scholes PINN\n\n"
        "## 6. Stochastic Model of the Underlying\n\n"
        "Stock price follows geometric Brownian motion:\n"
        "$$dS = \\mu S\\,dt + \\sigma S\\,dW$$\n\n"
        "where $W$ is a Wiener process. Applying **Ito's lemma** to a "
        "derivative price $V(S,t)$ (a function of a stochastic process, "
        "not an ordinary function -- this is where Ito's correction term "
        "differs from the classical chain rule):\n\n"
        "$$dV = \\left(\\frac{\\partial V}{\\partial t} + \\mu S\\frac{\\partial V}{\\partial S} "
        "+ \\frac{1}{2}\\sigma^2 S^2\\frac{\\partial^2 V}{\\partial S^2}\\right)dt "
        "+ \\sigma S\\frac{\\partial V}{\\partial S}\\,dW$$"
    ),
    nbf.v4.new_markdown_cell(
        "## 7. Delta-Hedging and the Black-Scholes PDE\n\n"
        "Construct a portfolio $\\Pi = V - \\Delta S$ with "
        "$\\Delta = \\partial V/\\partial S$, chosen to cancel the $dW$ term -- "
        "the portfolio becomes instantaneously riskless, so under "
        "no-arbitrage it must earn exactly the risk-free rate $r$: "
        "$d\\Pi = r\\Pi\\,dt$.\n\n"
        "Working through this with the Ito expansion above eliminates "
        "$\\mu$ entirely -- the drift of the underlying doesn't appear in "
        "the final PDE. This is the 'risk-neutral pricing' result: you can "
        "price as if $\\mu = r$. The result is the **Black-Scholes PDE**:\n\n"
        "$$\\frac{\\partial V}{\\partial t} + \\frac{1}{2}\\sigma^2 S^2"
        "\\frac{\\partial^2 V}{\\partial S^2} + rS\\frac{\\partial V}{\\partial S} - rV = 0$$\n\n"
        "with terminal condition (solved backward from expiry, not "
        "forward from an initial condition) for a European call:\n"
        "$$V(S,T) = \\max(S-K, 0)$$\n"
        "and boundary conditions $V(0,t) = 0$, $V(S,t) \\to S$ as $S \\to \\infty$."
    ),
    nbf.v4.new_markdown_cell(
        "## 8. Comparison to the Heat-Equation PINN\n\n"
        "The heat equation from the earlier project was "
        "$\\partial u/\\partial t = \\alpha\\,\\partial^2 u/\\partial x^2$ -- pure "
        "diffusion, constant coefficient. Black-Scholes has the same "
        "second-order diffusion term ($\\frac{1}{2}\\sigma^2 S^2 "
        "\\partial^2 V/\\partial S^2$), but three added complications:\n\n"
        "1. The diffusion coefficient $\\sigma^2 S^2$ is *state-dependent*, not constant\n"
        "2. A first-order convection term $rS\\,\\partial V/\\partial S$ (drift)\n"
        "3. A zeroth-order reaction term $-rV$ (discounting)\n\n"
        "It's a convection-diffusion-reaction PDE, not pure diffusion -- "
        "same PINN machinery (autograd computes the PDE residual as a loss "
        "term, alongside terminal/boundary condition losses), just a "
        "richer residual expression. This is what `pinn_bs.py` implements."
    ),
]

nb["cells"].extend(new_cells)

with open(PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Appended {len(new_cells)} cells to {PATH} (now {len(nb['cells'])} cells total)")