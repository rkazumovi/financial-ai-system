"""
build_math_notebook.py

One-time generator for notebooks/math_derivations.ipynb.
Run once with `python build_math_notebook.py`, then delete this file if you like
-- the notebook it produces is the artifact that matters, not this script.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell(
        "# Mathematical Derivations\n\n"
        "Foundations for the fraud-detection GNN and, later, the quant PINN "
        "components of the financial-ai-system project.\n\n"
        "## 1. The Transaction Graph\n\n"
        "Accounts/cards as nodes $V$, $|V| = n$, transactions as edges $E$.\n\n"
        "- Node feature matrix: $X \\in \\mathbb{R}^{n \\times F}$\n"
        "- Adjacency matrix: $A \\in \\{0,1\\}^{n \\times n}$ (or weighted by amount/frequency)\n"
        "- Degree matrix: $D = \\mathrm{diag}(d_1,\\dots,d_n)$, $d_i = \\sum_j A_{ij}$\n"
        "- Graph Laplacian: $L = D - A$, normalized $L_{sym} = I - D^{-1/2} A D^{-1/2}$\n\n"
        "$L$ is symmetric PSD for an undirected graph, so it eigendecomposes as "
        "$L = U\\Lambda U^\\top$ with $0 = \\lambda_1 \\le \\cdots \\le \\lambda_n$."
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Spectral Graph Theory and the Graph Fourier Transform\n\n"
        "The graph Fourier transform of a signal $x$ is $\\hat{x} = U^\\top x$. "
        "$U$'s columns play the role of complex exponentials in classical Fourier "
        "analysis; $\\lambda_i$ plays the role of frequency -- low $\\lambda$ means "
        "smooth across the graph, high $\\lambda$ means rapidly varying between "
        "neighbors.\n\n"
        "Spectral graph convolution:\n"
        "$$x *_G g = U\\big((U^\\top x) \\odot (U^\\top g)\\big)$$\n\n"
        "the graph analogue of the classical convolution theorem.\n\n"
        "**Problem:** computing $U$ is $O(n^3)$, not localized, and transductive "
        "(breaks on unseen nodes -- a dealbreaker when new accounts appear "
        "constantly in a live fraud system).\n\n"
        "GCN (Kipf & Welling, 2017) approximates the spectral filter with a "
        "first-order polynomial of $L$, avoiding the eigendecomposition entirely:\n"
        "$$H' = \\sigma\\big(\\tilde{D}^{-1/2}\\tilde{A}\\tilde{D}^{-1/2}HW\\big)$$\n\n"
        "but the neighbor weights $\\tilde{D}^{-1/2}\\tilde{A}\\tilde{D}^{-1/2}$ here "
        "are fixed by graph structure alone -- not learned, no notion of "
        "'this neighbor matters more.'"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Graph Attention Networks (GAT)\n\n"
        "GAT replaces fixed structural weighting with learned, data-dependent "
        "attention -- no eigendecomposition needed, inductive, and the attention "
        "weights are directly interpretable (feeds GNNExplainer for regulatory "
        "compliance).\n\n"
        "For node $i$ with neighborhood $\\mathcal{N}(i)$:\n\n"
        "$$z_i = W h_i$$\n"
        "$$e_{ij} = \\mathrm{LeakyReLU}\\big(a^\\top [z_i \\,\\|\\, z_j]\\big)$$\n"
        "$$\\alpha_{ij} = \\frac{\\exp(e_{ij})}{\\sum_{k \\in \\mathcal{N}(i)} \\exp(e_{ik})}$$\n"
        "$$h_i' = \\sigma\\Big(\\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij} z_j\\Big)$$\n\n"
        "$W \\in \\mathbb{R}^{F' \\times F}$ and $a \\in \\mathbb{R}^{2F'}$ are learned. "
        "The softmax runs only over $i$'s actual neighbors -- that's where graph "
        "structure enters. With $K$ attention heads, outputs are concatenated on "
        "hidden layers and averaged on the final layer."
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Temporal Extension\n\n"
        "Plain GAT is order-blind. Two options to fix this:\n\n"
        "1. Encode time into the attention logit -- append a learnable "
        "time-encoding to $[z_i \\| z_j]$ before the LeakyReLU.\n"
        "2. Run GAT per time-window snapshot, feed the resulting node-embedding "
        "sequence into a GRU.\n\n"
        "Architecture choice to be finalized in `src/fraud/model.py`."
    ),
    nbf.v4.new_markdown_cell(
        "## 5. Physics-Inspired Flow Conservation\n\n"
        "For an account that is neither a cash-in source nor a cash-out sink, "
        "inflow should approximately equal outflow over a time window -- the same "
        "continuity-equation structure as $\\partial \\rho/\\partial t + "
        "\\nabla \\cdot J = 0$, or Kirchhoff's current law.\n\n"
        "Soft regularizer added to the training loss:\n"
        "$$L_{flow} = \\sum_{i} \\Big(\\sum_{j \\to i} \\text{amt}_{ji} - "
        "\\sum_{i \\to k} \\text{amt}_{ik}\\Big)^2$$\n\n"
        "Same pattern as adding a PDE residual to the loss in the heat-equation "
        "PINN project -- here the 'PDE' is a conservation law over the "
        "transaction graph instead of physical space."
    ),
]

nb["cells"] = cells

os.makedirs("notebooks", exist_ok=True)
out_path = os.path.join("notebooks", "math_derivations.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} with {len(cells)} cells")

if __name__ == "__main__":
    pass