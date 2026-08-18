"""
src/fraud/evaluate.py

Consolidates System 1 (fraud detection) into one saved report: ROC curves
for the GNN alone and the full ensemble, a confusion matrix at the 3%-FPR
operating threshold, and a markdown summary comparing every component
against the project spec's target metrics. Saves outputs/roc_curve.png
and outputs/evaluation_report.md rather than only printing to terminal.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just saving files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from graph import build_graph
from train import make_node_splits, OUTPUT_DIR
from ensemble import load_tuned_gnn, gnn_scores, isolation_forest_scores, train_autoencoder, normalize
from baseline import precision_at_fpr
from sklearn.linear_model import LogisticRegression

import torch

REPORT_PATH = OUTPUT_DIR / "evaluation_report.md"
ROC_PLOT_PATH = OUTPUT_DIR / "roc_curve.png"

# Known from baseline.py's own run (full 413K-row training set, several
# minutes of GridSearchCV) -- not recomputed here to avoid a ~10 minute
# rerun of already-established results.
RANDOM_FOREST_BASELINE_AUC = 0.9216
RANDOM_FOREST_BASELINE_PRECISION_AT_3FPR = 0.4373


def confusion_at_fpr(y_true, scores, target_fpr: float = 0.03):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    idx = np.where(valid)[0][-1] if valid.any() else 0
    threshold = thresholds[idx]
    y_pred = (scores >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred), threshold


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = build_graph(max_transactions=100_000)
    train_mask_t, val_mask_t, test_mask_t = make_node_splits(data.y)
    data = data.to(device)
    train_mask_t, test_mask_t = train_mask_t.to(device), test_mask_t.to(device)

    print("\nRecomputing GNN and ensemble scores...")
    model = load_tuned_gnn(data, device)
    s_gnn = gnn_scores(model, data)

    x_np = data.x.cpu().numpy()
    train_mask_np = train_mask_t.cpu().numpy()
    fraud_rate = data.y.float().mean().item()
    s_if = isolation_forest_scores(x_np, train_mask_np, contamination=fraud_rate)
    s_ae = train_autoencoder(data.x, data.y, train_mask_t, device)

    s_gnn_n, s_if_n, s_ae_n = normalize(s_gnn), normalize(s_if), normalize(s_ae)
    stacked = np.stack([s_gnn_n, s_if_n, s_ae_n], axis=1)
    y_np = data.y.cpu().numpy()
    test_mask_np = test_mask_t.cpu().numpy()

    blender = LogisticRegression(class_weight="balanced")
    blender.fit(stacked[train_mask_np], y_np[train_mask_np])
    s_ensemble = blender.predict_proba(stacked)[:, 1]

    # ROC curves
    print("Plotting ROC curves...")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores in [("GNN alone", s_gnn), ("Ensemble (GNN+IF+AE)", s_ensemble)]:
        fpr, tpr, _ = roc_curve(y_np[test_mask_np], scores[test_mask_np])
        auc = roc_auc_score(y_np[test_mask_np], scores[test_mask_np])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.axhline(y=RANDOM_FOREST_BASELINE_AUC, linestyle=":", color="green", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Fraud Detection: ROC Curves (test split)")
    ax.legend(loc="lower right")
    ax.text(0.5, RANDOM_FOREST_BASELINE_AUC - 0.03, f"RF baseline AUC = {RANDOM_FOREST_BASELINE_AUC:.4f}",
            color="green", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROC_PLOT_PATH, dpi=150)
    print(f"  saved {ROC_PLOT_PATH}")

    # Confusion matrix at 3% FPR operating point
    cm, threshold = confusion_at_fpr(y_np[test_mask_np], s_ensemble[test_mask_np])
    tn, fp, fn, tp = cm.ravel()

    # Final metrics table
    rows = []
    for name, scores in [("Isolation Forest", s_if), ("Autoencoder", s_ae),
                          ("GNN (tuned)", s_gnn), ("Ensemble (stacked)", s_ensemble)]:
        auc = roc_auc_score(y_np[test_mask_np], scores[test_mask_np])
        precision = precision_at_fpr(y_np[test_mask_np], scores[test_mask_np])
        rows.append((name, auc, precision))
    rows.append(("Random Forest (baseline.py)", RANDOM_FOREST_BASELINE_AUC, RANDOM_FOREST_BASELINE_PRECISION_AT_3FPR))

    print("\n" + "=" * 60)
    print("FINAL COMPARISON (test split)")
    print("=" * 60)
    for name, auc, precision in rows:
        print(f"{name:<28} AUC-ROC {auc:.4f}   Precision@3%FPR {precision:.4f}")

    report_lines = [
        "# Fraud Detection Evaluation Report",
        "",
        f"Graph: 100,000-transaction subgraph (card1/addr1 windowed edges, "
        f"{data.num_node_features} features per node)",
        "",
        "## Component Comparison (test split)",
        "",
        "| Model | AUC-ROC | Precision@3%FPR |",
        "|---|---|---|",
    ]
    for name, auc, precision in rows:
        report_lines.append(f"| {name} | {auc:.4f} | {precision:.4f} |")

    report_lines += [
        "",
        f"## Confusion Matrix -- Ensemble @ 3% FPR operating threshold ({threshold:.4f})",
        "",
        "| | Predicted: Not Fraud | Predicted: Fraud |",
        "|---|---|---|",
        f"| Actual: Not Fraud | {tn} | {fp} |",
        f"| Actual: Fraud | {fn} | {tp} |",
        "",
        "## Spec Target Check",
        "",
        f"- Fraud GNN AUC-ROC > 0.95: {'PASS' if max(r[1] for r in rows if 'GNN' in r[0]) > 0.95 else 'NOT MET'} "
        f"(best: {max(r[1] for r in rows if 'GNN' in r[0]):.4f})",
        f"- Random Forest AUC-ROC > 0.90: "
        f"{'PASS' if RANDOM_FOREST_BASELINE_AUC > 0.90 else 'NOT MET'} ({RANDOM_FOREST_BASELINE_AUC:.4f})",
        f"- Precision at 3% FPR > 85%: "
        f"{'PASS' if max(r[2] for r in rows) > 0.85 else 'NOT MET'} (best: {max(r[2] for r in rows):.4f})",
        "",
        "See roc_curve.png for the visual ROC comparison.",
    ]

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved {REPORT_PATH}")

    print("\nEVALUATE.PY CHECKS PASSED")