"""
src/fraud/tune.py

Systematic hyperparameter search for FraudGAT, in place of one-off manual
graph-construction tweaks. Builds the graph ONCE (graph structure doesn't
depend on model hyperparameters) and reuses it across every trial, then
retrains the winning configuration once more with full logging to produce
the final outputs/fraud_gat_best.pt checkpoint.

Grid covers model capacity (layers, hidden width, attention heads) and
regularization (dropout) -- the axes most likely to matter given the
0.7425-0.7851 test AUC range from manual graph tuning, still well below
the 0.9216 Random Forest baseline. 2x2x2x2 = 16 configs is a deliberately
modest grid, same spirit as baseline.py's 2x2 Random Forest grid, not an
exhaustive search.

Each trial explicitly clears CUDA's cached (but unreturned) memory
afterward, and a single trial running out of GPU memory is skipped rather
than crashing the whole search -- the largest config (3 layers, 128
hidden channels, 8 heads) on an 8GB laptop GPU is the most likely to hit
this.
"""

import gc
import itertools
import time

import torch

from graph import build_graph
from train import train, make_node_splits, OUTPUT_DIR, CHECKPOINT_PATH

GRID = {
    "num_layers": [2, 3],
    "hidden_channels": [64, 128],
    "heads": [4, 8],
    "dropout": [0.2, 0.5],
}

TRIAL_CHECKPOINT = OUTPUT_DIR / "tune_trial.pt"


if __name__ == "__main__":
    print("Building graph once, reused across every trial...")
    data = build_graph(max_transactions=100_000)
    train_mask, val_mask, test_mask = make_node_splits(data.y)

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"\nSearching {len(combos)} configurations...\n")

    results = []
    for i, combo in enumerate(combos, 1):
        config = dict(zip(keys, combo))
        t0 = time.time()
        try:
            result = train(
                data=data,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                checkpoint_path=TRIAL_CHECKPOINT,
                max_epochs=60,   # shorter budget per trial during search
                patience=10,
                verbose=False,
                **config,
            )
            elapsed = time.time() - t0
            results.append(result)
            print(f"[{i}/{len(combos)}] {config}  "
                  f"val AUC {result['val_auc']:.4f}  test AUC {result['test_auc']:.4f}  ({elapsed:.0f}s)")
        except torch.OutOfMemoryError:
            elapsed = time.time() - t0
            print(f"[{i}/{len(combos)}] {config}  SKIPPED -- CUDA out of memory ({elapsed:.0f}s)")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("Every trial failed -- nothing to report")

    results.sort(key=lambda r: r["val_auc"], reverse=True)

    print("\n" + "=" * 70)
    print("LEADERBOARD (top 5, sorted by val AUC)")
    print("=" * 70)
    for r in results[:5]:
        print(f"val {r['val_auc']:.4f}  test {r['test_auc']:.4f}  {r['config']}")

    best_config = results[0]["config"]
    print(f"\nBest config: {best_config}")
    print("Retraining winner with full epoch budget and logging...")

    final_result = train(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        checkpoint_path=CHECKPOINT_PATH,
        max_epochs=150,
        patience=20,
        verbose=True,
        **best_config,
    )

    print("\n" + "=" * 50)
    print("TUNE.PY FINAL RESULT")
    print("=" * 50)
    print(f"FraudGAT (tuned)   test AUC {final_result['test_auc']:.4f}   "
          f"test Precision@3%FPR {final_result['test_precision']:.4f}")
    print("vs Random Forest baseline: 0.9216")
    print("vs best manual graph-tuning result: 0.7851")

    print("\nTUNE.PY CHECKS PASSED")