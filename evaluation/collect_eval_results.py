"""
Collect eval_results.json files from eval_results/<eval_id>/ and plot the
comparison grid: rows = policies, columns = conditions.

Usage:
  python collect_eval_results.py --results_dir eval_results --output eval_grid.pdf
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="eval_results")
    ap.add_argument("--output", default="eval_grid.pdf")
    args = ap.parse_args()

    # ── Load all result JSONs ──
    all_results = []
    for entry in sorted(os.listdir(args.results_dir)):
        path = os.path.join(args.results_dir, entry, "eval_results.json")
        if os.path.isfile(path):
            with open(path) as f:
                all_results.append(json.load(f))

    if not all_results:
        print("No results found.")
        return

    # ── Organize into grid: results_grid[policy][condition] = rewards ──
    results_grid = defaultdict(dict)
    for r in all_results:
        policy = r["policy_run_id"]
        condition = r["condition"]
        results_grid[policy][condition] = np.array(r["rewards"])

    policy_labels = list(results_grid.keys())
    cond_labels = sorted(
        {r["condition"] for r in all_results},
        key=lambda c: ["Default", "Mass", "Friction"].index(c)
        if c in ["Default", "Mass", "Friction"] else 99,
    )

    # ── Plot ──
    n_rows = len(policy_labels)
    n_cols = len(cond_labels)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    all_rewards = [
        results_grid[p][c]
        for p in policy_labels for c in cond_labels
        if c in results_grid[p]
    ]
    global_min = min(r.min() for r in all_rewards)
    global_max = max(r.max() for r in all_rewards)
    pad = (global_max - global_min) * 0.1
    bins = np.linspace(global_min - pad, global_max + pad, 20)

    for row, policy in enumerate(policy_labels):
        for col, cond in enumerate(cond_labels):
            ax = axes[row][col]
            if cond not in results_grid[policy]:
                ax.set_visible(False)
                continue

            rews = results_grid[policy][cond]
            ax.hist(rews, bins=bins, color="#2E86AB", edgecolor="white", alpha=0.8)
            ax.axvline(rews.mean(), color="red", linestyle="--", linewidth=2,
                       label=f"μ={rews.mean():.1f}")
            ax.axvline(rews.mean() - rews.std(), color="gray", linestyle=":", linewidth=1.2)
            ax.axvline(rews.mean() + rews.std(), color="gray", linestyle=":", linewidth=1.2,
                       label=f"σ={rews.std():.1f}")
            ax.legend(fontsize=8)

            if row == 0:
                ax.set_title(cond, fontsize=13, fontweight="bold")
            if col == 0:
                # Shorten policy label for y-axis
                short = policy.split("_20")[0] if "_20" in policy else policy[:30]
                ax.set_ylabel(short, fontsize=10, fontweight="bold")
            if row == n_rows - 1:
                ax.set_xlabel("Episode Reward", fontsize=10)

    fig.suptitle(f"Robustness Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", dpi=300)
    print(f"✓ Saved to {args.output}")

    # ── Print summary table ──
    print(f"\n{'Policy':<45} {'Condition':<15} {'Mean':>8} {'Std':>8}")
    print("-" * 80)
    for policy in policy_labels:
        for cond in cond_labels:
            if cond in results_grid[policy]:
                rews = results_grid[policy][cond]
                short = policy.split("_20")[0] if "_20" in policy else policy[:40]
                print(f"{short:<45} {cond:<15} {rews.mean():>8.1f} {rews.std():>8.1f}")


if __name__ == "__main__":
    main()
