"""Train-vs-sim-vs-real reward comparison grid (Task 2).

One PNG: one policy per row, three columns:
  [ W&B training curve | 10 MuJoCo rollouts | 10 real-robot runs ]

Column 1 (Training) is the classic W&B learning curve: episode return vs
training step, mean +/- std band (mirrors evaluation/Show_Graphs.ipynb's
reward_comparison plot style). Columns 2-3 are per-step TOTAL reward vs
EPISODE step for individual rollouts/runs -- a different x-axis than column
1. The two are made comparable via:
  - a dashed horizontal reference line at the training run's converged
    (final) episode return,
  - a per-rollout/run episode-sum annotation,
so the sim-to-real reward gap is readable across the row without pretending
the axes are the same thing. Pass --cumulative to instead plot CUMULATIVE
reward vs episode step in columns 2-3, which converges toward the same
episode-return quantity column 1 plots (closer axes, noisier read of the
per-step shape) -- off by default.

Column 2 (MuJoCo) calls ur3_reward_replay.sim_rollout_reward, which builds
and steps an MJX env. Per CLAUDE.md, MJX must NEVER run on this (Matthias's)
local machine -- no GPU, and CPU MJX OOM-kills it. Run this script on the
ZHAW HPC or the robot PC, never here.

Column 3 (Real) reads the ur3_pick_reward.csv files written by
robots/UR3e/ur3_realrobot_pickloop.py (one per run, under
real_robot_results/{policy_name}/{run_stamp}/) -- collect >= N_REAL real
runs per policy before running this.

Usage:
    python evaluation/compare_reward_train_sim_real.py [--cumulative]
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ur3_reward_replay import sim_rollout_reward  # noqa: E402  (MJX -- HPC/robot PC only)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# Config -- edit per comparison run.
# ===========================================================================

WANDB_ENTITY = "weissma6-zhaw-school-of-engineering"
WANDB_PROJECT = "UR3_pick_ppo"

N_SIM = 10
N_REAL = 10

# One row per policy. `real_runs_glob` defaults to the per-run folder layout
# real_robot_results/{name}/*/ur3_pick_reward.csv (see the FOLDER_OUT change
# in ur3_realrobot_pickloop.py) if left as None.
POLICY_ROWS = [
    {
        "name": "EP200_AS0.025_78fd8855a6eeb72d8ef1a0395cd6e4860db1439d",
        "run_id": "EP200_AS0.025_d0.99_20260703_132735_2201",
        "real_runs_glob": None,
    },
    # Add more policies here, e.g.:
    # {"name": "AGATE250_posemid_easy_offOFF_...", "run_id": "<wandb run id>",
    #  "real_runs_glob": None},
]

OUT = os.path.join(EVAL_DIR, "reward_train_sim_real.png")

_MK_FORMATTER = plt.FuncFormatter(
    lambda x, _: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"
)


# ===========================================================================
# Column 1 -- training curve (W&B).
# ===========================================================================


def fetch_training_curve(run_id: str, entity: str = WANDB_ENTITY,
                          project: str = WANDB_PROJECT) -> dict:
    """Returns {"steps", "mean", "std", "final_return"} or None on failure."""
    import wandb

    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        hist = run.history(
            keys=["eval/episode_reward", "eval/episode_reward_std",
                  "training/num_steps"],
            pandas=True,
        ).dropna(subset=["eval/episode_reward"])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] failed to fetch W&B history for {run_id}: {e}")
        return None
    if len(hist) == 0:
        print(f"[warn] no eval/episode_reward history for {run_id}")
        return None
    steps = hist["training/num_steps"] if "training/num_steps" in hist.columns else hist["_step"]
    mean = hist["eval/episode_reward"]
    std = hist["eval/episode_reward_std"] if "eval/episode_reward_std" in hist.columns else pd.Series(np.zeros(len(mean)))
    return {
        "steps": steps.to_numpy(dtype=float),
        "mean": mean.to_numpy(dtype=float),
        "std": std.to_numpy(dtype=float),
        "final_return": float(mean.iloc[-1]),
    }


def plot_training_column(ax, curve: dict, title: str):
    if curve is None:
        ax.text(0.5, 0.5, "no W&B data", ha="center", va="center",
                 transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(title)
        return
    ax.plot(curve["steps"], curve["mean"], color="#2E86AB", linewidth=2)
    ax.fill_between(curve["steps"], curve["mean"] - curve["std"],
                     curve["mean"] + curve["std"], color="#2E86AB", alpha=0.25)
    ax.xaxis.set_major_formatter(_MK_FORMATTER)
    ax.set_xlabel("Environment steps")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)


# ===========================================================================
# Column 2 -- MuJoCo rollouts (MJX -- HPC/robot PC only, see module docstring).
# ===========================================================================


def collect_sim_rollouts(policy_dir: str, n_sim: int = N_SIM,
                          xml_path: str = None) -> list:
    """Returns a list of per-rollout reward DataFrames (from
    sim_rollout_reward). MJX -- do not call this from the local machine.
    """
    rollouts = []
    for seed in range(n_sim):
        try:
            rollouts.append(sim_rollout_reward(policy_dir, seed, xml_path=xml_path))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] sim rollout seed={seed} failed: {e}")
    return rollouts


def plot_sim_column(ax, rollouts: list, final_return: float, cumulative: bool,
                     title: str):
    if not rollouts:
        ax.text(0.5, 0.5, "no MuJoCo rollouts\n(run on HPC/robot PC)",
                 ha="center", va="center", transform=ax.transAxes,
                 fontsize=10, color="gray")
        ax.set_title(title)
        return
    sums = []
    max_len = max(len(r) for r in rollouts)
    stacked = np.full((len(rollouts), max_len), np.nan)
    for i, r in enumerate(rollouts):
        y = r["reward_total"].to_numpy(dtype=float)
        if cumulative:
            y = np.cumsum(y)
        stacked[i, : len(y)] = y
        ax.plot(r["step"], y, color="#F18F01", alpha=0.35, linewidth=1.0)
        sums.append(float(r["reward_total"].sum()))
    mean_curve = np.nanmean(stacked, axis=0)
    ax.plot(range(max_len), mean_curve, color="#F18F01", linewidth=2.5,
             label="mean")
    if final_return is not None:
        ax.axhline(final_return, linestyle="--", color="black", alpha=0.4,
                     label="train final return")
    ax.set_xlabel("Episode step")
    ax.set_title(
        f"{title}\nepisode-sum {np.mean(sums):.0f} ± {np.std(sums):.0f}",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)


# ===========================================================================
# Column 3 -- real-robot runs.
# ===========================================================================


def collect_real_runs(name: str, real_runs_glob: str = None,
                       n_real: int = N_REAL) -> list:
    """Returns a list of (reward_df, stop_reason) for the N_REAL newest
    logged real runs matching `real_runs_glob` (default:
    real_robot_results/{name}/*/ur3_pick_reward.csv).
    """
    if real_runs_glob is None:
        real_runs_glob = os.path.join(
            REPO_ROOT, "robots", "UR3e", "real_robot_results", name, "*",
            "ur3_pick_reward.csv",
        )
    paths = sorted(glob.glob(real_runs_glob), key=os.path.getmtime, reverse=True)
    paths = paths[:n_real]
    if len(paths) < n_real:
        print(f"[warn] only found {len(paths)}/{n_real} real runs for "
              f"'{name}' matching {real_runs_glob}")

    runs = []
    for csv_path in paths:
        try:
            reward_df = pd.read_csv(csv_path)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to read {csv_path}: {e}")
            continue
        stop_reason = "unknown"
        meta_path = os.path.join(os.path.dirname(csv_path), "ur3_pick_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                stop_reason = meta.get("stats", {}).get("stopped_reason", "unknown")
            except Exception:  # noqa: BLE001
                pass
        runs.append((reward_df, stop_reason))
    return runs


def plot_real_column(ax, runs: list, final_return: float, cumulative: bool,
                      title: str):
    if not runs:
        ax.text(0.5, 0.5, "no real runs logged yet", ha="center", va="center",
                 transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(title)
        return
    sums = []
    max_len = max(len(r) for r, _ in runs)
    stacked = np.full((len(runs), max_len), np.nan)
    for i, (r, stop_reason) in enumerate(runs):
        y = r["reward_total"].to_numpy(dtype=float)
        if cumulative:
            y = np.cumsum(y)
        stacked[i, : len(y)] = y
        x = r["step"] if "step" in r.columns else np.arange(len(y))
        ax.plot(x, y, color="#A23B72", alpha=0.35, linewidth=1.0)
        if len(y) > 0:
            ax.scatter([x.iloc[-1] if hasattr(x, "iloc") else x[-1]], [y[-1]],
                       s=12, color="#A23B72", alpha=0.6)
        sums.append(float(r["reward_total"].sum()))
    mean_curve = np.nanmean(stacked, axis=0)
    ax.plot(range(max_len), mean_curve, color="#A23B72", linewidth=2.5,
             label="mean")
    if final_return is not None:
        ax.axhline(final_return, linestyle="--", color="black", alpha=0.4,
                     label="train final return")
    stop_reasons = [sr for _, sr in runs]
    reason_counts = pd.Series(stop_reasons).value_counts().to_dict()
    ax.set_xlabel("Episode step")
    ax.set_title(
        f"{title}\nepisode-sum {np.mean(sums):.0f} ± {np.std(sums):.0f}  "
        f"({reason_counts})",
        fontweight="bold", fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)


# ===========================================================================
# Main
# ===========================================================================


def main(cumulative: bool = False):
    import sys
    sys.path.insert(0, os.path.join(EVAL_DIR, "downloaded_policies"))
    from policy_downloader import default_policy_dir, download_policy

    n = len(POLICY_ROWS)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False)
    col_titles = ["Training (W&B)", f"MuJoCo ({N_SIM} rollouts)",
                  f"Real robot ({N_REAL} runs)"]
    for j, ct in enumerate(col_titles):
        axes[0, j].annotate(ct, xy=(0.5, 1.15), xycoords="axes fraction",
                             ha="center", fontsize=13, fontweight="bold")

    for i, row_cfg in enumerate(POLICY_ROWS):
        name, run_id = row_cfg["name"], row_cfg["run_id"]
        print(f"\n=== policy {i + 1}/{n}: {name} ===")

        curve = fetch_training_curve(run_id)
        final_return = curve["final_return"] if curve else None
        plot_training_column(axes[i, 0], curve, name if i == 0 else "")
        axes[i, 0].set_ylabel(name, fontsize=9)

        policy_dir = download_policy(run_id, out_dir=default_policy_dir(run_id))
        rollouts = collect_sim_rollouts(policy_dir, N_SIM)
        plot_sim_column(axes[i, 1], rollouts, final_return, cumulative, "")

        real_runs = collect_real_runs(name, row_cfg.get("real_runs_glob"), N_REAL)
        plot_real_column(axes[i, 2], real_runs, final_return, cumulative, "")

    fig.text(
        0.5, -0.02 / n,
        "Col 1 = episode return vs training step (learning curve). "
        "Cols 2-3 = per-step reward vs episode step for individual "
        "rollouts/runs; dashed line = training's converged episode return; "
        "episode-sum annotation is the per-run total for direct comparison.",
        ha="center", fontsize=8, color="gray",
    )
    plt.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cumulative", action="store_true",
        help="Plot cumulative reward vs episode step in columns 2-3 instead "
             "of per-step reward (closer to column 1's axis, at the cost of "
             "a noisier per-step read).",
    )
    args = parser.parse_args()
    main(cumulative=args.cumulative)
