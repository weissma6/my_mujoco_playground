"""Replay archived W&B eval curves through the PatienceTracker.

This is a REPORT, not an assertion. Its job is to let the patience / min_delta
defaults be chosen against curves that really happened instead of guessed. Read
the output and sanity-check it before trusting the defaults on the cluster:

  * a stop pinned at the min_steps floor on a curve that was still climbing
    means `patience` is too small or `min_delta` too large;
  * "never stops" on every rung means the criterion will never fire and the
    ladder degenerates to a fixed 5 x 24M budget.

The curve is cached to JSON next to this file on first fetch, so re-runs work
offline and the numbers stay reproducible.

Run:
    python batch_runs/curriculum/tests/replay_real_curve.py
    python batch_runs/curriculum/tests/replay_real_curve.py --runs L0_none_vel_s1_20260729_104930_2201
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from learning.curriculum.early_stop import PatienceTracker  # noqa: E402

ENTITY = "weissma6-zhaw-school-of-engineering"
PROJECT = "UR3_pick_ppo"
METRIC = "eval/episode_reward"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_curve_cache.json")

# The five velocity DR-ladder rungs -- the closest existing analogue to what the
# curriculum will train, and the runs whose checkpoints are cached locally.
DEFAULT_RUNS = [
    "L0_none_vel_s1_20260729_104930_2201",
    "L1_pos_vel_s1_20260729_112044_2201",
    "L2_pos_cube_vel_s1_20260729_112246_2201",
    "L3_pos_cube_robot_vel_s1_20260729_115408_2201",
    "L4_full_vel_s1_20260729_122736_2201",
]


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def fetch_curve(run_id, cache):
    """Return [[step, reward], ...] for one run, from cache or W&B."""
    if run_id in cache:
        return cache[run_id]
    import wandb

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    # DO NOT use scan_history here. wandb's step index in this project IS the
    # env step, so the history spans ~25M and scan_history walks every one of
    # them -- measured at >15 min per run with no output. The sampled endpoint
    # returns the same points in about a second, and these runs only ever log
    # 20-40 eval rows anyway, so `samples` far above that loses nothing.
    hist = run.history(keys=[METRIC], samples=2000, pandas=False)
    pts = sorted(
        [int(row["_step"]), float(row[METRIC])]
        for row in hist
        if row.get(METRIC) is not None
    )
    cache[run_id] = pts
    return pts


def replay(pts, patience, min_delta, min_steps):
    t = PatienceTracker(patience=patience, min_delta=min_delta, min_steps=min_steps)
    for step, r in pts:
        if t.update(step, {METRIC: r}):
            return t, step
    return t, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=0.02)
    ap.add_argument("--min-steps", type=int, default=6_000_000)
    args = ap.parse_args()

    cache = load_cache()
    curves = {}
    for run_id in args.runs:
        try:
            curves[run_id] = fetch_curve(run_id, cache)
        except Exception as e:                      # noqa: BLE001 - report, do not crash
            print(f"[skip] {run_id}: {type(e).__name__}: {e}")
    save_cache(cache)

    if not curves:
        print("\nNo curves available. Is `wandb login` done on this machine?")
        return 1

    print(f"\n=== replay @ patience={args.patience} "
          f"min_delta={args.min_delta} min_steps={args.min_steps:,} ===\n")
    hdr = f"{'run':<46}{'evals':>6}{'budget':>12}{'stop@':>12}{'saved':>8}{'peak':>10}{'peak@':>12}"
    print(hdr)
    print("-" * len(hdr))
    for run_id, pts in curves.items():
        if not pts:
            print(f"{run_id:<46}{'(no eval points)':>50}")
            continue
        budget = pts[-1][0]
        t, stop = replay(pts, args.patience, args.min_delta, args.min_steps)
        saved = f"{100.0 * (1 - stop / budget):.0f}%" if stop else "--"
        stop_s = f"{stop:,}" if stop else "never"
        print(f"{run_id:<46}{len(pts):>6}{budget:>12,}{stop_s:>12}{saved:>8}"
              f"{t.max_reward:>10.0f}{t.max_step:>12,}")

    print("\n--- sensitivity: how many of the 5 rungs stop, and mean budget saved ---")
    print(f"{'patience':>9}{'min_delta':>11}{'stopped':>9}{'mean saved':>12}")
    for p in (3, 4, 5, 6, 8):
        for d in (0.01, 0.02, 0.05):
            n, savings = 0, []
            for pts in curves.values():
                if not pts:
                    continue
                _, stop = replay(pts, p, d, args.min_steps)
                if stop:
                    n += 1
                    savings.append(1 - stop / pts[-1][0])
            mean = f"{100.0 * sum(savings) / len(savings):.0f}%" if savings else "--"
            print(f"{p:>9}{d:>11}{n:>9}{mean:>12}")

    print("\nEval cadence: archived 31,948,800/39 = 819,200 steps per eval;")
    print("planned 24,000,000/29 = 827,586. Near-identical, so the resolution")
    print("above IS representative of production -- not a rough proxy.")
    print("\nBUT the `saved` column is against each run's own ~31.9M budget.")
    print("Truncated to the 24M production cap, a stop beyond 24M becomes a")
    print("full-budget run: at the defaults that is 3/5 rungs stopping and")
    print("~9% saved, not ~29%. These are also FROM-SCRATCH curves; warm-started")
    print("rungs should converge earlier, so treat 3/5 as a conservative floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
