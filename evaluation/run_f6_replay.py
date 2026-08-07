"""Batch dynamics-replay (F6) over every COMPLETE real gap-protocol run.

Plain MuJoCo mj_step only (evaluation/ur3_dynamics_replay.py) -- no MJX, no
robot, safe locally per CLAUDE.md.

Why batch rather than one example run
-------------------------------------------------------------------------
A single run's F6 panel shows THAT the model diverges. Aggregating over all
complete runs supports a stronger claim for the results section:

  E_replay / time-to-divergence measure the MODEL-vs-ROBOT dynamics gap. That
  is a property of the (MuJoCo model, UR3e) pair -- the replay drives the sim
  with the LOGGED servoJ targets, so the policy never acts in the loop. If
  these numbers come out roughly flat across the DR ladder while the RETURN
  gap (F1) varies strongly with DR level, then DR is not closing the dynamics
  gap at all; it is changing how much the policy's return depends on it.
  Dynamics fidelity and policy robustness are separable, and only the second
  responds to domain randomization.

That decoupling is checkable from this script's summary table (see the
per-config aggregate printed at the end + f6_replay_summary.csv).

Aborted runs are EXCLUDED: they stop mid-episode, so their open-loop horizon
(and therefore time-to-divergence) is truncated by the abort rather than by
model error, which would bias the aggregate downward. Only stop_reason ==
"complete" runs are replayed. The count of excluded runs is printed (no silent
truncation).

Usage:
    python evaluation/run_f6_replay.py [--limit N] [--out_dir evaluation/gap_plots]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import gap_metrics as gm  # noqa: E402
from evaluation import plots_gap as pg  # noqa: E402
from evaluation.ur3_dynamics_replay import replay_real_run  # noqa: E402

DEFAULT_REAL_ROOT = os.path.join(
    REPO_ROOT, "robots", "UR3e", "real_robot_results", "gap_protocol")
DEFAULT_OUT_DIR = os.path.join(_THIS_DIR, "gap_plots")
PROTOCOL_ID = "gap_v1"
LEVELS = ["L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot", "L4_full"]


def _iter_complete_runs(real_root, protocol_id):
    """Yield (config_id, cube_size, leaf_dir, meta) for COMPLETE D23 real runs.

    stop_reason is read from each run's own ur3_pick_meta.json (never inferred
    from the folder name -- real leaf names are inconsistent about the
    condition-letter prefix).

    CAMPAIGN FILTER (same rule as gap_metrics.build_long_csv): only runs whose
    policy_run_id matches gap_protocol_policy_map.json are kept. The tree also
    holds the superseded 2026-07-22 campaign (different policies, 26-D obs,
    stale table geometry); replaying those would mix two robots' worth of
    model-fidelity data into one aggregate.
    """
    policy_map = gm.load_policy_map()
    if not policy_map:
        print("[f6] WARNING: no policy map -- campaign filter OFF, aggregate "
              "may mix the superseded 2026-07-22 campaign with D23.")
    n_wrong_policy = 0
    found = []
    for config_id in sorted(os.listdir(real_root)):
        proto = os.path.join(real_root, config_id, protocol_id)
        if not os.path.isdir(proto):
            continue
        for leaf in sorted(os.listdir(proto)):
            leaf_dir = os.path.join(proto, leaf)
            meta_p = os.path.join(leaf_dir, "ur3_pick_meta.json")
            if not os.path.isfile(meta_p):
                continue
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("stop_reason") != "complete":
                continue
            want = policy_map.get(meta["config_id"]) if policy_map else None
            if want and meta.get("policy_run_id") != want:
                n_wrong_policy += 1
                continue
            found.append((config_id, meta.get("cube_size"), leaf_dir, meta))
    if n_wrong_policy:
        print(f"[f6] campaign filter dropped {n_wrong_policy} complete runs "
              f"(policy_run_id not in gap_protocol_policy_map.json)")

    # Same superseded-re-run dedup as gap_metrics.build_long_csv, so F6's per-
    # config n matches the rest of the analysis: L0_none episodes 3-5 were run
    # twice on 2026-07-29 (before/after the condition-prefix naming). Keep the
    # condition-prefixed (canonical, later) execution.
    by_key = {}
    for item in found:
        _cfg, cube, leaf_dir, meta = item
        by_key.setdefault(
            (meta["config_id"], meta["episode_id"], meta["repeat"], cube),
            []).append(item)
    n_superseded = 0
    for key, cands in by_key.items():
        if len(cands) > 1:
            cands.sort(key=lambda it: (
                os.path.basename(it[2]).startswith("ep"),
                it[3].get("timestamp") or ""))
            n_superseded += len(cands) - 1
        yield cands[0]
    if n_superseded:
        print(f"[f6] dropped {n_superseded} superseded re-runs "
              f"(kept the condition-prefixed execution)")


def f6_aggregate(summary, out_path, levels=LEVELS):
    """Aggregate F6 across runs: per-config one-step E_replay and
    time-to-divergence. This is the figure that carries the decoupling
    argument (see module docstring) -- a single run cannot.
    """
    levels = [c for c in levels if c in set(summary["config_id"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    data_e = [summary.loc[summary.config_id == c,
                          "E_replay_onestep_joint_rad2"].to_numpy()
              for c in levels]
    bp1 = ax1.boxplot(data_e, tick_labels=levels, showmeans=True,
                       patch_artist=True)
    for i, box in enumerate(bp1["boxes"]):
        box.set_facecolor(pg._config_color(i))
        box.set_alpha(0.7)
    ax1.set_yscale("log")
    ax1.set_ylabel("E_replay one-step (joint, rad^2)")
    ax1.set_title("One-step dynamics-replay error by DR level",
                  fontweight="bold", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

    data_t = [summary.loc[summary.config_id == c,
                          "time_to_divergence_step"].dropna().to_numpy()
              for c in levels]
    bp2 = ax2.boxplot(data_t, tick_labels=levels, showmeans=True,
                       patch_artist=True)
    for i, box in enumerate(bp2["boxes"]):
        box.set_facecolor(pg._config_color(i))
        box.set_alpha(0.7)
    ax2.set_ylabel("time-to-divergence (steps, >1cm open-loop TCP err)")
    ax2.set_title("Open-loop time-to-divergence by DR level",
                  fontweight="bold", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    # NOTE (caption correctness): the replay is driven by the run's LOGGED
    # servoJ targets, so the policy is not in the FEEDBACK loop -- but those
    # targets were produced by that policy, so the replay measures model
    # fidelity ALONG THE STATES THAT POLICY VISITED. It is not a
    # policy-independent measurement, and the caption must not claim to be.
    fig.suptitle(
        "F6b -- dynamics-replay fidelity across the DR ladder (real runs, "
        "complete episodes only)\n"
        "Replay is driven by each run's LOGGED servoJ targets: the policy is "
        "not in the feedback loop, so this isolates MODEL-vs-ROBOT error -- "
        "but along the states that policy visited (not policy-independent).",
        fontweight="bold", fontsize=10, y=1.04,
    )
    plt.tight_layout()
    return pg._savefig(fig, out_path)


def run(real_root, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    runs = list(_iter_complete_runs(real_root, PROTOCOL_ID))
    n_total = len(runs)
    if limit:
        runs = runs[:limit]
        print(f"[f6] --limit {limit}: replaying {len(runs)} of {n_total} "
              f"complete runs (REST DROPPED -- summary is not the full set)")
    print(f"[f6] replaying {len(runs)} complete real runs "
          f"(aborted runs excluded by design -- see module docstring)")

    rows, per_step_cache = [], {}
    for i, (config_id, cube_size, leaf_dir, meta) in enumerate(runs, 1):
        try:
            s = replay_real_run(leaf_dir, command_mode="ctrl")
        except Exception as e:  # noqa: BLE001 - one bad run must not kill the batch
            print(f"  [{i}/{len(runs)}] FAILED {os.path.basename(leaf_dir)}: {e}")
            continue
        rows.append(dict(
            config_id=config_id, cube_size=cube_size,
            episode_id=meta.get("episode_id"), repeat=meta.get("repeat"),
            run_dir=os.path.relpath(leaf_dir, REPO_ROOT),
            n_ticks=s["n_ticks"],
            E_replay_onestep_joint_rad2=s["E_replay_onestep_joint_rad2"],
            E_replay_openloop_joint_rad2=s["E_replay_openloop_joint_rad2"],
            onestep_tcp_rms_m=s["onestep_tcp_rms_m"],
            onestep_box_rms_m=s["onestep_box_rms_m"],
            time_to_divergence_step=s["time_to_divergence_step"],
            time_to_divergence_s=s["time_to_divergence_s"],
        ))
        # Keep the per-step frames of the first run of each config for the
        # per-run F6 panel (memory: 5 configs x 2 small frames).
        if config_id not in per_step_cache:
            per_step_cache[config_id] = (s["one_step"], s["open_loop"], leaf_dir)
        if i % 10 == 0 or i == len(runs):
            print(f"  [{i}/{len(runs)}] {config_id} {cube_size} "
                  f"ttd={s['time_to_divergence_step']}")

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(_THIS_DIR, "f6_replay_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n[f6] wrote {csv_path}  ({len(summary)} runs)")

    if summary.empty:
        print("[f6] nothing replayed -- no figures written.")
        return summary

    agg = summary.groupby("config_id").agg(
        n=("run_dir", "count"),
        E_onestep_mean=("E_replay_onestep_joint_rad2", "mean"),
        E_onestep_med=("E_replay_onestep_joint_rad2", "median"),
        tcp_rms_mean_mm=("onestep_tcp_rms_m", lambda s: 1000 * s.mean()),
        ttd_med=("time_to_divergence_step", "median"),
        ttd_s_med=("time_to_divergence_s", "median"),
        ttd_nan=("time_to_divergence_step", lambda s: int(s.isna().sum())),
    )
    print("\n[f6] per-config aggregate (the decoupling check -- compare the "
          "spread here against F1's return gap):")
    print(agg.to_string(float_format=lambda v: f"{v:,.4g}"))

    outputs = []
    # Per-run F6 panel: use the first cached run of the mid ladder rung if
    # available, else whatever config came first.
    pick = "L2_pos_cube" if "L2_pos_cube" in per_step_cache else \
        next(iter(per_step_cache))
    os_df, ol_df, leaf = per_step_cache[pick]
    print(f"\n[f6] per-run F6 panel from {os.path.relpath(leaf, REPO_ROOT)}")
    outputs.append(pg.f6_replay_error(
        os_df, ol_df, os.path.join(out_dir, "f6_replay_error")))
    outputs.append(f6_aggregate(summary, os.path.join(out_dir, "f6b_replay_aggregate")))

    print(f"\n[f6] wrote {len(outputs)} figures (png+pdf each) to {out_dir}:")
    for png in outputs:
        for p in (png, png.replace(".png", ".pdf")):
            print(f"  {p}  ({os.path.getsize(p):,} bytes)")
    return summary


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real_root", default=DEFAULT_REAL_ROOT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None,
                    help="replay only the first N complete runs (smoke check)")
    args = ap.parse_args()
    run(args.real_root, args.out_dir, limit=args.limit)


if __name__ == "__main__":
    _cli()
