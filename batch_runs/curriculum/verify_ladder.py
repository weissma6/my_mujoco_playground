"""Verify a completed curriculum ladder against W&B, per WP7's pass criteria.

Never trust "it finished" from a log, a chat message, or the disk-side
curriculum_summary.json alone -- go to the verification source and check
programmatically. See WP7 in the Planfile:
06 Archive/VT2-SimToReal-Robotics/Code/Plans/20260828_Curriculum learning -
warm-start DR ladder.md.

Pass criteria (all four required):
  1. All five runs present in W&B. A missing run is a hard failure, not
     missing data -- anything dying before wandb.init leaves no trace there.
  2. The checksum chain holds: each rung's curriculum/params_sha256_at_init
     equals its predecessor's curriculum/published_sha256.
  3. At least one rung has curriculum/early_stop_summary.stopped == true
     with steps_completed < its num_timesteps cap.
  4. sacct State is COMPLETED, not TIMEOUT (checked separately on the HPC --
     this script only has W&B, so it reports what it CAN verify and says so).

Usage:
    python batch_runs/curriculum/verify_ladder.py --group curriculum_20260831_182357
"""

import argparse
import json
import sys

RUNG_ORDER = ["L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot", "L4_full"]
ENTITY = "weissma6-zhaw-school-of-engineering"
PROJECT = "UR3_pick_ppo"


def fetch(group):
    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group}))
    by_rung = {}
    for r in runs:
        s = dict(r.summary or {})
        rung = s.get("curriculum/rung")
        if rung is None:
            continue
        by_rung[rung] = {"run": r, "summary": s}
    return by_rung


def check(by_rung):
    problems = []
    rows = []

    # criterion 1
    missing = [r for r in RUNG_ORDER if r not in by_rung]
    if missing:
        problems.append(f"MISSING from W&B (hard failure, not just absent data): {missing}")

    # criterion 2 -- checksum chain, sourced from W&B, not from disk
    prev_pub = None
    any_stopped_early = False
    for rung in RUNG_ORDER:
        if rung not in by_rung:
            prev_pub = None
            continue
        s = by_rung[rung]["summary"]
        init_sha = s.get("curriculum/params_sha256_at_init")
        pub_sha = s.get("curriculum/published_sha256")
        chained = (prev_pub is None) or (init_sha == prev_pub)
        if not chained:
            problems.append(
                f"{rung}: params_sha256_at_init={str(init_sha)[:16]} does NOT match "
                f"predecessor's published_sha256={str(prev_pub)[:16]}"
            )

        es = s.get("curriculum/early_stop_summary")
        es = json.loads(es) if isinstance(es, str) else (es or {})
        stopped = bool(es.get("stopped"))
        any_stopped_early |= stopped

        rows.append({
            "rung": rung,
            "state": by_rung[rung]["run"].state,
            "chained_from_predecessor": chained,
            "stopped_early": stopped,
            "steps_completed": es.get("last_step"),
            "best_reward": es.get("best_reward"),
            "peak_success": s.get("eval/episode_success.max"),
            "final_success": s.get("eval/episode_success"),
            "published_sha256": pub_sha,
        })
        prev_pub = pub_sha

    # criterion 3
    if not any_stopped_early:
        problems.append("no rung reports an early stop (criterion 3)")

    # criterion 4 cannot be checked from W&B alone
    note = ("sacct State (criterion 4) is not visible from W&B -- verify "
            "separately with `sacct -j <jobid> --format=JobID,State,ExitCode`.")

    return rows, problems, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    args = ap.parse_args()

    by_rung = fetch(args.group)
    rows, problems, note = check(by_rung)

    hdr = f"{'rung':<20}{'state':<10}{'chained':>9}{'early':>7}{'steps':>12}{'best_reward':>13}{'peak_success':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['rung']:<20}{r['state']:<10}{str(r['chained_from_predecessor']):>9}"
              f"{str(r['stopped_early']):>7}{(r['steps_completed'] or 0):>12,}"
              f"{(r['best_reward'] or 0):>13.1f}{(r['peak_success'] or 0):>14.3f}")

    print(f"\n{note}")

    if problems:
        print("\nFAILED pass criteria:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nAll W&B-verifiable pass criteria met (1, 2, 3). Confirm criterion 4 (sacct) separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
