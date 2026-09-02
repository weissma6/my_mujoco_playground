"""Verify a completed curriculum ladder against W&B, per WP7's pass criteria.

Never trust "it finished" from a log, a chat message, or the disk-side
curriculum_summary.json alone -- go to the verification source and check
programmatically. See WP7 in the Planfile:
06 Archive/VT2-SimToReal-Robotics/Code/Plans/20260828_Curriculum learning -
warm-start DR ladder.md.

Pass criteria (all four required):
  1. All six runs present in W&B. A missing run is a hard failure, not
     missing data -- anything dying before wandb.init leaves no trace there.
  2. The checksum chain holds: each rung's curriculum/params_sha256_at_init
     equals its predecessor's curriculum/published_sha256.
  3. v3 trains every rung to its full fixed budget -- no early stop is
     possible by design, so it is now a FAILURE (not a required success
     signal) if a rung reports curriculum/stopped_early == True, or carries
     a curriculum/early_stop_summary whose stopped is true. Every rung's
     final logged training step must also be >= 30,000,000 (read from
     training/num_steps, falling back to wandb's own _step); the bound is
     >=, never ==, since the real quantized total overshoots to 30,474,240.
     Every matched run's name must start with Curr_v5_.
  4. sacct State is COMPLETED, not TIMEOUT (checked separately on the HPC --
     this script only has W&B, so it reports what it CAN verify and says so).

Usage:
    python batch_runs/curriculum/verify_ladder.py --group curriculum_20260831_182357
"""

import argparse
import json
import sys

# Order matters: check()'s checksum-chain walk is ADJACENCY-BASED -- it
# compares each rung's curriculum/params_sha256_at_init against the PREVIOUS
# entry's curriculum/published_sha256. L0_5_light must sit at index 1 because
# L1_pos now warm-starts from L0_5_light, not from L0_none. Reordering or
# dropping an entry here does not raise an error -- it silently pairs the
# wrong two rungs and reports a FALSE broken chain (or misses that a rung
# never reached W&B at all, which criterion 1 treats as a hard failure, not
# missing data). Do not "tidy" this list without re-deriving it from the
# spec's actual warm_start_from chain.
RUNG_ORDER = ["L0_none", "L0_5_light", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot", "L4_full"]
ENTITY = "weissma6-zhaw-school-of-engineering"
PROJECT = "UR3_pick_ppo"

# The true quantized total (31*983_040, see gen_curriculum.py's NUM_EVALS
# comment) overshoots this -- the floor must stay >=, never ==.
STEP_FLOOR = 30_000_000


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
    for rung in RUNG_ORDER:
        if rung not in by_rung:
            prev_pub = None
            continue
        run = by_rung[rung]["run"]
        s = by_rung[rung]["summary"]
        init_sha = s.get("curriculum/params_sha256_at_init")
        pub_sha = s.get("curriculum/published_sha256")
        chained = (prev_pub is None) or (init_sha == prev_pub)
        if not chained:
            problems.append(
                f"{rung}: params_sha256_at_init={str(init_sha)[:16]} does NOT match "
                f"predecessor's published_sha256={str(prev_pub)[:16]}"
            )

        # criterion 3, inverted from v1/v2: v3 has no windowed tracker, so any
        # sign of an early stop is a fault, whether it arrives as the plain
        # flag or as a (dict or JSON-string) early_stop_summary.
        es = s.get("curriculum/early_stop_summary")
        es = json.loads(es) if isinstance(es, str) else (es or {})
        stopped = bool(s.get("curriculum/stopped_early")) or bool(es.get("stopped"))
        if stopped:
            problems.append(
                f"{rung}: reports an early stop -- v3 trains every rung to "
                f"its full fixed budget, so this is a fault"
            )

        steps = s.get("training/num_steps")
        if steps is None:
            steps = s.get("_step")
        if steps is None:
            problems.append(
                f"{rung}: neither training/num_steps nor _step is present -- "
                f"the step floor cannot be verified"
            )
        elif isinstance(steps, bool) or not isinstance(steps, (int, float)):
            # W&B summaries are not schema-enforced -- a value like the
            # string "30000000" is a realistic shape, not a contrived one.
            # Report it rather than raise, and rather than silently coerce.
            problems.append(
                f"{rung}: step count {steps!r} is not numeric -- "
                f"the step floor cannot be verified"
            )
            steps = None
        elif steps < STEP_FLOOR:
            problems.append(
                f"{rung}: logged only {steps:,} steps, below the "
                f"{STEP_FLOOR:,} floor"
            )

        if not run.name.startswith("Curr_v5_"):
            problems.append(
                f"{rung}: run name {run.name!r} does not start with Curr_v5_"
            )

        rows.append({
            "rung": rung,
            "state": run.state,
            "chained_from_predecessor": chained,
            "stopped_early": stopped,
            "steps_completed": steps,
            "peak_success": s.get("eval/episode_success.max"),
            "final_success": s.get("eval/episode_success"),
            "published_sha256": pub_sha,
        })
        prev_pub = pub_sha

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

    hdr = f"{'rung':<20}{'state':<10}{'chained':>9}{'early':>7}{'steps':>12}{'peak_success':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['rung']:<20}{r['state']:<10}{str(r['chained_from_predecessor']):>9}"
              f"{str(r['stopped_early']):>7}{(r['steps_completed'] or 0):>12,}"
              f"{(r['peak_success'] or 0):>14.3f}")

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
