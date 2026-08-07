"""Driver script: render the gap-protocol figures that are ready RIGHT NOW
from the real evaluation/gap_metrics.csv built by `gap_metrics.py --build`.

plots_gap.py's own docstring says it "has no non-demo CLI entry point; call
the f1..f8 functions directly from a driver script once real gap_metrics.csv
/ replay / timing data exist" -- this is that driver, for the subset of
figures gap_metrics.csv alone can feed:

  READY NOW (this script):
    F1  f1_dose_response_cube_split  -- dose-response, 3cm vs 4cm overlay
    F2  f2_term_retention            -- per-term/per-stage retention bars
    F4  f4_paired_scatter            -- one per config, paired episode scatter
    F5  f5_selection_regret          -- sim-selection regret, 2-panel
    F9  f9_abort_rate_bar            -- abort rate by (policy, cube_size)

  NOT YET (need more than gap_metrics.csv -- see the printed note at the end):
    F3  per-step exact-parity trace  -- needs per-step (n_ep, H) arrays, not
        just the episode-summed scaled_return_H this CSV carries. Sim has the
        per-step data on disk already (sim_states.csv); real needs re-running
        ur3_reward_replay.replay_dataframe per run and keeping the per-step
        output instead of summing it. Only full-length (non-aborted) episodes
        can go in (F3 requires a shared H).
    F6  dynamics-replay error        -- needs evaluation/ur3_dynamics_replay.py
        run locally (plain MuJoCo, not yet executed).
    F7  timing honesty before/after  -- needs the achieved-Hz/loop_dt_true_s
        columns already logged in ur3_pick_states.csv, split by whichever
        run/git_sha marks the gripper-thread-decoupling fix (not yet
        identified).
    F8  LOO attribution (sim only)   -- needs LOO cluster-ablation sim configs
        (LOO_no_pos etc.) that were never trained/rolled out. New HPC
        experiments, not a data-prep gap.

Usage:
    python evaluation/run_gap_plots.py [--csv evaluation/gap_metrics.csv]
                                        [--out_dir evaluation/gap_plots]
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import gap_metrics as gm  # noqa: E402
from evaluation import plots_gap as pg  # noqa: E402

DEFAULT_CSV = os.path.join(_THIS_DIR, "gap_metrics.csv")
DEFAULT_OUT_DIR = os.path.join(_THIS_DIR, "gap_plots")

# All 5 D18-D25 velocity-ladder configs -- per
# robots/UR3e/gap_protocol_policy_map.json, ALL FIVE learned this time
# (D20's softened burst-force C4 fixed the earlier L4/L5 collapse), so unlike
# the stale D14_REAL_POLICIES module constant (4 configs, pre-fix), nothing
# needs to be excluded as "failed to train" for this campaign.
LEVELS = ["L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot", "L4_full"]
FAILED_CONFIGS = ()  # none -- see LEVELS comment
PRIMARY_CUBE = "3cm"  # in-distribution cube; non-cube-split figures pin here


def run(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = gm.load_long(csv_path)
    # Which trained checkpoint is behind each ladder rung -- a bare "L1_pos"
    # names the DR level, not the policy. Annotated on the figures where the
    # policy is the series (F2's legend, F9's x-groups).
    policy_map = pg.load_policy_map()
    if not policy_map:
        print(f"[run_gap_plots] WARNING: no policy map at "
              f"{pg.POLICY_MAP_PATH} -- figures will show bare config_ids "
              f"without checkpoint provenance")
    present = sorted(df["config_id"].unique())
    levels = [c for c in LEVELS if c in present]
    missing = [c for c in LEVELS if c not in present]
    if missing:
        print(f"[run_gap_plots] WARNING: expected configs not in {csv_path}: "
              f"{missing} -- continuing with {levels}")

    outputs = []

    outputs.append(pg.f1_dose_response_cube_split(
        df, levels, os.path.join(out_dir, "f1_dose_response"),
        failed_configs=FAILED_CONFIGS, cube_sizes=("3cm", "4cm")))

    outputs.append(pg.f2_term_retention(
        df, levels, os.path.join(out_dir, "f2_term_retention"),
        cube_size=PRIMARY_CUBE, policy_map=policy_map))

    for c in levels:
        outputs.append(pg.f4_paired_scatter(
            df, c, os.path.join(out_dir, f"f4_paired_scatter_{c}"),
            cube_size=PRIMARY_CUBE))

    outputs.append(pg.f5_selection_regret(
        df, levels, os.path.join(out_dir, "f5_selection_regret"),
        cube_size=PRIMARY_CUBE))

    outputs.append(pg.f9_abort_rate_bar(
        df, levels, os.path.join(out_dir, "f9_abort_rate_bar"),
        cube_sizes=("3cm", "4cm"), policy_map=policy_map))

    print(f"\n[run_gap_plots] wrote {len(outputs)} figures (png+pdf each) to "
          f"{out_dir}:")
    for png in outputs:
        pdf = png.replace(".png", ".pdf")
        for p in (png, pdf):
            print(f"  {p}  ({os.path.getsize(p):,} bytes)")

    print(
        "\n[run_gap_plots] NOT rendered here (need more than gap_metrics.csv "
        "-- see this file's module docstring for exactly what's missing):\n"
        "  F3 per-step exact-parity trace, F6 dynamics-replay error, "
        "F7 timing honesty before/after, F8 LOO attribution."
    )
    return outputs


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    run(args.csv, args.out_dir)


if __name__ == "__main__":
    _cli()
