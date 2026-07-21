"""D10: median-seed selection over the gap protocol's SIM returns (Commit 8).

Plain Python + file I/O -- no MJX, no jax, safe to run locally. Glue script
that unblocks real-robot data collection: robots/UR3e/run_gap_protocol.py
refuses to run without a resolved `policy_run_id` for its --config_id, and
gap_protocol_policy_map.json (the file it reads that from) is what this
script writes.

Selection rule (locked decision, D10)
--------------------------------------------------------------------------
For a DR-ladder config trained at 3 seeds, pick the seed whose SIM return on
the gap protocol's 10 episodes is the MEDIAN of the 3 seeds -- explicitly
NOT the seed with the best training/W&B reward, which is a different
distribution over different episodes (the protocol's held-out test-split
poses vs. the training env's own random resets). The per-seed score is the
MEAN return over the 10 protocol episodes; the "median seed" is whichever
seed's score is the middle value when the (odd count, 3) scores are sorted --
no interpolation, since there is no such thing as "the seed between two
seeds." A warning fires (and the LOWER of the two central scores is picked,
documented, not silently) if ever called with an even seed count.

Input format -- expects evaluation/run_gap_protocol_sim.py's
--protocol_only output
--------------------------------------------------------------------------
Run the sim mirror once PER SEED first, each into its own seed-tagged
--out_root (the config_id stays the DR-ladder config's own id; --protocol_only
is required here because at seed-selection time no real run folder exists
yet to mirror -- see run_gap_protocol_sim.py's module docstring):

    for seed, run_id in {0: "<wandb_run_0>", 1: "<wandb_run_1>", 2: "<wandb_run_2>"}.items():
        python evaluation/run_gap_protocol_sim.py --config_id L1_pos \\
            --policy_run_id <wandb_run_id_for_that_seed> --protocol_only \\
            --out_root robots/UR3e/sim_results/seed_selection/s{seed}

That writes .../s{seed}/L1_pos/{protocol_id}/ep{ID}_rep0/sim_meta.json (one
per protocol episode), each carrying "episode_return" + "policy_run_id".
This script reads exactly those files -- see `load_seed_returns`.

Local usage:
    python evaluation/select_median_seed.py --selftest
    python evaluation/select_median_seed.py --config_id L1_pos \\
        --seed_run_id_map '{"0":"<run0>","1":"<run1>","2":"<run2>"}' \\
        --sim_root robots/UR3e/sim_results/seed_selection
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_POLICY_MAP = os.path.join(REPO_ROOT, "robots", "UR3e", "gap_protocol_policy_map.json")
DEFAULT_PROTOCOL = os.path.join(REPO_ROOT, "evaluation", "protocols", "gap_protocol_v1.json")
DEFAULT_SIM_ROOT = os.path.join(REPO_ROOT, "robots", "UR3e", "sim_results", "seed_selection")


# ===========================================================================
# Load per-seed per-episode sim returns.
# ===========================================================================


def load_seed_returns(
    sim_root: str, config_id: str, protocol_id: str, seed_run_id_map: dict
) -> pd.DataFrame:
    """Read run_gap_protocol_sim.py --protocol_only output for every seed.

    seed_run_id_map: {"<seed>": "<wandb_run_id>"} (str keys -- JSON-friendly;
    cast to int internally). Returns a long DataFrame: seed, run_id,
    episode_id, episode_return. Raises loudly (FileNotFoundError /
    ValueError) rather than silently skipping a missing or mismatched seed --
    a seed silently dropped from the median computation would corrupt D10's
    selection without any visible sign.
    """
    rows = []
    for seed_str, run_id in seed_run_id_map.items():
        seed = int(seed_str)
        pattern = os.path.join(
            sim_root, f"s{seed}", config_id, protocol_id, "ep*_rep0", "sim_meta.json"
        )
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"no sim_meta.json under {pattern} -- run "
                f"evaluation/run_gap_protocol_sim.py --config_id {config_id} "
                f"--policy_run_id {run_id} --protocol_only --out_root "
                f"{os.path.join(sim_root, f's{seed}')} first (on the HPC; "
                f"see this module's docstring)."
            )
        for p in paths:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("policy_run_id") != run_id:
                raise ValueError(
                    f"{p}: policy_run_id={m.get('policy_run_id')!r} != the "
                    f"expected {run_id!r} for seed {seed} -- seed_run_id_map "
                    f"and the sim_results tree have gone out of sync."
                )
            rows.append({
                "seed": seed, "run_id": run_id,
                "episode_id": int(m["episode_id"]),
                "episode_return": float(m["episode_return"]),
            })
    return pd.DataFrame(rows, columns=["seed", "run_id", "episode_id", "episode_return"])


# ===========================================================================
# Selection.
# ===========================================================================


def select_median_seed(df: pd.DataFrame) -> dict:
    """Pick the seed whose MEAN return over the protocol episodes is the
    median across seeds. `df`: columns seed, run_id, episode_id,
    episode_return (>=1 row per seed per episode, as `load_seed_returns`
    produces). Returns the selection + the min/median/max band across seeds
    (the paper's "training-variance band").
    """
    per_seed = (
        df.groupby(["seed", "run_id"])["episode_return"].mean()
        .reset_index().rename(columns={"episode_return": "score"})
    )
    n = len(per_seed)
    if n == 0:
        raise ValueError("select_median_seed: no seeds in df")
    per_seed_sorted = per_seed.sort_values("score").reset_index(drop=True)
    mid = (n - 1) // 2
    if n % 2 == 0:
        print(
            f"[select_median_seed] WARNING: {n} seeds (even) -- no unique "
            f"middle element; picking the LOWER of the two central scores "
            f"(sorted index {mid}) rather than interpolating between seeds."
        )
    median_row = per_seed_sorted.iloc[mid]
    return {
        "selected_seed": int(median_row["seed"]),
        "selected_run_id": str(median_row["run_id"]),
        "n_seeds": n,
        "n_episodes": int(df.groupby("seed")["episode_id"].nunique().max()),
        "per_seed_scores": {
            int(r.seed): float(r.score) for r in per_seed.itertuples()
        },
        "per_seed_run_ids": {
            int(r.seed): str(r.run_id) for r in per_seed.itertuples()
        },
        "band_min": float(per_seed_sorted["score"].iloc[0]),
        "band_median": float(median_row["score"]),
        "band_max": float(per_seed_sorted["score"].iloc[-1]),
    }


# ===========================================================================
# gap_protocol_policy_map.json -- EXACT schema robots/UR3e/run_gap_protocol.py
# reads (resolve_policy_run_id: a flat {config_id: wandb_run_id} dict).
# ===========================================================================


def update_policy_map(policy_map_path: str, config_id: str, run_id: str) -> dict:
    """Merge {config_id: run_id} into the policy map, creating it if it does
    not exist yet, NEVER clobbering other configs' entries.
    """
    policy_map = {}
    if os.path.exists(policy_map_path):
        with open(policy_map_path, encoding="utf-8") as f:
            policy_map = json.load(f)
    policy_map[config_id] = run_id
    out_dir = os.path.dirname(policy_map_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(policy_map_path, "w", encoding="utf-8") as f:
        json.dump(policy_map, f, indent=2, sort_keys=True)
    return policy_map


# ===========================================================================
# Seed-spread ("training-variance band") report.
# ===========================================================================


def format_band_table(results: dict) -> str:
    """results: {config_id: select_median_seed() output}. A small text table
    for stdout or a report file -- the paper's per-config seed-spread band."""
    header = (
        f"{'config_id':<20} {'seed_min':>8} {'min':>10} "
        f"{'seed_med*':>9} {'median (selected)':>18} "
        f"{'seed_max':>8} {'max':>10}"
    )
    lines = [header, "-" * len(header)]
    for cid, r in results.items():
        scores = r["per_seed_scores"]
        seed_min = min(scores, key=scores.get)
        seed_max = max(scores, key=scores.get)
        lines.append(
            f"{cid:<20} {seed_min:>8} {r['band_min']:>10.2f} "
            f"{r['selected_seed']:>9} {r['band_median']:>18.2f} "
            f"{seed_max:>8} {r['band_max']:>10.2f}"
        )
    return "\n".join(lines)


# ===========================================================================
# Smoke test -- fabricated sim_meta.json files on disk, no HPC/MJX needed.
# ===========================================================================


def _write_fake_sim_meta(sim_root, config_id, protocol_id, seed, run_id,
                          mean_return, n_episodes=10):
    for ep in range(n_episodes):
        out_dir = os.path.join(sim_root, f"s{seed}", config_id, protocol_id, f"ep{ep}_rep0")
        os.makedirs(out_dir, exist_ok=True)
        # Symmetric per-episode spread around mean_return so the per-seed
        # MEAN stays exactly mean_return (sum of (ep - (n-1)/2) over a
        # centered integer range is exactly 0).
        ep_return = mean_return + (ep - (n_episodes - 1) / 2.0)
        meta = {
            "episode_id": ep, "repeat": 0, "config_id": config_id,
            "protocol_id": protocol_id, "policy_run_id": run_id,
            "episode_return": ep_return,
        }
        with open(os.path.join(out_dir, "sim_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)


def _selftest():
    import tempfile

    tmp = tempfile.mkdtemp(prefix="select_median_seed_selftest_")
    sim_root = os.path.join(tmp, "sim_results")
    protocol_id, config_id = "test_proto", "L1_pos"
    seed_run_id_map = {"0": "run_seed0", "1": "run_seed1", "2": "run_seed2"}
    # Known per-seed mean scores. Sorted: seed2=60 (min), seed0=90 (MEDIAN),
    # seed1=120 (max) -> the median seed is seed0, NOT the highest or lowest.
    seed_scores = {0: 90.0, 1: 120.0, 2: 60.0}

    print("Fabricating sim_meta.json for 3 seeds x 10 episodes ...")
    for seed_str, run_id in seed_run_id_map.items():
        seed = int(seed_str)
        _write_fake_sim_meta(sim_root, config_id, protocol_id, seed, run_id,
                              seed_scores[seed])

    print("load_seed_returns ...", end=" ")
    df = load_seed_returns(sim_root, config_id, protocol_id, seed_run_id_map)
    assert len(df) == 30, len(df)
    assert set(df["seed"].unique()) == {0, 1, 2}
    print("OK")

    print("select_median_seed picks the MIDDLE seed (not best/worst) ...", end=" ")
    result = select_median_seed(df)
    assert result["selected_seed"] == 0, result
    assert result["selected_run_id"] == "run_seed0", result
    assert abs(result["band_min"] - 60.0) < 1e-6, result
    assert abs(result["band_median"] - 90.0) < 1e-6, result
    assert abs(result["band_max"] - 120.0) < 1e-6, result
    print(f"OK  (selected seed={result['selected_seed']} "
          f"run_id={result['selected_run_id']}, band="
          f"[{result['band_min']:.1f}, {result['band_median']:.1f}, "
          f"{result['band_max']:.1f}])")

    print("update_policy_map merges without clobbering other configs ...", end=" ")
    policy_map_path = os.path.join(tmp, "gap_protocol_policy_map.json")
    with open(policy_map_path, "w", encoding="utf-8") as f:
        json.dump({"OTHER_CONFIG": "some_other_run"}, f)
    pm = update_policy_map(policy_map_path, config_id, result["selected_run_id"])
    assert pm["OTHER_CONFIG"] == "some_other_run", pm
    assert pm[config_id] == "run_seed0", pm
    with open(policy_map_path, encoding="utf-8") as f:
        pm_reloaded = json.load(f)
    assert pm_reloaded == pm, (pm_reloaded, pm)
    print(f"OK  ({policy_map_path} = {pm})")

    print("format_band_table renders ...", end=" ")
    table = format_band_table({config_id: result})
    assert config_id in table and "90.00" in table
    print("OK")
    print("\n" + table)

    print("\nSELFTEST OK (fabricated data, no HPC/MJX).")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config_id", default=None)
    ap.add_argument("--seed_run_id_map", default=None,
                     help='JSON dict, e.g. \'{"0":"run0","1":"run1","2":"run2"}\'')
    ap.add_argument("--sim_root", default=DEFAULT_SIM_ROOT)
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--policy_map", default=DEFAULT_POLICY_MAP)
    ap.add_argument("--dry_run", action="store_true",
                     help="print the selection but do not write policy_map")
    args = ap.parse_args()

    if args.selftest or not args.config_id:
        _selftest()
        return

    from evaluation.protocols import load_protocol
    protocol = load_protocol(args.protocol, dry_run=True)

    seed_run_id_map = json.loads(args.seed_run_id_map)
    df = load_seed_returns(args.sim_root, args.config_id, protocol.protocol_id,
                            seed_run_id_map)
    result = select_median_seed(df)
    print(format_band_table({args.config_id: result}))

    if args.dry_run:
        print(f"\n--dry_run: NOT writing {args.policy_map}")
        return
    pm = update_policy_map(args.policy_map, args.config_id, result["selected_run_id"])
    print(f"\nWrote {args.policy_map}: {args.config_id} -> {result['selected_run_id']}")
    print(f"Full policy map now: {pm}")


if __name__ == "__main__":
    _cli()
