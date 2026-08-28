"""Drive the L0 -> L4 curriculum ladder inside ONE SLURM job.

Each rung is a separate `run_experiment` call, and therefore a separate
`wandb.init`. That is deliberate, not incidental: run_experiment logs with
`wandb.log(log_dict, step=num_steps)`, and brax always restarts `env_steps` at 0
for a warm-started run, so five rungs inside one W&B run would log a
non-monotonic step axis and W&B would silently drop the later points. The rungs
are joined by a shared `wandb_group` instead.

Two guards exist because the job runs under `--time=04:00:00`:

  1. The params handed to the next rung are written to disk BEFORE that rung
     starts, so a timeout costs one rung rather than the whole ladder.
  2. The remaining wall clock is checked before each rung after the first. If
     what is left is below the longest rung observed so far, the driver stops
     cleanly, writes a resume marker and exits 0 -- rather than being SIGKILLed
     mid-rung, which would lose that rung's params entirely.

The first rung always runs: a job that trains nothing is not a useful outcome,
and there is no measured per-rung estimate before one has finished.

Run:
    python batch_runs/scripts/run_curriculum_ur3.py \
        --spec batch_runs/curriculum/UR3Pick_curriculum.json \
        --out-root results --wall-budget-s 13800
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _default_runner(cfg, out_dir):
    """Imported lazily so the module itself stays importable without brax."""
    from learning.notebooks.run_experiment import run_experiment
    return run_experiment(cfg=cfg, out_dir=out_dir)


def _write_handoff(path, params):
    """Persist the params the NEXT rung warm-starts from.

    Written next to run_experiment's own params.msgpack under a distinct name:
    run_experiment writes the FINAL params there and the peak to
    best_params.msgpack, and the handoff is the peak. Overwriting params.msgpack
    with a different tree would quietly change what the deploy loader picks up.
    """
    from flax import serialization
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    return os.path.getsize(path)


def resolve_group(spec, override=None):
    if override:
        return override
    if spec.get("wandb_group"):
        return spec["wandb_group"]
    prefix = spec.get("wandb_group_prefix", "curriculum")
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_rung_cfg(spec, rung, group, warm_params=None):
    cfg = dict(spec["defaults"])
    cfg.update(rung["overrides"])
    cfg["run_id"] = rung["run_id"]
    cfg["seed"] = spec["seed"]
    cfg["wandb_project"] = spec["wandb_project"]
    cfg["wandb_group"] = group
    cfg["wandb_tags"] = ["curriculum", rung["config_id"]]
    cfg["curriculum_rung"] = rung["config_id"]
    cfg["warm_start_from"] = rung["warm_start_from"]
    if warm_params is not None:
        cfg["warm_start_params"] = warm_params
    return cfg


def run_curriculum(spec, out_root, *, wall_budget_s, runner=None,
                   clock=time.monotonic, group=None, writer=_write_handoff):
    runner = runner or _default_runner
    group = resolve_group(spec, group)
    group_dir = os.path.join(out_root, group)
    os.makedirs(group_dir, exist_ok=True)

    print(f"[curriculum] group={group}  rungs={len(spec['rungs'])}  "
          f"wall_budget={wall_budget_s}s", flush=True)

    t0 = clock()
    durations, completed = [], []
    warm_params = None
    stopped_reason = None

    for i, rung in enumerate(spec["rungs"]):
        if i > 0:
            elapsed = clock() - t0
            remaining = wall_budget_s - elapsed
            estimate = max(durations)
            if remaining < estimate:
                stopped_reason = (
                    f"wall budget: {remaining:.0f}s left, longest rung so far "
                    f"took {estimate:.0f}s -- stopping before {rung['config_id']} "
                    f"rather than being killed mid-rung"
                )
                print(f"[curriculum] STOP {stopped_reason}", flush=True)
                break

        rung_out = os.path.join(group_dir, rung["config_id"])
        os.makedirs(rung_out, exist_ok=True)
        cfg = build_rung_cfg(spec, rung, group, warm_params)

        print(f"[curriculum] --- rung {i + 1}/{len(spec['rungs'])}: "
              f"{rung['config_id']} (warm_start_from="
              f"{rung['warm_start_from']}) ---", flush=True)

        started = clock()
        result = runner(cfg, rung_out)
        durations.append(clock() - started)

        # The peak, not the final params: a post-peak collapse must not be what
        # the next rung inherits.
        handoff = result.get("best_params")
        if handoff is None:
            handoff = result.get("params")
        if handoff is None:
            raise RuntimeError(
                f"{rung['config_id']} returned neither best_params nor params"
            )

        # On disk BEFORE the next rung starts.
        size = writer(os.path.join(rung_out, "trained_policy",
                                   "handoff_params.msgpack"), handoff)

        completed.append({
            "config_id": rung["config_id"],
            "run_id": rung["run_id"],
            "wandb_run_id": result.get("wandb_run_id"),
            "steps_completed": result.get("steps_completed"),
            "stopped_early": result.get("stopped_early"),
            "best_reward": result.get("best_reward"),
            "best_step": result.get("best_step"),
            "params_sha256_at_init": result.get("params_sha256_at_init"),
            "published_sha256": result.get("published_sha256"),
            "handoff_bytes": size,
            "seconds": round(durations[-1], 1),
        })
        print(f"[curriculum] {rung['config_id']} done: "
              f"steps={result.get('steps_completed')} "
              f"early={result.get('stopped_early')} "
              f"handoff={size}B", flush=True)

        warm_params = handoff

    summary = {
        "group": group,
        "completed": completed,
        "n_completed": len(completed),
        "n_total": len(spec["rungs"]),
        "stopped_reason": stopped_reason,
    }
    with open(os.path.join(group_dir, "curriculum_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if stopped_reason is not None:
        # A resume marker, not a failure: exit 0 so SLURM records COMPLETED and
        # the finished rungs stay on disk.
        with open(os.path.join(group_dir, "_resume.json"),
                  "w", encoding="utf-8") as f:
            json.dump({
                "next_index": len(completed),
                "next_config_id": spec["rungs"][len(completed)]["config_id"],
                "completed": [c["config_id"] for c in completed],
                "reason": stopped_reason,
            }, f, indent=2)

    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(
        REPO, "batch_runs", "curriculum", "UR3Pick_curriculum.json"))
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--wall-budget-s", type=float, default=13800.0,
                    help="usable wall clock; default 3h50m of a 4h --time")
    ap.add_argument("--group", default=None)
    args = ap.parse_args(argv)

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    summary = run_curriculum(spec, args.out_root,
                             wall_budget_s=args.wall_budget_s,
                             group=args.group)
    print(f"[curriculum] {summary['n_completed']}/{summary['n_total']} rungs "
          f"completed in group {summary['group']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
