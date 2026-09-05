"""Guarded, staged CPU-MJX timing probe for the UR3e+Hand-E cube-pick training.

Measures env-steps/s for env.reset/env.step (stages 1-3, at 1/8/32 envs) and
for one epoch of brax PPO (stage 4, 32 envs) on this MacBook's CPU, then
extrapolates that rate to the defence run and to one curriculum rung so the
campaign can budget cluster time before committing a SLURM allocation.

Both the vault CLAUDE.md and this repo's own CLAUDE.md say: never train or
smoke-test MJX locally on this machine -- CPU MJX has OOM-killed it before.
This script is the one sanctioned exception to that rule, authorised for the
campaign document dated 2026-09-04, and only for the measurement it runs
here. It refuses to run at all unless invoked with --acknowledge-cpu, and it
refuses outright on a SLURM node (the inverse of the cluster's usual
"assert GPU" guard, on purpose: this script must never run there).

Safety envelope, all enforced in this file:
  - the actual env/PPO work always happens in a child process, launched with
    JAX_PLATFORM_NAME=cpu set only in that child's environment (never in the
    parent's os.environ) and niced to 19;
  - a wall-clock timeout (<= 900 s) kills the child if it overruns;
  - an RssWatchdog thread polls the child's RSS and SIGKILLs it the moment
    it crosses the configured cap, independent of the timeout;
  - at most 64 envs, and the envs escalate stage by stage (1, 8, 32, 32)
    rather than jumping straight to a training-sized batch;
  - stage 4 (the PPO epoch) only runs when the most recent stage-3 record in
    --results-dir finished "ok" with peak RSS under STAGE4_PRIOR_PEAK_MAX_GB
    -- a stage-3 crash or a fat stage-3 peak blocks it automatically.

No rendering happens anywhere in this probe (env.reset/env.step and
ppo.train run headless), so MUJOCO_GL is neither read nor set. PPO's
network_factory always passes max_devices_per_host=1: this shell's .zshrc
forces 14 XLA host devices, and without that pin brax would pmap the tiny
stage-4 batch across all 14 of them.

Every run appends one JSON record to --results-dir (default
results/local_probe/, gitignored) and prints a human-readable report; those
records and reports are what gets pasted back into the plan.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STANDING_RULE = (
    "Never train or smoke-test locally: the vault CLAUDE.md and this repo's "
    "own CLAUDE.md both forbid running MJX on this Mac -- CPU MJX has "
    "OOM-killed it before. This script is the one sanctioned exception, "
    "authorised for the campaign document dated 2026-09-04, and it still "
    "refuses to do anything unless it is invoked with --acknowledge-cpu."
)

RSS_CAP_DEFAULT_GB = 8.0
RSS_CAP_MAX_GB = 12.0
NUM_ENVS_MAX = 64
TIMEOUT_MAX_S = 900
STAGE_NUM_ENVS = {1: 1, 2: 8, 3: 32, 4: 32}
STAGE4_PRIOR_PEAK_MAX_GB = 4.0

DEFENCE_RUN_ID = "Snappy2_as04_ar70_g01_s1"
DEFENCE_TOTAL_STEPS = 24_903_680
RUNG_TOTAL_STEPS = 30_474_240
TARGETS = {
    "defence run (Snappy2, 24 903 680 steps)": DEFENCE_TOTAL_STEPS,
    "curriculum rung (30 474 240 steps)": RUNG_TOTAL_STEPS,
}


def total_steps(
    num_timesteps,
    num_evals,
    batch_size,
    unroll_length,
    num_minibatches,
    num_resets_per_eval=1,
    action_repeat=1,
):
    """Reproduce jax_ppo_paramcalculation.md §1: the actual TOTAL_STEPS a
    brax ppo.train call runs, given the ceil() rounding up from a floor
    target num_timesteps."""
    env_step = batch_size * unroll_length * num_minibatches * action_repeat
    evals_after_init = max(num_evals - 1, 1)
    resets = max(num_resets_per_eval, 1)
    training_steps = math.ceil(num_timesteps / (evals_after_init * env_step * resets))
    return evals_after_init * resets * training_steps * env_step


def extrapolate(env_steps_per_s, total_steps):
    """Wall-clock seconds to cover total_steps at env_steps_per_s."""
    if not (env_steps_per_s > 0 and total_steps > 0):
        raise ValueError("env_steps_per_s and total_steps must both be positive")
    return total_steps / env_steps_per_s


def ppo_stage_kwargs():
    """The tiny, fixed brax ppo.train kwargs for stage 4: one epoch (three
    gradient updates), no eval loop, num_envs capped well under
    NUM_ENVS_MAX, purely to measure wall time for a real PPO iteration."""
    return dict(
        num_envs=32,
        batch_size=32,
        num_minibatches=4,
        unroll_length=10,
        num_updates_per_batch=1,
        num_evals=0,
        run_evals=False,
        num_resets_per_eval=1,
        num_timesteps=3 * 32 * 10 * 4,
        max_devices_per_host=1,
        normalize_observations=True,
        discounting=0.99,
        max_grad_norm=1.0,
        action_repeat=1,
    )


def load_defence_cfg(jsonl_path, run_id):
    """Parse one JSONL sweep file and return the record whose run_id
    matches, so the probe trains the exact defence-run config."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            if record.get("run_id") == run_id:
                return record
    raise KeyError(run_id)


def rss_bytes(pid):
    """Current RSS of pid in bytes, via `ps`; 0 if the pid is gone or ps
    returns something unparsable."""
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    )
    out = result.stdout.strip()
    try:
        return int(out) * 1024
    except ValueError:
        return 0


class RssWatchdog:
    """Polls a child pid's RSS on a daemon thread and SIGKILLs it the first
    time it exceeds cap_bytes; keeps polling afterwards (harmless) so
    peak_bytes still reflects the true peak once stop() is called."""

    def __init__(self, pid, cap_bytes, interval_s=0.5):
        self.pid = pid
        self.cap_bytes = cap_bytes
        self.interval_s = interval_s
        self.peak_bytes = 0
        self.tripped = False
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop_event.is_set():
            rss = rss_bytes(self.pid)
            self.peak_bytes = max(self.peak_bytes, rss)
            if rss > self.cap_bytes and not self.tripped:
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.tripped = True
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()


def stage4_allowed(results_dir):
    """Gate stage 4 on the LAST stage-3 record across every *.jsonl in
    results_dir (files read in sorted order, so a later file's stage-3
    record wins over an earlier one)."""
    last_stage3 = None
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("stage") == 3:
                last_stage3 = record

    if last_stage3 is None:
        return False, f"no stage 3 record in {results_dir}"

    outcome = last_stage3.get("outcome")
    if outcome != "ok":
        return False, f"last stage 3 outcome was {outcome}, not ok"

    peak = last_stage3.get("peak_rss_gb")
    if peak is None or peak >= STAGE4_PRIOR_PEAK_MAX_GB:
        return False, (
            f"last stage 3 peak RSS {peak} GB is not under "
            f"{STAGE4_PRIOR_PEAK_MAX_GB} GB"
        )

    return True, f"stage 3 ok at {peak} GB peak"


def format_report(record):
    stage = record.get("stage")
    num_envs = record.get("num_envs")
    steps = record.get("steps")
    outcome = record.get("outcome")
    env_steps_per_s = record.get("env_steps_per_s")
    peak_rss_gb = record.get("peak_rss_gb")
    rss_cap_gb = record.get("rss_cap_gb")
    compile_s = record.get("compile_s")
    run_s = record.get("run_s")

    lines = [
        f"Stage {stage} — num_envs={num_envs}, steps={steps or 'n/a'}, outcome={outcome}",
        f"env-steps/s: {env_steps_per_s or 'n/a'}",
        f"peak RSS: {peak_rss_gb} GB (cap {rss_cap_gb} GB)",
        f"JIT compile: {compile_s if compile_s is not None else 'n/a'} s",
        f"run: {run_s if run_s is not None else 'n/a'} s",
    ]

    for name, target in TARGETS.items():
        if outcome == "ok" and env_steps_per_s and env_steps_per_s > 0:
            seconds = extrapolate(env_steps_per_s, target)
            hours = seconds / 3600.0
            days = hours / 24.0
            lines.append(f"extrapolation → {name}: {hours:.1f} h ({days:.2f} d)")
        else:
            lines.append(f"extrapolation → {name}: n/a ({outcome})")

    lines.append(
        f"Extrapolation assumes the aggregate env-steps/s measured with "
        f"{num_envs} envs holds for the whole run. A real run batches 2048 "
        f"envs on 14 CPU cores; there the aggregate rate is unknown — it "
        f"rises with the env count until the cores saturate while the "
        f"per-env rate falls throughout — so read these as an order of "
        f"magnitude, not a schedule."
    )
    return "\n".join(lines)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rss-cap-gb", type=float, default=RSS_CAP_DEFAULT_GB)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "local_probe"
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=REPO_ROOT / "batch_runs" / "sweeps" / "UR3Pick_snappy2.jsonl",
    )
    parser.add_argument("--run-id", default=DEFENCE_RUN_ID)
    parser.add_argument("--acknowledge-cpu", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-json", default=None)
    return parser


def build_env(cfg):
    from learning.notebooks.run_experiment import build_env_overrides
    from mujoco_playground import registry

    return registry.load(str(cfg["env_name"]), config_overrides=build_env_overrides(cfg))


def run_env_stage(cfg, num_envs, steps):
    import jax

    env = build_env(cfg)
    key = jax.random.PRNGKey(0)
    k_reset, k_act = jax.random.split(key)

    if num_envs == 1:
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        reset_arg = k_reset
        actions = jax.random.uniform(
            k_act, (steps + 1, env.action_size), minval=-1.0, maxval=1.0
        )
    else:
        reset = jax.jit(jax.vmap(env.reset))
        step = jax.jit(jax.vmap(env.step))
        reset_arg = jax.random.split(k_reset, num_envs)
        actions = jax.random.uniform(
            k_act, (steps + 1, num_envs, env.action_size), minval=-1.0, maxval=1.0
        )
    jax.block_until_ready(actions)

    t0 = time.perf_counter()
    state = reset(reset_arg)
    state = step(state, actions[0])
    jax.block_until_ready(state)
    compile_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    for i in range(1, steps + 1):
        state = step(state, actions[i])
    jax.block_until_ready(state)
    run_s = time.perf_counter() - t1

    env_steps_per_s = steps * num_envs / run_s
    obs_size = env.observation_size
    if not isinstance(obs_size, int):
        obs_size = str(obs_size)

    return dict(
        compile_s=round(compile_s, 3),
        run_s=round(run_s, 3),
        env_steps_per_s=round(env_steps_per_s, 1),
        steps=steps,
        num_envs=num_envs,
        obs_size=obs_size,
        action_size=int(env.action_size),
    )


def run_ppo_stage(cfg):
    import functools

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    env = build_env(cfg)

    nf = dict(cfg.get("network_factory") or {})
    if isinstance(nf.get("policy_hidden_layer_sizes"), list):
        nf["policy_hidden_layer_sizes"] = tuple(nf["policy_hidden_layer_sizes"])
    if isinstance(nf.get("value_hidden_layer_sizes"), list):
        nf["value_hidden_layer_sizes"] = tuple(nf["value_hidden_layer_sizes"])
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf)

    seen = []

    def progress_fn(step, metrics):
        seen.append((int(step), {k: float(v) for k, v in metrics.items()}))

    kw = ppo_stage_kwargs()
    t0 = time.perf_counter()
    ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        progress_fn=progress_fn,
        episode_length=int(cfg["episode_length"]),
        learning_rate=float(cfg["learning_rate"]),
        entropy_cost=float(cfg["entropy_cost"]),
        reward_scaling=float(cfg["reward_scaling"]),
        seed=int(cfg["seed"]),
        **kw,
    )
    wall_s = time.perf_counter() - t0

    last_metrics = seen[-1][1] if seen else {}
    return dict(
        wall_s=round(wall_s, 3),
        compile_s=None,
        run_s=round(wall_s, 3),
        env_steps_per_s=round(kw["num_timesteps"] / wall_s, 1),
        training_sps=last_metrics.get("training/sps"),
        training_walltime_s=last_metrics.get("training/walltime"),
        num_envs=kw["num_envs"],
        steps=kw["num_timesteps"],
        ppo_iterations=3,
        note=(
            "single epoch: compile time not separable from the 3 PPO "
            "iterations; env_steps_per_s includes compile"
        ),
    )


def run_worker(args):
    sys.path.insert(0, str(REPO_ROOT))
    cfg = load_defence_cfg(args.jsonl, args.run_id)

    import jax

    if args.stage in (1, 2, 3):
        record = run_env_stage(cfg, args.num_envs, args.steps)
    else:
        record = run_ppo_stage(cfg)

    record["jax_platform"] = jax.devices()[0].platform
    record["jax_device_count"] = len(jax.devices())
    record["jax_version"] = jax.__version__

    Path(args.result_json).write_text(json.dumps(record))
    return 0


def run_supervisor(args):
    stage = args.stage
    num_envs = args.num_envs

    if stage == 4:
        ok, why = stage4_allowed(args.results_dir)
        if not ok:
            print(f"refusing stage 4: {why}", file=sys.stderr)
            return 2

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    result_json = results_dir / f"{stamp}_stage{stage}_result.json"
    log_path = results_dir / f"{stamp}_stage{stage}.log"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        str(stage),
        "--num-envs",
        str(num_envs),
        "--steps",
        str(args.steps),
        "--jsonl",
        str(args.jsonl),
        "--run-id",
        args.run_id,
        "--rss-cap-gb",
        str(args.rss_cap_gb),
        "--timeout-s",
        str(args.timeout_s),
        "--acknowledge-cpu",
        "--worker",
        "--result-json",
        str(result_json),
    ]
    child_env = {**os.environ, "JAX_PLATFORM_NAME": "cpu"}

    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
            preexec_fn=lambda: os.nice(19),
        )

    wd = RssWatchdog(proc.pid, cap_bytes=int(args.rss_cap_gb * 1024**3), interval_s=0.25)
    wd.start()
    print(
        f"stage {stage}: num_envs={num_envs} rss_cap={args.rss_cap_gb}GB "
        f"timeout={args.timeout_s}s pid={proc.pid}"
    )

    timed_out = False
    try:
        proc.wait(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        timed_out = True
    wd.stop()

    if timed_out:
        outcome = "timeout"
    elif wd.tripped:
        outcome = "watchdog"
    elif proc.returncode == 0 and result_json.exists():
        outcome = "ok"
    else:
        outcome = "error"

    record = dict(
        stage=stage,
        num_envs=num_envs,
        steps=args.steps,
        outcome=outcome,
        peak_rss_gb=round(wd.peak_bytes / 1024**3, 2),
        rss_cap_gb=args.rss_cap_gb,
        timeout_s=args.timeout_s,
        timestamp=stamp,
        log=str(log_path),
        python=sys.version.split()[0],
        returncode=proc.returncode,
    )

    if outcome == "ok":
        record.update(json.loads(result_json.read_text()))
    else:
        last_line = ""
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    last_line = line
        record["error"] = last_line if last_line else f"returncode {proc.returncode}"

    with open(results_dir / f"{stamp}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(format_report(record))
    return 0 if outcome == "ok" else 1


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    num_envs = args.num_envs if args.num_envs is not None else STAGE_NUM_ENVS[args.stage]

    if "SLURM_JOB_ID" in os.environ:
        print(
            "refusing: SLURM_JOB_ID is set -- this probe never runs on the "
            "cluster",
            file=sys.stderr,
        )
        return 2

    if not args.acknowledge_cpu:
        print(STANDING_RULE, file=sys.stderr)
        return 2

    if not (0 < args.rss_cap_gb <= RSS_CAP_MAX_GB):
        print(
            f"refusing: --rss-cap-gb {args.rss_cap_gb} is outside the "
            f"allowed cap (0, {RSS_CAP_MAX_GB}]",
            file=sys.stderr,
        )
        return 2

    if not (1 <= num_envs <= NUM_ENVS_MAX):
        print(
            f"refusing: --num-envs {num_envs} is outside the allowed "
            f"number of envs (1, {NUM_ENVS_MAX}]",
            file=sys.stderr,
        )
        return 2

    if not (0 < args.timeout_s <= TIMEOUT_MAX_S):
        print(
            f"refusing: --timeout-s {args.timeout_s} is outside the "
            f"allowed timeout (0, {TIMEOUT_MAX_S}]",
            file=sys.stderr,
        )
        return 2

    import jax

    if jax.devices()[0].platform != "cpu":
        print(
            "refusing: jax platform is not cpu -- this probe must never "
            "run off-CPU",
            file=sys.stderr,
        )
        return 2

    args.num_envs = num_envs
    if args.worker:
        return run_worker(args)
    return run_supervisor(args)


if __name__ == "__main__":
    sys.exit(main())
