"""
Time/step analysis of the most recent UR3 W&B runs.

Pulls the time- and step-related fields W&B records for the last few finished
runs, lets you paste the runtimes the SLURM job manager reported (W&B never sees
queue/container/teardown time), and emits:
  - a table (run_ids on the columns) printed to stdout,
  - one twin-axis graph (runtime bars + env-timesteps line)  -> timeanalysis.png,
  - a markdown summary with the table + computed takeaways    -> timeanalysis.md

All final values come from run.summary / run.config (no history scraping).
W&B auth is taken from ~/.netrc (same as the deploy download helper).

Run:  python evaluation/timeanalysis.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless: never call plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ── Config ──────────────────────────────────────────────────────────────────
ENTITY = "weissma6-zhaw-school-of-engineering"
PROJECT = "UR3_pick_ppo"
N_RUNS = 4
FINISHED_ONLY = True

# Runtime as recorded by the SLURM job manager, in SECONDS.
# Newest-first, index-aligned to the run list this script prints at startup
# (index 0 = most recent run). Use None where you don't have the value.
# FOOTGUN: the "last 4 finished" set shifts as new runs finish — re-check the
# printed index->run_id mapping before trusting these. (A {run_id: seconds}
# dict would be more robust if you ever want it.)
SLURMTIME = [7, 17, 8, 15]  # minutes
SLURMTIME = SLURMTIME*60    # Seconds

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_OUT = os.path.join(HERE, "timeanalysis.png")
MD_OUT = os.path.join(HERE, "timeanalysis.md")

# Metric rows (table index). Key -> (label, summary/config source).
# Resolved per run below; kept here so table order is explicit.
RUNTIME_BAR_KEYS = ["slurm_time", "wandb_runtime", "training_walltime", "eval_walltime"]


def _g(d, key, default=np.nan):
  """summary/config .get that tolerates wandb's SummarySubDict + missing keys."""
  try:
    v = d.get(key, default)
  except Exception:
    return default
  return default if v is None else v


def fetch_runs():
  api = wandb.Api()
  runs = list(api.runs(f"{ENTITY}/{PROJECT}", order="-created_at"))
  if FINISHED_ONLY:
    runs = [r for r in runs if r.state == "finished"]
  return runs[:N_RUNS]


def build_record(run, slurm_time):
  s, c = run.summary, run.config
  wandb_runtime = float(_g(s, "_runtime"))
  training_walltime = float(_g(s, "training/walltime"))
  eval_walltime = float(_g(s, "eval/walltime", _g(s, "eval/epoch_eval_time")))
  setup_overhead = wandb_runtime - training_walltime - eval_walltime
  slurm_time = np.nan if slurm_time is None else float(slurm_time)
  slurm_overhead = slurm_time - wandb_runtime  # NaN if slurm_time is NaN
  return {
      "run_id": run.name,
      "state": run.state,
      "created_at": str(run.created_at),
      "num_envs": _g(c, "num_envs"),
      "unroll_length": _g(c, "unroll_length"),
      "num_timesteps_requested": _g(c, "num_timesteps"),
      "num_steps_actual": _g(s, "training/num_steps"),
      "wandb_step": _g(s, "_step"),
      "slurm_time": slurm_time,
      "wandb_runtime": wandb_runtime,
      "training_walltime": training_walltime,
      "eval_walltime": eval_walltime,
      "setup_overhead": setup_overhead,
      "slurm_overhead": slurm_overhead,
      "training_sps": _g(s, "training/sps"),
      "eval_sps": _g(s, "eval/sps"),
  }


def make_graph(df, run_ids):
  """Twin-axis: runtime bars (left, s) + env-timesteps line (right)."""
  labels = [rid.rsplit("_", 2)[0] for rid in run_ids]  # drop date/id suffix
  x = np.arange(len(run_ids))
  width = 0.2

  fig, ax = plt.subplots(figsize=(max(8, 2.6 * len(run_ids)), 6))
  for i, key in enumerate(RUNTIME_BAR_KEYS):
    vals = df.loc[key, run_ids].to_numpy(dtype=float)
    ax.bar(x + (i - 1.5) * width, vals, width, label=key)

  ax.set_ylabel("runtime [s]")
  ax.set_xlabel("run")
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
  ax.set_title(f"{PROJECT}: runtime vs env-steps (last {len(run_ids)} finished runs)")

  ax2 = ax.twinx()
  steps = df.loc["num_steps_actual", run_ids].to_numpy(dtype=float)
  ax2.plot(x, steps, "o-", color="black", label="num_steps_actual")
  ax2.set_ylabel("env timesteps")

  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

  fig.tight_layout()
  fig.savefig(PNG_OUT, dpi=130)
  plt.close(fig)


def fmt_secs(v):
  return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


def _cell(v):
  """Markdown cell: NaN/None -> em dash, floats trimmed, ints kept as ints."""
  if v is None or (isinstance(v, float) and np.isnan(v)):
    return "—"
  if isinstance(v, float):
    return str(int(v)) if v.is_integer() else f"{v:.2f}"
  return str(v)


def df_to_markdown(df):
  """GitHub-flavoured table (no `tabulate` dependency). Index = first column."""
  header = ["metric"] + list(df.columns)
  rows = ["| " + " | ".join(header) + " |",
          "| " + " | ".join("---" for _ in header) + " |"]
  for metric, row in df.iterrows():
    rows.append("| " + " | ".join([metric] + [_cell(v) for v in row]) + " |")
  return "\n".join(rows)


def write_markdown(df, records):
  lines = [
      f"# Time analysis — {PROJECT} (last {len(records)} finished runs)",
      "",
      "All times in seconds. `slurm_time` is hand-entered (SLURMTIME); every other",
      "value is from W&B (`run.summary` / `run.config`). `setup_overhead` ="
      " W&B runtime − training − eval; `slurm_overhead` = SLURM time − W&B runtime.",
      "",
      df_to_markdown(df),
      "",
      "## Takeaways",
  ]
  for rec in records:
    rid = rec["run_id"]
    wr, tw, ew, so = (rec["wandb_runtime"], rec["training_walltime"],
                      rec["eval_walltime"], rec["setup_overhead"])
    bullet = [f"- **{rid}** ({int(rec['num_steps_actual'])} env steps,"
              f" {rec['num_envs']} envs):"]
    bullet.append(f"W&B runtime {fmt_secs(wr)} s")
    if wr and not np.isnan(wr):
      bullet.append(f"(training {fmt_secs(tw)} s / eval {fmt_secs(ew)} s /"
                    f" setup {fmt_secs(so)} s)")
    if not np.isnan(rec["slurm_time"]):
      sov, pct = rec["slurm_overhead"], 100.0 * rec["slurm_overhead"] / wr if wr else np.nan
      bullet.append(f"; SLURM {fmt_secs(rec['slurm_time'])} s → overhead"
                    f" {fmt_secs(sov)} s ({pct:.0f}% beyond W&B)")
    else:
      bullet.append("; SLURM time not provided")
    bullet.append(f"; training {rec['training_sps']:.0f} sps")
    lines.append(" ".join(bullet))
  with open(MD_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")


def main():
  runs = fetch_runs()
  if not runs:
    raise SystemExit(f"No {'finished ' if FINISHED_ONLY else ''}runs in {ENTITY}/{PROJECT}")

  print(f"Last {len(runs)} {'finished ' if FINISHED_ONLY else ''}runs in "
        f"{ENTITY}/{PROJECT} (newest first):")
  for i, r in enumerate(runs):
    slot = SLURMTIME[i] if i < len(SLURMTIME) else None
    print(f"  [{i}] {r.name:46s} state={r.state:9s} SLURMTIME[{i}]={slot}")
  if len(SLURMTIME) != len(runs):
    print(f"  ! SLURMTIME has {len(SLURMTIME)} entries but {len(runs)} runs — "
          "missing slots treated as None.")

  records = [build_record(r, SLURMTIME[i] if i < len(SLURMTIME) else None)
             for i, r in enumerate(runs)]

  run_ids = [rec["run_id"] for rec in records]
  metrics = [k for k in records[0].keys() if k != "run_id"]
  df = pd.DataFrame(
      {rec["run_id"]: {m: rec[m] for m in metrics} for rec in records},
      index=metrics,
  )[run_ids]  # preserve newest-first column order

  pd.set_option("display.width", 200)
  pd.set_option("display.max_columns", None)
  print("\n=== Time/step table (run_ids on columns) ===")
  print(df.to_string())

  make_graph(df, run_ids)
  write_markdown(df, records)
  print(f"\nWrote {PNG_OUT}")
  print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
  main()
