# batch_runs/scripts/run_one_ur10.py
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from learning.notebooks.UR10_ppo_2 import run_experiment

def _read_jsonl_line(path: Path, line_no: int) -> Dict[str, Any]:
    if line_no < 0:
        raise ValueError(f"line_no must be >= 0, got {line_no}")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == line_no:
                line = line.strip()
                if not line:
                    raise ValueError(f"Empty JSONL line at {path}:{i+1}")
                return json.loads(line)
    raise IndexError(f"Index out of range: requested line {line_no} but file ended earlier: {path}")
def _collect_slurm_info() -> Dict[str, Any]:
    env = os.environ
    keys = [
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT",
        "SLURM_SUBMIT_DIR",
        "SLURM_NODELIST",
        "SLURMD_NODENAME",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "HOSTNAME",
    ]
    info: Dict[str, Any] = {k: env.get(k) for k in keys if env.get(k) is not None}
    return info
def _make_run_id(index: int, slurm_info: Dict[str, Any]) -> str:
    job_id = slurm_info.get("SLURM_JOB_ID")
    task_id = slurm_info.get("SLURM_ARRAY_TASK_ID")
    if job_id is not None and task_id is not None:
        return f"{job_id}_{int(task_id):04d}"
    if job_id is not None:
        return f"{job_id}_{index:06d}"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"local_{ts}_{index:06d}"
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str, required=True, help="Path to sweep JSONL")
    p.add_argument("--index", type=int, required=True, help="Sweep index coming from SLURM array task id or local index")
    p.add_argument("--out_root", type=str, default="results", help="Root output directory")
    p.add_argument("--one_based", action="store_true", help="Treat --index as 1-based (default if set)")
    p.add_argument("--zero_based", action="store_true", help="Treat --index as 0-based (overrides --one_based)")
    return p.parse_args()
def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    out_root = Path(args.out_root)
    raw_index = int(args.index)
    if args.zero_based:
        line_no = raw_index
    else:
        # default behavior: if you pass SLURM_ARRAY_TASK_ID starting at 1, use 1-based
        line_no = raw_index - 1 if (args.one_based or raw_index >= 1) else raw_index
    sweep_cfg = _read_jsonl_line(jsonl_path, line_no)
    slurm_info = _collect_slurm_info()
    run_id = _make_run_id(raw_index, slurm_info)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "index": raw_index,
        "line_no": line_no,
        "run_id": run_id,
        "sweep_jsonl": str(jsonl_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm": slurm_info,
        "config": sweep_cfg,
    }
    (run_dir / "resolved_run.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    # Import here to keep this script minimal and avoid importing MuJoCo/JAX unless needed.
    # Adjust the import path to match your repo.

    # Pass ONLY external references + sweep config; env loading happens inside run_experiment.
    run_experiment(
        sweep_cfg,
        index=raw_index,
        run_id=run_id,
        slurm_info=slurm_info,
        out_dir=str(run_dir),
    )
if __name__ == "__main__":
    main()
