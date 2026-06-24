#!/usr/bin/env python3
"""Run a single Panda experiment from a JSONL sweep file."""

import argparse
import json
import os
from learning.notebooks.run_experiment import run_experiment


def load_config(jsonl_path: str, index_1based: int) -> dict:
    """Load a specific config line from a JSONL file (1-indexed)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        i = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            i += 1
            if i == index_1based:
                return json.loads(line)
    raise IndexError(f"Index {index_1based} out of range for {jsonl_path}")


def main():
    ap = argparse.ArgumentParser(description="Run one Panda experiment from sweep")
    ap.add_argument("--jsonl", required=True, help="Path to JSONL sweep file")
    ap.add_argument("--index", type=int, required=True, help="1-based index into JSONL")
    ap.add_argument("--out_root", default="results", help="Output directory root")
    args = ap.parse_args()

    cfg = load_config(args.jsonl, args.index)
    run_id = cfg.get("run_id", f"run_{args.index:04d}")
    out_dir = os.path.join(args.out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Running experiment: {run_id}")
    print(f"Config: {cfg}")
    print(f"Output dir: {out_dir}")

    run_experiment(cfg=cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
