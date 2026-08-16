#!/usr/bin/env python3
"""Factorial analysis of the saturating-reward PPO sweep (plan WP5).

Pulls the Tier A / Tier B runs from W&B and reports:

  * a ranking of every cell by final ``eval/episode_success``
  * the four main effects and six 2-way interactions of the 2^4 factorial
  * the A x C interaction explicitly (the plan's leading hypothesis)
  * measured throughput -> a Tier B ``num_timesteps`` that fits a time budget
  * the per-term reward decomposition, so a saturation shortfall can be
    attributed to a phase rather than guessed at

Read-only against W&B. Runs on any machine with network + a wandb login;
needs no GPU and does no MJX work, so it is safe on the Mac.

Why the per-term decomposition is trustworthy: ``ur3_pick.py`` logs each
reward term's RAW (unscaled) per-episode sum as ``eval/episode_<term>``.
Multiplying by ``reward_config.scales[<term>]`` and summing reproduces
``final_eval_return`` to within float noise -- the script asserts exactly
that (``--check-decomposition``) instead of assuming it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
from pathlib import Path

# Repo root resolved relative to THIS file -- never an absolute path, so the
# script behaves identically on the Mac, the robotics PC and the HPC.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENTITY = "weissma6-zhaw-school-of-engineering"
DEFAULT_PROJECT = "UR3_pick_ppo"

# The four binary factors, in the order they appear in the cell id `a b c d`.
FACTORS = [
    ("A", "policy_net", "(32,32,32,32)", "(256,256,256)"),
    ("B", "entropy_cost", "2e-2", "5e-3"),
    ("C", "learning_rate", "6e-4", "3e-4"),
    ("D", "reward_scaling", "0.05", "0.03"),
]

# brax's env_step_per_training_step for UR3Pick, from the plan's anchors:
# batch_size 512 * unroll_length 10 * num_minibatches 32 * num_resets_per_eval 1
ENV_STEP_PER_TRAINING_STEP = 512 * 10 * 32 * 1


def cell_ids():
    """The 16 factorial cell ids, in canonical (a,b,c,d) order."""
    return [
        f"a{a}b{b}c{c}d{d}"
        for a, b, c, d in itertools.product([0, 1], repeat=4)
    ]


def levels(cell):
    """'a1b0c1d0' -> dict(A=1, B=0, C=1, D=0)."""
    return {"A": int(cell[1]), "B": int(cell[3]), "C": int(cell[5]), "D": int(cell[7])}


# --------------------------------------------------------------------------
# W&B pull
# --------------------------------------------------------------------------

def fetch(entity, project, tier, cache_path=None, refresh=False):
    """Return {cell_id: [run_dict, ...]} plus the raw list, cached to JSON."""
    if cache_path and Path(cache_path).exists() and not refresh:
        with open(cache_path) as f:
            print(f"# using cache {cache_path} (--refresh to re-pull)")
            return json.load(f)

    import wandb  # imported lazily so --help works without wandb installed

    api = wandb.Api(timeout=60)
    path = f"{entity}/{project}"
    tag = f"tier{tier.upper()}"
    # Server-side filter on the cheap `satsweep` tag only, then narrow to the
    # tier in Python. Compound tag filters are rejected by the W&B GraphQL API
    # with a 500, which would look like an empty result rather than an error.
    runs = [r for r in api.runs(path, filters={"tags": {"$in": ["satsweep"]}})
            if tag in (r.tags or [])]
    print(f"# W&B: {len(runs)} runs tagged satsweep+{tag} in {path}")

    out = {}
    for r in runs:
        summary = {k: v for k, v in dict(r.summary).items()}
        # eval/episode_success and eval/episode_reward carry a {'max': ...}
        # summary dict (wandb define_metric summary="max"), so the FINAL value
        # has to come from the history, not the summary.
        # `history()` (sampled) not `scan_history()`: wandb's step index here IS
        # the env step, so scan_history paginates over a ~25M-wide range and
        # takes minutes per run. There are only num_evals (~20-40) eval rows, so
        # a 10k sample ceiling returns all of them exactly.
        hist = r.history(
            keys=["eval/episode_success", "eval/episode_reward"],
            samples=10_000, pandas=False)
        # run_experiment.py logs with the env step as wandb's own step, so
        # `_step` IS `training/num_steps`. Normalise so downstream code can read
        # one key regardless of which fetch path produced the row.
        for row in hist:
            row.setdefault("training/num_steps", row.get("_step"))
        rec = {
            "name": r.name,
            "id": r.id,
            "state": r.state,
            "created_at": str(r.created_at),
            "tags": list(r.tags or []),
            "url": r.url,
            "runtime_s": summary.get("_runtime"),
            "max_num_steps": summary.get("training/num_steps"),
            "final_eval_return": summary.get("final_eval_return"),
            "summary": {k: v for k, v in summary.items()
                        if isinstance(v, (int, float, str, bool, type(None)))},
            "history": hist,
        }
        # cell id is carried as its own tag by gen_satsweep.py
        cell = next((t for t in rec["tags"] if len(t) == 8 and t[0] == "a"), None)
        if cell is None:  # fall back to parsing the run name
            cell = r.name.split("_")[1]
        out.setdefault(cell, []).append(rec)
        print(f"#   {r.name}  ({len(hist)} eval rows)")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"# cached -> {cache_path}")
    return out


def pick_authoritative(runs):
    """If a cell was resubmitted, the newest run that reached the most steps wins."""
    return sorted(runs, key=lambda r: (r.get("max_num_steps") or 0,
                                       r.get("created_at") or ""))[-1]


def final_of(rec, key):
    """Last non-null value of `key` in the run's eval history."""
    vals = [row.get(key) for row in rec["history"] if row.get(key) is not None]
    return vals[-1] if vals else None


def best_of(rec, key):
    vals = [(row.get(key), row.get("training/num_steps"))
            for row in rec["history"] if row.get(key) is not None]
    return max(vals, key=lambda t: t[0]) if vals else (None, None)


def first_cross(rec, key, thresh):
    for row in rec["history"]:
        v = row.get(key)
        if v is not None and v >= thresh:
            return row.get("training/num_steps")
    return None


# --------------------------------------------------------------------------
# Reward-term decomposition
# --------------------------------------------------------------------------

def reward_scales():
    """The live reward scales from the env, or None if the env can't import.

    Read from the env rather than hardcoded: a hardcoded copy would silently
    go stale the moment a scale moves, and the whole point of the
    decomposition is to be checkable against `final_eval_return`.
    """
    # MUJOCO_GL is deliberately NOT set here: this script never renders, and the
    # backend is the machine's business (vault portability rule). JAX is pinned
    # to CPU because reading a config must never try to claim a GPU.
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from mujoco_playground._src.manipulation.my_ur3 import ur3_pick
    except Exception as e:  # pragma: no cover - depends on the machine's env
        print(f"# reward scales unavailable ({type(e).__name__}: {e});"
              " skipping the decomposition", file=sys.stderr)
        return None
    return dict(ur3_pick.default_config().reward_config.scales)


def decompose(rec, scales):
    """{term: (raw_sum, scale, contribution)} + the reconstruction residual."""
    if not scales:
        return None, None
    parts = {}
    for term, scale in scales.items():
        raw = rec["summary"].get(f"eval/episode_{term}")
        if raw is None:
            continue
        parts[term] = (raw, float(scale), raw * float(scale))
    total = sum(v[2] for v in parts.values())
    ref = rec.get("final_eval_return")
    residual = None if ref is None else total - ref
    return parts, residual


# --------------------------------------------------------------------------
# Factorial effects
# --------------------------------------------------------------------------

def main_effects(values):
    """values: {cell_id: metric}. Effect = mean(level 1) - mean(level 0)."""
    eff = {}
    for key, name, lo, hi in FACTORS:
        hi_v = [v for c, v in values.items() if levels(c)[key] == 1]
        lo_v = [v for c, v in values.items() if levels(c)[key] == 0]
        if hi_v and lo_v:
            eff[key] = {
                "name": name, "lo_label": lo, "hi_label": hi,
                "mean_lo": statistics.fmean(lo_v), "mean_hi": statistics.fmean(hi_v),
                "effect": statistics.fmean(hi_v) - statistics.fmean(lo_v),
                "n_lo": len(lo_v), "n_hi": len(hi_v),
            }
    return eff


def interactions(values):
    """Standard 2-level factorial interaction: half the difference of the
    simple effects of X across the two levels of Y."""
    out = {}
    keys = [f[0] for f in FACTORS]
    for x, y in itertools.combinations(keys, 2):
        quad = {}
        for xv, yv in itertools.product([0, 1], repeat=2):
            sel = [v for c, v in values.items()
                   if levels(c)[x] == xv and levels(c)[y] == yv]
            quad[(xv, yv)] = statistics.fmean(sel) if sel else None
        if any(v is None for v in quad.values()):
            continue
        simple_at_y0 = quad[(1, 0)] - quad[(0, 0)]   # effect of X when Y=0
        simple_at_y1 = quad[(1, 1)] - quad[(0, 1)]   # effect of X when Y=1
        out[f"{x}x{y}"] = {
            "quadrants": {f"{x}{xv}{y}{yv}": quad[(xv, yv)]
                          for xv, yv in itertools.product([0, 1], repeat=2)},
            "simple_effect_of_%s_at_%s0" % (x, y): simple_at_y0,
            "simple_effect_of_%s_at_%s1" % (x, y): simple_at_y1,
            "interaction": 0.5 * (simple_at_y1 - simple_at_y0),
        }
    return out


# --------------------------------------------------------------------------
# Tier B budget sizing
# --------------------------------------------------------------------------

def size_tier_b(steps, runtimes_s, budget_h, num_evals):
    """Measured throughput -> the largest brax-realisable budget under budget_h."""
    med_s = statistics.median(runtimes_s)
    slow_s = max(runtimes_s)
    rate_med = steps / (med_s / 3600.0)          # steps per hour
    rate_slow = steps / (slow_s / 3600.0)
    # Size against the SLOWEST cell, not the median: Tier B is an array job and
    # the wall clock is set by its worst member, not its typical one.
    affordable = rate_slow * budget_h
    n = num_evals - 1
    nts = max(1, math.floor(affordable / (n * ENV_STEP_PER_TRAINING_STEP)))
    total = n * nts * ENV_STEP_PER_TRAINING_STEP
    return {
        "median_runtime_s": med_s, "slowest_runtime_s": slow_s,
        "rate_median_per_h": rate_med, "rate_slowest_per_h": rate_slow,
        "budget_h": budget_h, "affordable_steps": affordable,
        "num_evals": num_evals, "nts": nts, "total_steps": total,
        "projected_h_slowest": total / rate_slow,
    }


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=["A", "B"], required=True)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--out", default=None, help="write the markdown report here")
    p.add_argument("--cache", default=None, help="JSON cache of the W&B pull")
    p.add_argument("--refresh", action="store_true", help="ignore the cache")
    p.add_argument("--expect", type=int, default=None,
                   help="fail if fewer than this many cells resolve")
    p.add_argument("--budget-hours", type=float, default=11.0,
                   help="wall-clock budget Tier B must fit inside")
    p.add_argument("--tier-b-evals", type=int, default=40)
    p.add_argument("--check-decomposition", action="store_true",
                   help="assert sum(term*scale) reconstructs final_eval_return")
    args = p.parse_args()

    by_cell = fetch(args.entity, args.project, args.tier, args.cache, args.refresh)
    expected = cell_ids()
    missing = [c for c in expected if c not in by_cell]
    dupes = {c: len(v) for c, v in by_cell.items() if len(v) > 1}

    L = []
    w = L.append
    w(f"# SatSweep Tier {args.tier} — factorial analysis\n")
    w(f"Source: W&B `{args.entity}/{args.project}`, tags `satsweep` + `tier{args.tier}`.\n")

    w(f"\n## Run inventory\n")
    w(f"- expected cells: **{len(expected)}**")
    w(f"- resolved: **{len(by_cell)}**")
    if missing:
        w(f"- **MISSING (hard failure): {', '.join(missing)}**")
    else:
        w("- missing: none")
    if dupes:
        w(f"- resubmitted cells (newest+furthest wins): {dupes}")

    recs, incomplete = {}, []
    for c in expected:
        if c not in by_cell:
            continue
        r = pick_authoritative(by_cell[c])
        recs[c] = r
        if r["state"] != "finished" or (r.get("max_num_steps") or 0) < 24_903_680:
            incomplete.append((c, r["state"], r.get("max_num_steps")))
    if incomplete:
        w(f"- **short of full budget:** {incomplete}")
    else:
        w("- all resolved runs `finished` at the full step budget")

    if args.expect is not None and len(recs) < args.expect:
        w(f"\n**FAIL: expected {args.expect} cells, resolved {len(recs)}.**")
        _emit(L, args.out)
        sys.exit(1)

    # ---- per-cell table -------------------------------------------------
    succ = {c: final_of(r, "eval/episode_success") for c, r in recs.items()}
    rew = {c: final_of(r, "eval/episode_reward") for c, r in recs.items()}
    succ = {c: v for c, v in succ.items() if v is not None}
    rew = {c: v for c, v in rew.items() if v is not None}

    w("\n## Cells, ranked by final `eval/episode_success`\n")
    w("| rank | cell | A policy | B ent | C lr | D rs | final success | best success | "
      "final reward | t_lift | t_grasp | t_reach | steps | runtime |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(recs, key=lambda c: (-(succ.get(c) or -1), -(rew.get(c) or -1)))
    for i, c in enumerate(order, 1):
        r = recs[c]
        lv = levels(c)
        bs, _ = best_of(r, "eval/episode_success")
        s = r["summary"]
        w("| {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            i, c,
            "256x3" if lv["A"] else "32x4",
            "5e-3" if lv["B"] else "2e-2",
            "3e-4" if lv["C"] else "6e-4",
            "0.03" if lv["D"] else "0.05",
            _f(succ.get(c), 3), _f(bs, 3), _f(rew.get(c), 0),
            _f(s.get("eval/episode_t_lift"), 1),
            _f(s.get("eval/episode_t_grasp"), 1),
            _f(s.get("eval/episode_t_reach"), 1),
            f"{r.get('max_num_steps'):,}" if r.get("max_num_steps") else "?",
            _hm(r.get("runtime_s"))))

    # ---- main effects ---------------------------------------------------
    for label, values, fmt in (("eval/episode_success", succ, 4),
                               ("eval/episode_reward", rew, 0)):
        if not values:
            continue
        w(f"\n## Main effects on `{label}`\n")
        w("| factor | parameter | level 0 | level 1 | mean@0 | mean@1 | effect (1−0) |")
        w("|---|---|---|---|---|---|---|")
        for k, e in main_effects(values).items():
            w("| **{}** | {} | {} | {} | {} | {} | **{}** |".format(
                k, e["name"], e["lo_label"], e["hi_label"],
                _f(e["mean_lo"], fmt), _f(e["mean_hi"], fmt), _f(e["effect"], fmt)))

        w(f"\n### 2-way interactions on `{label}`\n")
        w("| pair | interaction | simple effects |")
        w("|---|---|---|")
        ints = interactions(values)
        for name, d in sorted(ints.items(), key=lambda kv: -abs(kv[1]["interaction"])):
            x, y = name.split("x")
            w("| **{}** | {} | eff({}) at {}0 = {} · at {}1 = {} |".format(
                name, _f(d["interaction"], fmt), x, y,
                _f(d[f"simple_effect_of_{x}_at_{y}0"], fmt), y,
                _f(d[f"simple_effect_of_{x}_at_{y}1"], fmt)))

        if "AxC" in ints and label == "eval/episode_success":
            d = ints["AxC"]
            w("\n**A×C verdict (the plan's leading hypothesis).** The plan predicts a "
              "wide actor helps only at the lower learning rate.\n")
            w("| | C0 (lr 6e-4) | C1 (lr 3e-4) |")
            w("|---|---|---|")
            w("| **A0** (32×4) | {} | {} |".format(
                _f(d["quadrants"]["A0C0"], fmt), _f(d["quadrants"]["A0C1"], fmt)))
            w("| **A1** (256×3) | {} | {} |".format(
                _f(d["quadrants"]["A1C0"], fmt), _f(d["quadrants"]["A1C1"], fmt)))
            e0 = d["simple_effect_of_A_at_C0"]
            e1 = d["simple_effect_of_A_at_C1"]
            if e1 > 0 and e0 <= 0:
                v = "CONFIRMED — the wide actor helps at 3e-4 and hurts at 6e-4."
            elif e1 > e0 and e1 > 0:
                v = "PARTIALLY CONFIRMED — the wide actor helps more at 3e-4, but not only there."
            elif e0 > 0 and e1 > 0:
                v = "REFUTED (direction) — the wide actor helps at BOTH learning rates."
            elif e0 <= 0 and e1 <= 0:
                v = "REFUTED — the wide actor hurts at both learning rates."
            else:
                v = "REFUTED — the wide actor helps at 6e-4 and hurts at 3e-4, the opposite of the prediction."
            w(f"\n**Verdict: {v}**")

    # ---- reward decomposition ------------------------------------------
    scales = reward_scales()
    if scales:
        w("\n## Reward decomposition — where the return actually comes from\n")
        w("`eval/episode_<term>` is the RAW per-episode sum; contribution = raw × scale.\n")
        ref_cell = order[0]
        w(f"Best cell `{ref_cell}` vs baseline `a0b0c0d0`:\n")
        w("| term | scale | raw (best) | contrib (best) | % | contrib (a0b0c0d0) | % |")
        w("|---|---|---|---|---|---|---|")
        pb, resb = decompose(recs[ref_cell], scales)
        p0, res0 = (decompose(recs["a0b0c0d0"], scales)
                    if "a0b0c0d0" in recs else (None, None))
        tb = sum(v[2] for v in pb.values()) if pb else 0
        t0 = sum(v[2] for v in p0.values()) if p0 else 0
        for term in sorted(pb or {}, key=lambda t: -abs(pb[t][2])):
            raw, sc, con = pb[term]
            c0 = p0.get(term, (None, None, None))[2] if p0 else None
            w("| `{}` | {} | {} | {} | {} | {} | {} |".format(
                term, sc, _f(raw, 1), _f(con, 0),
                _f(100 * con / tb, 1) if tb else "—",
                _f(c0, 0), _f(100 * c0 / t0, 1) if (c0 is not None and t0) else "—"))
        w("| **total** | | | **{}** | | **{}** | |".format(_f(tb, 0), _f(t0, 0)))
        w(f"\nReconstruction residual vs `final_eval_return`: best `{_f(resb, 2)}`, "
          f"baseline `{_f(res0, 2)}` (should be ~0 — this is what makes the split trustworthy).")
        if args.check_decomposition:
            for c, r in recs.items():
                _, res = decompose(r, scales)
                ref = r.get("final_eval_return") or 1.0
                assert res is not None and abs(res) < max(1.0, 0.001 * abs(ref)), \
                    f"decomposition does not reconstruct final_eval_return for {c}: residual {res}"
            w("\n`--check-decomposition`: **PASS** — every cell reconstructs to <0.1%.")

    # ---- throughput / Tier B sizing -------------------------------------
    rts = [r["runtime_s"] for r in recs.values() if r.get("runtime_s")]
    steps_each = [r["max_num_steps"] for r in recs.values() if r.get("max_num_steps")]
    if rts and steps_each:
        steps = statistics.median(steps_each)
        b = size_tier_b(steps, rts, args.budget_hours, args.tier_b_evals)
        w("\n## Measured throughput → Tier B budget\n")
        w(f"Runtime is taken from each run's W&B `_runtime`, so it is measured, not assumed.\n")
        w("| quantity | value |")
        w("|---|---|")
        w(f"| steps per run (Tier A) | {int(steps):,} |")
        w(f"| median runtime | {_hm(b['median_runtime_s'])} |")
        w(f"| slowest runtime | {_hm(b['slowest_runtime_s'])} |")
        w(f"| median throughput | {b['rate_median_per_h']/1e6:.1f} M steps/h |")
        w(f"| **slowest throughput** (sizing basis) | **{b['rate_slowest_per_h']/1e6:.1f} M steps/h** |")
        w(f"| wall-clock budget | {b['budget_h']:.1f} h |")
        w(f"| affordable at the slowest rate | {b['affordable_steps']/1e6:.1f} M steps |")
        w(f"| Tier B `num_evals` | {b['num_evals']} |")
        w(f"| solved `nts` | {b['nts']} |")
        w(f"| **Tier B TOTAL_STEPS** | **{b['total_steps']:,}** |")
        w(f"| projected wall clock at the slowest rate | {b['projected_h_slowest']:.2f} h |")
        w("\nSized against the **slowest** cell, not the median: an array job's wall "
          "clock is set by its worst member.")

    # ---- top 4 ----------------------------------------------------------
    w("\n## Top 4 cells for Tier B\n")
    w("Ranked by final `eval/episode_success`, tie-broken by final `eval/episode_reward`.\n")
    for i, c in enumerate(order[:4], 1):
        w(f"{i}. `{c}` — success {_f(succ.get(c), 3)}, reward {_f(rew.get(c), 0)}")
    w(f"\n`--cells {','.join(order[:4])}`")

    _emit(L, args.out)


def _f(v, nd=3):
    if v is None:
        return "—"
    return f"{v:,.{nd}f}"


def _hm(s):
    if not s:
        return "—"
    s = float(s)
    return f"{int(s//3600)}h{int((s%3600)//60):02d}m"


def _emit(lines, out):
    text = "\n".join(lines)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(text + "\n")
        print(f"\n# written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
