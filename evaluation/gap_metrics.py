"""Sim-to-real gap metrics over a tidy long-format results table.

The table (gap_metrics.csv) is LONG: one row per
(config_id, seed, episode_id, repeat, cube_size, term, domain) with the
episode-summed scaled reward `scaled_return_H` for that term, plus per-run
bookkeeping (achieved_hz, overrun_count, stop_reason, protocol_hash).
`domain` is "sim" or "real". A run's total return is the sum of its term
rows; a term's stage is fixed by STAGE_MAP.

`cube_size` (D17, "3cm"/"4cm") is a grouping key like `config_id`, NOT an
axis to average over: 3cm is the in-distribution probe, 4cm is a deliberate
OOD probe (outside even the training-time size-DR clamp), and averaging the
two together would silently blend an in-distribution result with an OOD one.
Every function below that pools rows into a mean therefore takes an optional
`cube_size=` filter, and REFUSES to run (raises) if the input has more than
one distinct `cube_size` and the caller did not pin one -- see
`_filter_cube_size`. This is the code-level enforcement of D17's "3cm and 4cm
must always be reported/grouped separately, never averaged together".

Deliberate NON-features (this project's primary result is a LEVEL SHIFT between
sim and real return, and these would hide or misreport it):

  * NO normalisation, anywhere. z-scoring / min-max / per-config rescaling would
    destroy the level shift that is the headline finding. Every ratio and gap
    below is computed on RAW returns.
  * NO correlation coefficients: no MMRV, SRCC (policy or state), Pearson, or
    Kendall. They are invariant to exactly the level shift we are measuring and
    are expected near-zero by the project's own hypothesis, so a low value would
    misreport the finding as a defect. There is NO scipy.stats correlation
    import anywhere under evaluation/ (the --selftest enforces this by scanning
    the source tree).

All CIs are paired bootstrap, RESAMPLING EPISODES (never steps): episodes are
the exchangeable unit; the protocol evaluates every config/seed/repeat on the
SAME episode set, so a resample draws episode_ids and recomputes the statistic
from all rows of those episodes.

Local usage:
    python evaluation/gap_metrics.py --selftest
    python evaluation/gap_metrics.py --csv gap_metrics.csv --config L2_pos_cube
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed taxonomy (NAMES only -- no reward SCALES are hardcoded here; the CSV
# already carries scaled returns). Order + stages match the plots (F2).
# ---------------------------------------------------------------------------
TERM_ORDER = [
    "gripper_box", "approach_open", "gripper_align", "grasp", "lift",
    "box_target", "hold_target", "no_floor_collision", "robot_target_qpos",
    "action_rate",
]
STAGE_MAP = {
    "gripper_box": "approach", "approach_open": "approach",
    "gripper_align": "approach", "grasp": "grasp", "lift": "lift",
    "box_target": "transport", "hold_target": "transport",
    "no_floor_collision": "regularizer", "robot_target_qpos": "regularizer",
    "action_rate": "regularizer",
}
STAGE_ORDER = ["approach", "grasp", "lift", "transport", "regularizer"]
# Terms that are PROXIES on the real robot (no direct sensor -> supplied via a
# proxy signal). Flagged for the diagnostic caption in F2; not treated
# differently numerically.
PROXY_TERMS = {"grasp", "no_floor_collision"}
# D8: the "exact-parity" headline subset -- everything geometrically exact,
# i.e. TERM_ORDER minus the proxies above. This is the subset the per-step
# reward trace (plots_gap.f3_per_step_trace) is built from.
EXACT_PARITY_TERMS = [t for t in TERM_ORDER if t not in PROXY_TERMS]

LONG_COLUMNS = [
    "config_id", "seed", "episode_id", "repeat", "cube_size", "term", "domain",
    "scaled_return_H", "achieved_hz", "overrun_count", "stop_reason",
    "protocol_hash",
]

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# Load / validate
# ===========================================================================


def load_long(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in ("config_id", "domain", "episode_id", "term",
                           "scaled_return_H") if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    if not set(df["domain"].unique()) <= {"sim", "real"}:
        raise ValueError("domain must be in {'sim','real'}")
    return df


def _run_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Sum term rows -> one total per (config, domain, seed, episode, repeat,
    cube_size). `cube_size` MUST be a grouping key here, not just a filter
    upstream: real repeat numbers restart at 1 for each cube_size (D16/D17's
    campaign schedule), so "episode_id, repeat" alone collides a 3cm and a
    4cm run of the same repeat index -- dropping cube_size from the group-by
    would silently sum/pool their term rows into one fabricated "run".
    """
    keys = ["config_id", "domain", "seed", "episode_id", "repeat", "cube_size"]
    keys = [k for k in keys if k in df.columns]
    return (
        df.groupby(keys, dropna=False)["scaled_return_H"].sum()
        .reset_index().rename(columns={"scaled_return_H": "total"})
    )


def _filter_cube_size(df: pd.DataFrame, cube_size, fn_name: str) -> pd.DataFrame:
    """D17 enforcement: 3cm (in-distribution) and 4cm (OOD probe) must never
    be silently averaged together. If `cube_size` is given, filter to it
    (no-op if the column is absent). If not given and the column IS present
    with more than one distinct value, raise -- forces the caller to pick one
    explicitly instead of pooling an in-distribution result with an OOD one.
    """
    if "cube_size" not in df.columns:
        return df
    if cube_size is not None:
        return df[df["cube_size"] == cube_size]
    values = sorted(df["cube_size"].dropna().unique())
    if len(values) > 1:
        raise ValueError(
            f"{fn_name}: input has multiple cube_size values {values} and no "
            f"cube_size= was given -- refusing to silently average 3cm "
            f"(in-distribution) with 4cm (OOD probe, D17). Pass "
            f"cube_size='3cm' or cube_size='4cm' explicitly."
        )
    return df


def per_episode_returns(
    df: pd.DataFrame, config: str, episodes=None, cube_size=None
) -> pd.DataFrame:
    """Paired per-episode returns for one config (+ one cube_size, D17).

    Returns a DataFrame indexed by episode_id with columns ['sim','real'] --
    each the mean total return over that episode's seeds (sim) / repeats (real).
    Only episodes present in BOTH domains are kept (pairing requirement).
    """
    sub = _filter_cube_size(df[df["config_id"] == config], cube_size,
                             "per_episode_returns")
    tot = _run_totals(sub)
    if episodes is not None:
        tot = tot[tot["episode_id"].isin(episodes)]
    piv = (
        tot.groupby(["episode_id", "domain"])["total"].mean()
        .unstack("domain")
    )
    for d in ("sim", "real"):
        if d not in piv.columns:
            piv[d] = np.nan
    piv = piv[["sim", "real"]].dropna()
    return piv


# ===========================================================================
# Bootstrap core (resamples EPISODES)
# ===========================================================================


def _boot_episode_indices(n_ep: int, n_boot: int, rng) -> np.ndarray:
    return rng.integers(0, n_ep, size=(n_boot, n_ep))


def _percentile_ci(samples, lo=2.5, hi=97.5):
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


# ===========================================================================
# Metrics
# ===========================================================================


def retention(df, config, subset="exact", episodes=None, cube_size=None,
              n_boot=10000, seed=0):
    """Real/sim return retention for one config + paired-bootstrap 95% CI.

    ratio = mean_episode(real) / mean_episode(sim), on RAW returns (no
    normalisation -- see module docstring). CI resamples episodes.
    `subset` is a label recorded in the result (e.g. "exact" = exact-init
    matched episodes); pass `episodes` to restrict to a subset's episode_ids.
    `cube_size` (D17): pin to "3cm"/"4cm" when `df` mixes both -- see
    `_filter_cube_size`.
    """
    piv = per_episode_returns(df, config, episodes=episodes, cube_size=cube_size)
    real = piv["real"].to_numpy()
    sim = piv["sim"].to_numpy()
    n = len(piv)
    point = float(np.mean(real) / np.mean(sim))

    rng = np.random.default_rng(seed)
    idx = _boot_episode_indices(n, n_boot, rng)
    ratios = np.mean(real[idx], axis=1) / np.mean(sim[idx], axis=1)
    lo, hi = _percentile_ci(ratios)
    return {
        "config_id": config, "subset": subset, "cube_size": cube_size,
        "n_episodes": n,
        "R_sim": float(np.mean(sim)), "R_real": float(np.mean(real)),
        "retention": point, "ci95": (lo, hi),
    }


def noise_floor(df, config, episodes=None, cube_size=None):
    """D9's measurement floor -- spread of the k repeats' PAIRED sim-real
    gaps (corrected 2026-07-22), NOT the spread of the k raw real returns.

    Why pairing matters: a D9 repeat holds the arm pose and lift target
    fixed and re-places ONLY the physical cube, so the k repeats' real
    returns differ for two reasons that are otherwise inseparable: (1) the
    cube landed at a slightly different spot on the marker each time
    (placement-to-placement variance -- a property of the WORKSPACE, not a
    measurement artifact), and (2) genuine run-to-run measurement noise
    (control-loop timing, actuation, contact dynamics -- what "measurement
    resolution" should mean). run_gap_protocol_sim.py mirrors each real
    repeat's OWN measured placement into its own deterministic sim rollout
    (not one sim rollout per episode) specifically so this can be
    disentangled: for repeat r, `g_r = R_real_r - R_sim_r` subtracts a
    deterministic sim rollout that saw the IDENTICAL placement, so the
    placement effect present in both `R_real_r` and `R_sim_r` cancels out of
    `g_r`, leaving only the irreducible real-side noise. The floor is then
    the spread (std) of the k paired `g_r`, averaged over episodes -- the
    OLD computation (std of the k raw `R_real_r`) left the placement
    variance in and so overstated the floor by however much the workspace
    itself varies placement to placement.

    Sim rows are matched to each real repeat by (episode_id, repeat) (in
    addition to the config_id/cube_size filter already applied). If the
    input's sim domain has NO per-repeat granularity for an episode (e.g.
    only a repeat=0 seed-ensemble row, no real-repeat mirror was rolled
    out), that episode's single available sim value is reused as a
    repeat-INVARIANT stand-in for every one of its repeats' pairing.
    Subtracting the SAME constant from every repeat never changes a spread
    (std(x - c) == std(x)), so this fallback makes the "paired" floor
    numerically identical to the old raw-return floor -- the honest
    behaviour when no genuine per-repeat sim mirror exists to cancel
    anything against, not a bug. See --selftest for a case with genuine
    per-repeat sim data, where the two floors provably differ.

    `cube_size` (D17): pin to "3cm"/"4cm" when `df` mixes both -- see
    `_filter_cube_size`.
    """
    real = _filter_cube_size(
        df[(df["config_id"] == config) & (df["domain"] == "real")],
        cube_size, "noise_floor",
    )
    sim = _filter_cube_size(
        df[(df["config_id"] == config) & (df["domain"] == "sim")],
        cube_size, "noise_floor",
    )
    real_tot = _run_totals(real)  # one row per (episode_id, repeat[, cube_size])
    if episodes is not None:
        real_tot = real_tot[real_tot["episode_id"].isin(episodes)]

    sim_tot = _run_totals(sim)
    # Genuine per-repeat sim mirror, where present: mean sim total for the
    # EXACT (episode_id, repeat) pair (mean is a no-op unless the input has
    # >1 sim row for that exact pair, e.g. duplicate/seed rows).
    sim_per_repeat = sim_tot.groupby(["episode_id", "repeat"])["total"].mean()
    # Fallback: episode-level mean sim total, ignoring repeat -- used only
    # when no exact (episode_id, repeat) match exists (see docstring).
    sim_per_episode = sim_tot.groupby("episode_id")["total"].mean()

    def _matched_sim(episode_id, repeat):
        key = (episode_id, repeat)
        if key in sim_per_repeat.index:
            return float(sim_per_repeat.loc[key])
        if episode_id in sim_per_episode.index:
            return float(sim_per_episode.loc[episode_id])
        return np.nan

    real_tot = real_tot.copy()
    real_tot["sim_matched"] = [
        _matched_sim(ep, rp)
        for ep, rp in zip(real_tot["episode_id"], real_tot["repeat"])
    ]
    real_tot["paired_gap"] = real_tot["total"] - real_tot["sim_matched"]

    per_ep = real_tot.groupby("episode_id").agg(
        mean_gap=("paired_gap", "mean"), gap_std=("paired_gap", "std"),
        count=("paired_gap", "count"),
        min_gap=("paired_gap", "min"), max_gap=("paired_gap", "max"),
    )
    per_ep["gap_std"] = per_ep["gap_std"].fillna(0.0)
    floor = float(per_ep["gap_std"].mean())
    return {
        "config_id": config, "cube_size": cube_size,
        "noise_floor": floor,                 # mean within-episode PAIRED-GAP std (D9)
        "noise_floor_band": (-floor, floor),  # symmetric reportability band
        "per_episode": per_ep,
        "mean_repeats": float(per_ep["count"].mean()),
    }


def abort_rate(df, config, cube_size):
    """Fraction of REAL runs with stop_reason == "abort" for (config,
    cube_size) -- a reported result in its own right (D16/D17): a policy
    that protective-stops or loses the cube often is telling you something,
    and D17 explicitly expects this to be HIGH for the 4cm OOD cube (Hand-E
    clearance drops to ~0.5cm/side at 4cm, right at the mechanical grasp
    limit). Real-only: sim rollouts never abort (run_gap_protocol_sim.py's
    stop_reason is always "horizon_complete" -- D16's abort concept, hang
    watchdog / RTDE error / mocap loss, is a real-robot-only failure mode).

    `cube_size` has NO default (unlike the other functions' optional
    cube_size=) -- an abort rate is only ever meaningful FOR one physical
    cube (D17: "report the abort/failure rate per (policy, cube)"), so there
    is no sensible "unpinned" call to fall back to.

    Counted at the RUN level: one (episode_id, repeat) pair is one run, and
    every one of its ~10 term rows in the long table carries that same run's
    stop_reason, so de-duplicating to runs before counting (rather than
    counting raw rows) is the honest computation, even though the ratio
    would come out identical either way (every run contributes the same
    row-multiplicity to numerator and denominator).
    """
    sub = df[(df["config_id"] == config) & (df["domain"] == "real")]
    sub = _filter_cube_size(sub, cube_size, "abort_rate")
    if "stop_reason" not in sub.columns:
        raise ValueError("abort_rate: no 'stop_reason' column in df.")
    key_cols = [c for c in ("episode_id", "repeat") if c in sub.columns]
    runs = (
        sub.drop_duplicates(subset=key_cols)[key_cols + ["stop_reason"]]
        if key_cols else sub[["stop_reason"]].drop_duplicates()
    )
    n = len(runs)
    n_abort = int((runs["stop_reason"] == "abort").sum()) if n else 0
    return {
        "config_id": config, "cube_size": cube_size,
        "abort_rate": float(n_abort / n) if n else float("nan"),
        "n_runs": n, "n_aborted": n_abort,
    }


def term_retention(df, config, episodes=None, cube_size=None):
    """Per-term AND per-stage real/sim retention ratios for one config.

    Each ratio = mean real term (stage) return / mean sim term (stage) return,
    over episodes and seeds/repeats. Raw returns, no normalisation.
    `cube_size` (D17): pin to "3cm"/"4cm" when `df` mixes both -- see
    `_filter_cube_size`.
    """
    sub = _filter_cube_size(df[df["config_id"] == config], cube_size,
                             "term_retention")
    if episodes is not None:
        sub = sub[sub["episode_id"].isin(episodes)]
    means = (
        sub.groupby(["term", "domain"])["scaled_return_H"].mean()
        .unstack("domain")
    )
    for d in ("sim", "real"):
        if d not in means.columns:
            means[d] = np.nan
    means["stage"] = means.index.map(STAGE_MAP)
    means["is_proxy"] = means.index.isin(PROXY_TERMS)
    means["retention"] = means["real"] / means["sim"]

    stage = (
        sub.assign(stage=sub["term"].map(STAGE_MAP))
        .groupby(["stage", "domain"])["scaled_return_H"].mean()
        .unstack("domain")
    )
    for d in ("sim", "real"):
        if d not in stage.columns:
            stage[d] = np.nan
    stage["retention"] = stage["real"] / stage["sim"]

    order = [t for t in TERM_ORDER if t in means.index]
    return {
        "config_id": config, "cube_size": cube_size,
        "per_term": means.reindex(order),
        "per_stage": stage.reindex([s for s in STAGE_ORDER if s in stage.index]),
    }


def _config_returns_matrix(df, configs):
    """(episode_ids, R_sim[config x ep], R_real[config x ep]) on a shared episode set."""
    pivs = {c: per_episode_returns(df, c) for c in configs}
    common = None
    for p in pivs.values():
        s = set(p.index)
        common = s if common is None else (common & s)
    common = sorted(common)
    if not common:
        raise ValueError("configs share no common episodes -- cannot pair.")
    sim = np.array([pivs[c].loc[common, "sim"].to_numpy() for c in configs])
    real = np.array([pivs[c].loc[common, "real"].to_numpy() for c in configs])
    return np.array(common), sim, real


def sim_selection_regret(df, configs=None, cube_size=None, n_boot=10000, seed=0):
    """Sim-selection regret R = R_real(c*) - R_real(argmax_c R_sim).

    c*      = argmax_c R_real (the config you WOULD have picked with real data)
    chat_sim = argmax_c R_sim (the config you DO pick from sim alone)
    R >= 0; it is the real-return you lose by selecting on sim.

    Paired bootstrap resamples episodes and RECOMPUTES BOTH ARGMAXES inside each
    resample -- fixing chat_sim at the point estimate drops selection
    uncertainty (which dominates at ~6 configs) and understates the CI. Both the
    resampled-argmax CI and the (narrower) fixed-chat_sim CI are returned so the
    understatement is visible; the reportable CI is the resampled one.

    `cube_size` (D17): pin to "3cm"/"4cm" when `df` mixes both -- see
    `_filter_cube_size`. The config *set* to select over is always the 4
    D14-surviving real-deployed rungs (or whatever `configs` names); cube_size
    only says which physical-cube dataset that selection is scored against.
    """
    df = _filter_cube_size(df, cube_size, "sim_selection_regret")
    if configs is None:
        configs = sorted(df["config_id"].unique())
    configs = list(configs)
    eps, sim, real = _config_returns_matrix(df, configs)

    R_sim = sim.mean(axis=1)
    R_real = real.mean(axis=1)
    chat = int(np.argmax(R_sim))
    cstar = int(np.argmax(R_real))
    regret = float(R_real[cstar] - R_real[chat])

    rng = np.random.default_rng(seed)
    n_ep = len(eps)
    idx = _boot_episode_indices(n_ep, n_boot, rng)
    reg_full = np.empty(n_boot)
    reg_fixed = np.empty(n_boot)
    chat_counts = np.zeros(len(configs), dtype=int)
    for b in range(n_boot):
        cols = idx[b]
        Rs = sim[:, cols].mean(axis=1)
        Rr = real[:, cols].mean(axis=1)
        chat_b = int(np.argmax(Rs))       # selection recomputed -> full CI
        chat_counts[chat_b] += 1
        best_real = float(Rr.max())
        reg_full[b] = best_real - Rr[chat_b]
        reg_fixed[b] = best_real - Rr[chat]   # chat FIXED at point estimate

    ci_full = _percentile_ci(reg_full)
    ci_fixed = _percentile_ci(reg_fixed)
    table = pd.DataFrame({
        "config_id": configs, "R_sim": R_sim, "R_real": R_real,
    }).sort_values("R_sim", ascending=False).reset_index(drop=True)

    return {
        "configs": configs, "cube_size": cube_size,
        "chat_sim": configs[chat], "c_star": configs[cstar],
        "regret": regret,
        "ci95_resampled_argmax": ci_full,   # REPORTABLE
        "ci95_fixed_chat_sim": ci_fixed,    # understated (for comparison)
        "ci_width_resampled": ci_full[1] - ci_full[0],
        "ci_width_fixed": ci_fixed[1] - ci_fixed[0],
        "selection_freq": {
            configs[i]: float(chat_counts[i] / n_boot) for i in range(len(configs))
        },
        "table": table,
        "n_episodes": n_ep,
        # Raw resampled-argmax bootstrap distribution (n_boot floats) -- the
        # REPORTABLE one (ci95_resampled_argmax is just its 2.5/97.5
        # percentiles). Exposed so evaluation/plots_gap.py's F5 panel (b) can
        # histogram/KDE it directly instead of re-running the bootstrap.
        "regret_samples": reg_full,
    }


def gate_against_noise_floor(value, ci, floor):
    """Reportability gate -- every reported number passes through here (D9).

    Returns "reportable" iff the effect's 95% CI lies ENTIRELY outside the
    symmetric noise-floor band (-floor, +floor); otherwise "within_noise_floor"
    (indistinguishable from measurement noise -> not reportable). `value`, `ci`
    and `floor` must be in consistent units (e.g. return units, or a ratio
    deviation from 1). No normalisation is applied.
    """
    lo, hi = ci
    floor = abs(float(floor))
    if lo > floor or hi < -floor:
        return "reportable"
    return "within_noise_floor"


# ===========================================================================
# D22 (2026-07-29) -- drop-in-square eval metric.
#
# DELIBERATELY SEPARATE from R_real / the gap-scoring machinery above.
# place_error scores WHERE THE BOX LANDED after the D22 drop: the policy
# carries the cube to the eval target (D24 lowered target_z_jitter's floor to
# 0.05 precisely so that target sits inside the trained distribution), the
# eval script opens the gripper, and the cube falls ~50 mm into a taped
# square. It has nothing to do with scaled_return_H and must NEVER be blended
# into retention / noise_floor / sim_selection_regret -- report it on its own.
#
# There is no sim-side equivalent: MJX has no physical release event to
# script, so place_error is real-mocap-only (see run_gap_protocol_sim.py).
# ===========================================================================

# D22 drop square: 15x15 cm taped on the table, centred at base-frame
# (0.212, 0.212) -- r = sqrt(0.212^2 + 0.212^2) = 0.2998 ~= 0.30 at azim
# 45 deg, i.e. exactly the L0 fixed target's polar coordinates. Half-width
# 0.075 m. Well inside the real table (x in [0.10, 0.70], y in [-0.50, 0.50]).
DROP_SQUARE_CENTER = (0.212, 0.212)
DROP_SQUARE_HALF_WIDTH = 0.075


def place_error(landed_xy, square_center=DROP_SQUARE_CENTER,
                half_width=DROP_SQUARE_HALF_WIDTH):
    """D22: how far the dropped box landed from the taped square's centre.

    Args:
      landed_xy: (x, y) base-frame metres of the landed box centre, read from
        mocap AFTER the drop settles -- NOT the commanded lift target.
      square_center: (x, y) metres, taped-square centre.
      half_width: metres, half the square's side length.

    Returns:
      (distance, in_square): euclidean distance (m) to square_center, and
      whether the landing falls inside the square. The membership test is
      AXIS-ALIGNED (|dx| and |dy| against half_width), matching how the square
      is physically taped -- deliberately not a radial test.
    """
    lx, ly = float(landed_xy[0]), float(landed_xy[1])
    cx, cy = float(square_center[0]), float(square_center[1])
    dx, dy = lx - cx, ly - cy
    distance = float((dx * dx + dy * dy) ** 0.5)
    in_square = abs(dx) <= half_width and abs(dy) <= half_width
    return distance, bool(in_square)


# ===========================================================================
# Synthetic long-format data (shared with the plots' --demo and the selftest)
# ===========================================================================


def make_synthetic_long(
    config_specs, n_episodes=10, seeds=(0, 1, 2), repeats=(1, 2, 3),
    ep_noise=0.0, repeat_noise=0.0, seed_noise=0.0, protocol_hash="demo_hash",
    cube_size="3cm", abort_frac=0.0, rng=None,
):
    """Fabricate a plausible long-format table.

    config_specs: {config_id: {"sim_total": float, "real_total": float,
                               "term_frac": {term: frac} (optional)}}.
    Per-episode base drawn N(total, ep_noise); sim seeds jitter by seed_noise;
    real repeats jitter by repeat_noise. Totals are split across terms by
    term_frac (default: an approach-heavy split summing to 1).

    `cube_size` (D17): every row is tagged with this single value (default
    "3cm", the in-distribution probe). Callers that want a mixed-cube table
    (e.g. plots_gap.py's --demo) call this twice -- once per cube_size -- and
    concatenate; this function itself never mixes cube sizes in one call, so
    its own output never trips `_filter_cube_size`'s "which cube?" guard.

    `stop_reason` uses the REAL D16 vocabulary: sim rows are always
    "horizon_complete" (matches run_gap_protocol_sim.py's sim_meta.json --
    sim never aborts, D16); real rows are "complete" unless synthetically
    marked "abort" with per-run probability `abort_frac` (default 0.0, no
    aborts -- preserves old behaviour byte for byte). Abort marking is
    per-(episode, repeat) RUN, not per term row, matching the real convention
    that every term row of one run shares that run's stop_reason.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    default_frac = {
        "gripper_box": 0.18, "approach_open": 0.05, "gripper_align": 0.17,
        "grasp": 0.08, "lift": 0.12, "box_target": 0.22, "hold_target": 0.08,
        "no_floor_collision": 0.03, "robot_target_qpos": 0.04,
        "action_rate": 0.03,
    }
    rows = []
    for cid, spec in config_specs.items():
        frac = spec.get("term_frac", default_frac)
        for ep in range(n_episodes):
            ep_sim = spec["sim_total"] + rng.normal(0, ep_noise)
            ep_real = spec["real_total"] + rng.normal(0, ep_noise)
            for sd in seeds:
                tot = ep_sim + rng.normal(0, seed_noise)
                for term, f in frac.items():
                    rows.append(dict(
                        config_id=cid, seed=sd, episode_id=ep, repeat=0,
                        cube_size=cube_size,
                        term=term, domain="sim", scaled_return_H=tot * f,
                        achieved_hz=np.nan, overrun_count=0,
                        stop_reason="horizon_complete", protocol_hash=protocol_hash,
                    ))
            for rp in repeats:
                tot = ep_real + rng.normal(0, repeat_noise)
                # Only draws from `rng` when abort_frac > 0, so the default
                # (no aborts) consumes exactly the same rng sequence as
                # before abort_frac existed -- byte-for-byte reproducible.
                run_stop_reason = (
                    "abort" if abort_frac > 0.0 and rng.random() < abort_frac
                    else "complete"
                )
                for term, f in frac.items():
                    rows.append(dict(
                        config_id=cid, seed=-1, episode_id=ep, repeat=rp,
                        cube_size=cube_size,
                        term=term, domain="real", scaled_return_H=tot * f,
                        achieved_hz=50.4, overrun_count=0,
                        stop_reason=run_stop_reason, protocol_hash=protocol_hash,
                    ))
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


# ===========================================================================
# --selftest
# ===========================================================================


def _assert_no_correlation_imports():
    """(a) No correlation function imported anywhere under evaluation/."""
    forbidden = re.compile(r"spearmanr|pearsonr|kendalltau|scipy\.stats|"
                           r"\bmmrv\b|srcc_|pearson_policy")
    hits = []
    for path in glob.glob(os.path.join(_EVAL_DIR, "*.py")):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                # Skip lines that only NAME the ban (docstrings/comments in this
                # very file). Flag only actual import/attribute usage.
                if forbidden.search(line) and (
                    "import" in line or "stats." in line or "(" in line
                ) and "forbidden" not in line and "re.compile" not in line \
                  and "banned" not in line:
                    hits.append(f"{os.path.basename(path)}:{i}: {line.strip()}")
    assert not hits, "correlation usage found under evaluation/:\n" + "\n".join(hits)


def _selftest():
    print("(a) no correlation function imported under evaluation/ ...", end=" ")
    _assert_no_correlation_imports()
    # Also assert this module never defines the banned statistics.
    import evaluation.gap_metrics as m
    for banned in ("mmrv", "srcc_policy", "pearson_policy", "srcc_state"):
        assert not hasattr(m, banned), f"gap_metrics defines banned {banned}"
    print("OK")

    print("(c) no normalisation path (no z-score / min-max helpers) ...", end=" ")
    src_lines = [
        l for l in
        open(os.path.join(_EVAL_DIR, "gap_metrics.py"), encoding="utf-8").readlines()
        if "for tok in" not in l  # this check's own token tuple, not a hit
    ]
    src = "".join(src_lines)
    for tok in ("zscore", "z_score", "StandardScaler", "MinMaxScaler", "normalize(", "normalise("):  # for tok in
        assert tok not in src, f"normalisation token present: {tok}"
    print("OK")

    print("(b) retention CI finite and lo <= point <= hi ...", end=" ")
    specs = {
        "L0": {"sim_total": 100.0, "real_total": 60.0},
        "L1": {"sim_total": 120.0, "real_total": 90.0},
        "L2": {"sim_total": 130.0, "real_total": 80.0},
    }
    df = make_synthetic_long(specs, n_episodes=10, ep_noise=8.0,
                             repeat_noise=3.0, seed_noise=2.0,
                             rng=np.random.default_rng(1))
    r = retention(df, "L1", n_boot=3000, seed=0)
    lo, hi = r["ci95"]
    assert np.isfinite(lo) and np.isfinite(hi), r
    assert lo <= r["retention"] <= hi, (lo, r["retention"], hi)
    print(f"OK  (retention={r['retention']:.3f}, CI=[{lo:.3f},{hi:.3f}])")

    # ---- (d) regret = 0 on agreement; known gap on disagreement ----
    print("(d) R = 0 on argmax agreement; known gap on disagreement ...", end=" ")
    # Agreement: L1 is best in BOTH sim and real. No noise -> exact point est.
    agree = {
        "A": {"sim_total": 100.0, "real_total": 50.0},
        "B": {"sim_total": 130.0, "real_total": 95.0},  # best sim AND best real
        "C": {"sim_total": 110.0, "real_total": 70.0},
    }
    df_a = make_synthetic_long(agree, n_episodes=6, ep_noise=0.0)
    res_a = sim_selection_regret(df_a, n_boot=200, seed=0)
    assert res_a["chat_sim"] == res_a["c_star"] == "B", res_a
    assert abs(res_a["regret"]) < 1e-9, res_a["regret"]
    # Disagreement: sim picks B, but real-best is A. Known gap = R_real[A]-R_real[B].
    disagree = {
        "A": {"sim_total": 110.0, "real_total": 95.0},   # real best
        "B": {"sim_total": 130.0, "real_total": 60.0},   # sim best, poor real
        "C": {"sim_total": 100.0, "real_total": 70.0},
    }
    df_d = make_synthetic_long(disagree, n_episodes=6, ep_noise=0.0)
    res_d = sim_selection_regret(df_d, n_boot=200, seed=0)
    assert res_d["chat_sim"] == "B" and res_d["c_star"] == "A", res_d
    known_gap = 95.0 - 60.0
    assert abs(res_d["regret"] - known_gap) < 1e-6, (res_d["regret"], known_gap)
    print(f"OK  (agree R=0; disagree R={res_d['regret']:.1f} == {known_gap})")

    # ---- (e) resampled-argmax CI strictly wider than fixed-chat_sim CI ----
    print("(e) resampled-argmax CI strictly wider than fixed-chat_sim ...", end=" ")
    # Two near-tied sim leaders (B slightly ahead) with real-return that
    # DISAGREES on which is better + per-episode noise -> chat_sim flips across
    # resamples, so the full regret distribution spans {~0, gap} while the
    # fixed-chat_sim one does not.
    tie = {
        "A": {"sim_total": 119.0, "real_total": 95.0},   # real best
        "B": {"sim_total": 120.0, "real_total": 65.0},   # sim best (barely)
        "C": {"sim_total": 90.0, "real_total": 60.0},
    }
    df_e = make_synthetic_long(tie, n_episodes=12, ep_noise=14.0,
                               seed_noise=1.0, rng=np.random.default_rng(7))
    res_e = sim_selection_regret(df_e, n_boot=6000, seed=3)
    w_full = res_e["ci_width_resampled"]
    w_fixed = res_e["ci_width_fixed"]
    assert w_full > w_fixed, (w_full, w_fixed, res_e["selection_freq"])
    print(f"OK  (width_full={w_full:.2f} > width_fixed={w_fixed:.2f}; "
          f"sel_freq={ {k: round(v,2) for k,v in res_e['selection_freq'].items()} })")

    # ---- (f) cube_size (D17): mixed-cube input refuses to average silently,
    # and pinning cube_size recovers the two DISTINCT per-cube numbers ----
    print("(f) cube_size grouping: mixed input raises, pinned recovers per-cube "
          "values ...", end=" ")
    cube_specs = {
        "L1": {"sim_total": 120.0, "real_total": 110.0},  # same sim for both
    }
    df_3cm = make_synthetic_long(cube_specs, n_episodes=10, ep_noise=0.0,
                                 cube_size="3cm", rng=np.random.default_rng(2))
    cube_specs_4cm = {
        "L1": {"sim_total": 120.0, "real_total": 40.0},  # OOD: real collapses
    }
    df_4cm = make_synthetic_long(cube_specs_4cm, n_episodes=10, ep_noise=0.0,
                                 cube_size="4cm", rng=np.random.default_rng(3))
    df_mixed = pd.concat([df_3cm, df_4cm], ignore_index=True)
    # No cube_size given, 2 distinct values present -> every threaded function
    # must refuse rather than silently pool 110 (3cm) with 40 (4cm).
    for fn, kwargs in (
        (retention, dict(config="L1")),
        (noise_floor, dict(config="L1")),
        (term_retention, dict(config="L1")),
        (sim_selection_regret, dict(configs=["L1"])),
    ):
        try:
            fn(df_mixed, **kwargs)
            raise AssertionError(f"{fn.__name__} did not raise on mixed cube_size")
        except ValueError:
            pass
    # Pinned: recovers the two distinct real_totals exactly (no averaging).
    r3 = retention(df_mixed, "L1", cube_size="3cm")
    r4 = retention(df_mixed, "L1", cube_size="4cm")
    assert abs(r3["R_real"] - 110.0) < 1e-6, r3
    assert abs(r4["R_real"] - 40.0) < 1e-6, r4
    assert r3["R_real"] != r4["R_real"]
    print(f"OK  (3cm R_real={r3['R_real']:.1f}, 4cm R_real={r4['R_real']:.1f}, "
          f"mixed input correctly refused)")

    # ---- (g) noise_floor (D9, corrected): paired-gap spread != raw-return
    # spread when a genuine per-repeat sim mirror is present -- this is the
    # regression guard against reverting to the old raw-return-std floor ----
    print("(g) noise_floor: paired-gap spread != raw real-return spread "
          "(genuine per-repeat sim mirror) ...", end=" ")
    # One config, one episode, 3 repeats. A shared per-repeat "placement
    # effect" [0, +8, -8] is present in BOTH real and sim (mirrors the SAME
    # measured placement each repeat, D9's mechanics); real additionally
    # carries small irreducible noise [+1, -1, +2] sim never sees (sim is
    # deterministic). Pairing must cancel the (dominant) placement effect and
    # leave only that small irreducible noise.
    placement = [0.0, 8.0, -8.0]
    irreducible_noise = [1.0, -1.0, 2.0]
    base_sim, base_real = 80.0, 75.0
    real_vals = [base_real + p + n for p, n in zip(placement, irreducible_noise)]
    sim_vals = [base_sim + p for p in placement]  # deterministic, no noise
    rows = []
    for rp, (rv, sv) in enumerate(zip(real_vals, sim_vals), start=1):
        rows.append(dict(
            config_id="Lpaired", seed=-1, episode_id=0, repeat=rp,
            cube_size="3cm", term="gripper_box", domain="real",
            scaled_return_H=rv, achieved_hz=50.4, overrun_count=0,
            stop_reason="complete", protocol_hash="paired_test",
        ))
        rows.append(dict(
            config_id="Lpaired", seed=0, episode_id=0, repeat=rp,
            cube_size="3cm", term="gripper_box", domain="sim",
            scaled_return_H=sv, achieved_hz=np.nan, overrun_count=0,
            stop_reason="horizon_complete", protocol_hash="paired_test",
        ))
    df_paired = pd.DataFrame(rows, columns=LONG_COLUMNS)

    nf_paired = noise_floor(df_paired, "Lpaired")
    raw_std = float(np.std(real_vals, ddof=1))
    paired_gaps = [rv - sv for rv, sv in zip(real_vals, sim_vals)]
    expected_paired_std = float(np.std(paired_gaps, ddof=1))
    assert abs(nf_paired["noise_floor"] - expected_paired_std) < 1e-9, (
        nf_paired["noise_floor"], expected_paired_std)
    # The regression guard: if someone reverts noise_floor to std(raw real
    # returns), this assertion fails -- the placement effect (std~6.5) must
    # NOT survive pairing (std~1.5 here).
    assert nf_paired["noise_floor"] < 0.4 * raw_std, (
        f"paired floor {nf_paired['noise_floor']} is not clearly smaller than "
        f"the raw-return std {raw_std} -- looks like the OLD (pre-D9-fix) "
        f"raw-return computation, not the paired-gap one")
    print(f"OK  (paired floor={nf_paired['noise_floor']:.3f} << "
          f"raw real-return std={raw_std:.3f}; placement effect cancelled)")

    # ---- noise_floor fallback: no per-repeat sim mirror (only a repeat=0
    # seed-ensemble row) -> paired floor degrades EXACTLY to the raw-return
    # std (subtracting a constant never changes a spread) -- documents the
    # honest degradation path make_synthetic_long's general demo tables hit ----
    print("    noise_floor fallback (no per-repeat sim mirror) == raw std ...",
          end=" ")
    df_no_mirror = make_synthetic_long(
        {"L1": {"sim_total": 100.0, "real_total": 90.0}},
        n_episodes=3, ep_noise=5.0, repeat_noise=4.0,
        rng=np.random.default_rng(11),
    )
    nf_fallback = noise_floor(df_no_mirror, "L1")
    raw_per_ep = (
        _run_totals(df_no_mirror[(df_no_mirror["config_id"] == "L1")
                                 & (df_no_mirror["domain"] == "real")])
        .groupby("episode_id")["total"].std().fillna(0.0).mean()
    )
    assert abs(nf_fallback["noise_floor"] - float(raw_per_ep)) < 1e-9, (
        nf_fallback["noise_floor"], raw_per_ep)
    print(f"OK  (fallback floor={nf_fallback['noise_floor']:.3f} == "
          f"raw-return floor={raw_per_ep:.3f})")

    # ---- (h) abort_rate (D16/D17) ----
    print("(h) abort_rate: all-abort, no-abort, and an exact partial count ...",
          end=" ")
    df_all_abort = make_synthetic_long(
        {"L1": {"sim_total": 100.0, "real_total": 80.0}},
        n_episodes=4, repeats=(1, 2, 3), abort_frac=1.0, cube_size="4cm",
        rng=np.random.default_rng(5),
    )
    ar_all = abort_rate(df_all_abort, "L1", cube_size="4cm")
    assert ar_all["n_runs"] == 12, ar_all
    assert ar_all["n_aborted"] == 12, ar_all
    assert ar_all["abort_rate"] == 1.0, ar_all

    df_no_abort = make_synthetic_long(
        {"L1": {"sim_total": 100.0, "real_total": 80.0}},
        n_episodes=4, repeats=(1, 2, 3), abort_frac=0.0, cube_size="3cm",
        rng=np.random.default_rng(6),
    )
    ar_none = abort_rate(df_no_abort, "L1", cube_size="3cm")
    assert ar_none["n_runs"] == 12, ar_none
    assert ar_none["n_aborted"] == 0, ar_none
    assert ar_none["abort_rate"] == 0.0, ar_none

    # Exact partial count: 6 runs (2 episodes x 3 repeats), exactly 2 aborted.
    partial_rows = []
    abort_flags = {(0, 1): True, (0, 2): False, (0, 3): False,
                   (1, 1): True, (1, 2): False, (1, 3): False}
    for (ep, rp), is_abort in abort_flags.items():
        partial_rows.append(dict(
            config_id="L2", seed=-1, episode_id=ep, repeat=rp,
            cube_size="4cm", term="gripper_box", domain="real",
            scaled_return_H=50.0, achieved_hz=50.4, overrun_count=0,
            stop_reason="abort" if is_abort else "complete",
            protocol_hash="abort_test",
        ))
    df_partial = pd.DataFrame(partial_rows, columns=LONG_COLUMNS)
    ar_partial = abort_rate(df_partial, "L2", cube_size="4cm")
    assert ar_partial["n_runs"] == 6, ar_partial
    assert ar_partial["n_aborted"] == 2, ar_partial
    assert abs(ar_partial["abort_rate"] - 2 / 6) < 1e-9, ar_partial
    print(f"OK  (all-abort=1.0, no-abort=0.0, partial={ar_partial['abort_rate']:.3f} "
          f"== 2/6)")

    # ---- gate_against_noise_floor sanity ----
    print("gate_against_noise_floor ...", end=" ")
    assert gate_against_noise_floor(10.0, (6.0, 14.0), floor=5.0) == "reportable"
    assert gate_against_noise_floor(3.0, (-1.0, 7.0), floor=5.0) == "within_noise_floor"
    assert gate_against_noise_floor(-10.0, (-14.0, -6.0), 5.0) == "reportable"
    print("OK")

    # ---- D22 place_error ----
    print("place_error ...", end=" ")
    d0, in0 = place_error(DROP_SQUARE_CENTER)
    assert d0 == 0.0 and in0, (d0, in0)
    d1, in1 = place_error((0.212 + 0.075, 0.212))       # exactly on the edge
    assert abs(d1 - 0.075) < 1e-9 and in1, (d1, in1)
    d2, in2 = place_error((0.212 + 0.10, 0.212))        # outside in x
    assert abs(d2 - 0.10) < 1e-9 and not in2, (d2, in2)
    d3, in3 = place_error((0.212 + 0.03, 0.212 - 0.04))  # 3-4-5, inside
    assert abs(d3 - 0.05) < 1e-9 and in3, (d3, in3)
    # Corner case the axis-aligned test must get RIGHT and a radial test would
    # get wrong: dx=dy=0.07 is inside the square but 0.099 m from the centre.
    d4, in4 = place_error((0.212 + 0.07, 0.212 + 0.07))
    assert in4 and d4 > 0.075, (d4, in4)
    print("OK")

    print("\nSELFTEST OK (no MJX, no robot, no correlation, no normalisation).")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cube_size", default=None, choices=[None, "3cm", "4cm"],
                    help="D17: pin the cube-size grouping key. Required if the "
                         "CSV mixes 3cm and 4cm rows -- see _filter_cube_size.")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    if args.selftest or not args.csv:
        _selftest()
        return

    df = load_long(args.csv)
    configs = sorted(df["config_id"].unique())
    print(f"Loaded {len(df)} rows, configs={configs}")
    if args.config:
        r = retention(df, args.config, cube_size=args.cube_size, n_boot=args.n_boot)
        nf = noise_floor(df, args.config, cube_size=args.cube_size)
        print(f"\nretention({args.config}, cube_size={args.cube_size}) = "
              f"{r['retention']:.3f} CI{tuple(round(x,3) for x in r['ci95'])}  "
              f"[R_sim={r['R_sim']:.1f} R_real={r['R_real']:.1f}]")
        print(f"noise_floor({args.config}, cube_size={args.cube_size}) = "
              f"{nf['noise_floor']:.3f} (return units)")
        print(f"gate: {gate_against_noise_floor(r['R_real']-r['R_sim'], r['ci95'], nf['noise_floor'])}")
        print("\nterm_retention:")
        print(term_retention(df, args.config, cube_size=args.cube_size)["per_stage"])
    if len(configs) >= 2:
        res = sim_selection_regret(df, cube_size=args.cube_size, n_boot=args.n_boot)
        print(f"\nsim_selection_regret(cube_size={args.cube_size}): R={res['regret']:.2f} "
              f"CI{tuple(round(x,2) for x in res['ci95_resampled_argmax'])}  "
              f"chat_sim={res['chat_sim']} c*={res['c_star']}")
        print(res["table"])


if __name__ == "__main__":
    _cli()
