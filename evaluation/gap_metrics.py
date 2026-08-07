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
    python evaluation/gap_metrics.py --build   # merge raw run folders -> gap_metrics.csv
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
# proxy signal, or gated behind a latch that a proxy signal drives). Flagged
# for the diagnostic caption in F2; not treated differently numerically.
#
# CORRECTED 2026-07-30 -- the original split ({"grasp",
# "no_floor_collision"}) was wrong in both directions, found while tracing
# RewardReplayer.step() in ur3_reward_replay.py against what it actually
# reads from the real logs:
#   * `grasp` = finger_closed * reached * alignment -- reached and alignment
#     are both pure geometry (gripper_box_dist, box axes from mocap), and
#     grasp_contact only ever appears in the SEPARATE `grasped` latch below.
#     `grasp` itself never reads grasp_contact -- it is NOT a proxy term.
#   * `no_floor_collision` genuinely has no real sensor (defaults to 1 on
#     real) -- correctly a proxy.
#   * The real proxy chain is: grasp_contact (Hand-E flag, <=10 Hz) ->
#     `grasped` latch -> `lifted` latch, and `lifted` then GATES/MULTIPLIES
#     three more terms: `box_target` (x lifted), `hold_target` (in_hold
#     requires lifted), and `robot_target_qpos` (x (1-lifted)). All three
#     inherit the proxy even though they read geometry too -- a false
#     `grasped` (Hand-E never detects contact) forces all three to whatever
#     value lifted=False implies, regardless of the true box/target
#     geometry. `lift` itself is gated on `reached` only (pure geometry), so
#     it stays proxy-free.
PROXY_TERMS = {"no_floor_collision", "box_target", "hold_target",
               "robot_target_qpos"}
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


def _filter_terms(df: pd.DataFrame, terms, fn_name: str) -> pd.DataFrame:
    """Restrict to a specific set of `term` rows before summing.

    `terms=None` (the default everywhere below) keeps every row -- i.e. the
    FULL reward, proxy terms included. This is deliberately opt-in rather
    than defaulted to `EXACT_PARITY_TERMS`, because term_retention()'s
    per-term/per-stage breakdown (F2) needs every term present to report on.
    Callers computing a HEADLINE cross-domain metric -- retention(),
    noise_floor(), sim_selection_regret() -- must pass
    `terms=EXACT_PARITY_TERMS` explicitly; nothing upstream filters for them.
    (Found 2026-07-30: EXACT_PARITY_TERMS was defined but never actually
    applied anywhere in this call chain, so every headline number silently
    included the proxy-affected terms despite 3 Method.md's "the headline
    result is defined on this exact-parity subset." This parameter, plus the
    explicit `terms=` passed at every headline call site, is the fix.)
    """
    if terms is None or "term" not in df.columns:
        return df
    return df[df["term"].isin(terms)]


def per_episode_returns(
    df: pd.DataFrame, config: str, episodes=None, cube_size=None, terms=None
) -> pd.DataFrame:
    """Paired per-episode returns for one config (+ one cube_size, D17).

    Returns a DataFrame indexed by episode_id with columns ['sim','real'] --
    each the mean total return over that episode's seeds (sim) / repeats (real).
    Only episodes present in BOTH domains are kept (pairing requirement).
    `terms`: optional iterable of term names to restrict to before summing
    (e.g. EXACT_PARITY_TERMS) -- see `_filter_terms`. Default None = full
    reward, all terms.
    """
    sub = _filter_cube_size(df[df["config_id"] == config], cube_size,
                             "per_episode_returns")
    sub = _filter_terms(sub, terms, "per_episode_returns")
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
    if s.size == 0:
        # No finite bootstrap samples (e.g. 0 paired episodes fed in --
        # np.percentile raises IndexError on an empty array otherwise).
        # NaN is the honest "undefined" answer, not a crash.
        return float("nan"), float("nan")
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


# ===========================================================================
# Metrics
# ===========================================================================


def retention(df, config, subset="exact", episodes=None, cube_size=None,
              n_boot=10000, seed=0, terms=None):
    """Real/sim return retention for one config + paired-bootstrap 95% CI.

    ratio = mean_episode(real) / mean_episode(sim), on RAW returns (no
    normalisation -- see module docstring). CI resamples episodes.
    `subset` is a label recorded in the result (e.g. "exact" = exact-init
    matched episodes); pass `episodes` to restrict to a subset's episode_ids.
    `cube_size` (D17): pin to "3cm"/"4cm" when `df` mixes both -- see
    `_filter_cube_size`.
    `terms`: optional iterable of term names to sum over instead of the
    full reward (e.g. `EXACT_PARITY_TERMS`) -- see `_filter_terms`. THE
    HEADLINE RETENTION NUMBER MUST PASS `terms=EXACT_PARITY_TERMS`: 3
    Method.md's exact-parity paragraph promises the headline is computed on
    that subset, and nothing does so unless the caller asks. Default None
    (full reward) is kept only for callers that deliberately want it.
    """
    piv = per_episode_returns(df, config, episodes=episodes,
                               cube_size=cube_size, terms=terms)
    real = piv["real"].to_numpy()
    sim = piv["sim"].to_numpy()
    n = len(piv)
    if n == 0:
        # No episode has BOTH a sim and a real row for this (config,
        # cube_size) -- e.g. the sim mirror (Commit 7) was never rolled out
        # for this cube on this config. NaN everywhere is the honest
        # "undefined, not measured" answer; the alternative (falling through
        # to np.mean([])) would still be NaN for R_sim/R_real but crash later
        # in the bootstrap (rng.integers(0, 0, ...) -> all-NaN ratios ->
        # np.percentile on an empty finite-value array raises IndexError).
        return {
            "config_id": config, "subset": subset, "cube_size": cube_size,
            "n_episodes": 0,
            "R_sim": float("nan"), "R_real": float("nan"),
            "retention": float("nan"), "ci95": (float("nan"), float("nan")),
        }
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


def noise_floor(df, config, episodes=None, cube_size=None, terms=None):
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
    `_filter_cube_size`. `terms`: optional iterable of term names (e.g.
    `EXACT_PARITY_TERMS`) to restrict to before summing -- see
    `_filter_terms`. Pass the SAME `terms` used for the paired `retention()`
    call, so the floor is measured on the same reward definition the gap is
    checked against.
    """
    real = _filter_terms(_filter_cube_size(
        df[(df["config_id"] == config) & (df["domain"] == "real")],
        cube_size, "noise_floor",
    ), terms, "noise_floor")
    sim = _filter_terms(_filter_cube_size(
        df[(df["config_id"] == config) & (df["domain"] == "sim")],
        cube_size, "noise_floor",
    ), terms, "noise_floor")
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


def _config_returns_matrix(df, configs, terms=None):
    """(episode_ids, R_sim[config x ep], R_real[config x ep]) on a shared episode set."""
    pivs = {c: per_episode_returns(df, c, terms=terms) for c in configs}
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


def sim_selection_regret(df, configs=None, cube_size=None, n_boot=10000,
                          seed=0, terms=None):
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
    `terms`: optional iterable of term names (e.g. `EXACT_PARITY_TERMS`) --
    3 Method.md defines both R_sim/R_real here as "mean exact-parity return",
    so the report-facing caller must pass `terms=EXACT_PARITY_TERMS`; see
    `_filter_terms`.
    """
    df = _filter_cube_size(df, cube_size, "sim_selection_regret")
    if configs is None:
        configs = sorted(df["config_id"].unique())
    configs = list(configs)
    eps, sim, real = _config_returns_matrix(df, configs, terms=terms)

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
# Build gap_metrics.csv from the raw per-run folders (the missing half of
# Commit 8 -- run_gap_protocol_sim.py's own docstring calls this "a later
# merge into evaluation/gap_metrics.csv [that] can walk both trees
# symmetrically"). PLAIN MuJoCo FK only (via ur3_reward_replay) -- no MJX, no
# robot, safe to run locally per CLAUDE.md.
#
# Real leaf dirs (robots/UR3e/real_robot_results/gap_protocol/<config>/
# <protocol_id>/<leaf>/) carry RAW geometry only (ur3_pick_meta.json,
# measured_init.json, ur3_pick_states.csv) -- no reward terms are logged on
# the real robot. Sim leaf dirs (robots/UR3e/sim_results/gap_protocol/
# <config>/<protocol_id>/<leaf>/, written by run_gap_protocol_sim.py) already
# carry per-step scaled reward terms in sim_states.csv (sim_meta.json for
# bookkeeping). So:
#   * sim  rows: sum sim_states.csv's existing TERM_ORDER columns over steps.
#   * real rows: replay ur3_pick_states.csv through
#     ur3_reward_replay.replay_dataframe() (line-for-line port of
#     ur3_pick.py's _get_reward, plain MuJoCo forward kinematics) to
#     reconstruct the same per-step term columns, THEN sum over steps.
# Both are then episode-summed scaled returns per term, matching the module
# docstring's definition of scaled_return_H.
#
# Folder names are NOT parsed for episode_id/repeat/cube_size/config_id --
# real leaf names are inconsistent (some carry a condition-letter prefix like
# "A3_ep2_rep3_3cm", some don't, e.g. "ep1_rep1_3cm") and are easy to get
# subtly wrong. Every field instead comes straight out of the run's own
# meta json (ur3_pick_meta.json / sim_meta.json), which is unambiguous by
# construction.
#
# Aborted real runs (stop_reason == "abort") ARE included: they still write
# a meta json + a (truncated) states.csv, abort_rate() (D16/D17) needs their
# stop_reason present in the long table, and per-episode pairing (D16)
# already excludes them downstream where relevant -- silently dropping them
# here would just make the abort rate itself unmeasurable.
# ===========================================================================


# CAMPAIGN FILTER (added 2026-07-30 after a contamination bug was found).
#
# robots/UR3e/real_robot_results/gap_protocol/ contains TWO campaigns:
#   * 2026-07-22, git 1c624db3: policies `L*_s?_20260717_*` (novelocitymodel
#     branch, 26-D obs, stale table geometry), leaf dirs named `ep0_rep1_3cm`
#     (no condition prefix).
#   * 2026-07-29, git 15426ef3: the D23 velocity ladder, policies
#     `L*_vel_s1_20260729_*` (33-D obs, real 95mm table), leaf dirs named
#     `A1_ep0_rep1_3cm` (condition-prefixed).
# gap_protocol_policy_map.json marks the first set `_superseded_...`.
#
# Mixing them is not merely noisy, it is CORRUPTING: both campaigns reuse
# episode_id 0/1/2 AND seed 1, so a legacy `ep0_rep1_3cm` and a D23
# `A1_ep0_rep1_3cm` share an identical (config_id, domain, seed, episode_id,
# repeat, cube_size) key. `_run_totals` groups on exactly that key and SUMS,
# so the two runs silently merge into one fabricated run with ~double the
# return. Measured on the unfiltered build: 290 colliding term-rows (29 runs).
#
# A sim/real pair is only meaningful if BOTH SIDES RAN THE SAME POLICY
# CHECKPOINT, so the filter below is by policy_run_id against the map, plus
# (sim side) a check that the mirrored real folder is itself a D23 folder --
# 12 sim runs use a D23 policy but were initialized from a LEGACY real run's
# measured_init.json, i.e. they mirror a run the robot executed with a
# different policy entirely.
POLICY_MAP_PATH = os.path.join(
    os.path.dirname(_EVAL_DIR), "robots", "UR3e", "gap_protocol_policy_map.json")


def load_policy_map(path: str = None):
    """{config_id: policy_run_id} from gap_protocol_policy_map.json.
    Underscore-prefixed keys (`_comment`, `_superseded_*`) are bookkeeping.
    Returns {} if absent -- callers must then SKIP filtering loudly rather
    than silently accepting a mixed-campaign table.
    """
    import json
    path = path or POLICY_MAP_PATH
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, str)}


def _is_legacy_leafname(name):
    """D23 leaf dirs are condition-prefixed (`A1_ep0_rep1_3cm`); the
    superseded 2026-07-22 campaign's are not (`ep0_rep1_3cm`)."""
    return os.path.basename(str(name).rstrip("/")).startswith("ep")


def _extract_seed(policy_run_id):
    """Best-effort seed extraction from a policy_run_id like
    'L1_pos_vel_s1_20260729_112044_2201' -> 1. Returns None if absent --
    callers must tolerate a missing seed (LONG_COLUMNS' 'seed' is bookkeeping
    only; no downstream function here groups by it).
    """
    if not policy_run_id:
        return None
    m = re.search(r"_s(\d+)_", policy_run_id)
    return int(m.group(1)) if m else None


def _iter_run_dirs(root, protocol_id):
    """Yield (config_id, leaf_dir_path) for every run folder under
    root/<config_id>/<protocol_id>/<leaf>/. Silently skips configs that don't
    have a <protocol_id> subfolder (e.g. unrelated configs living next to
    gap_protocol/ in the same real_robot_results tree).
    """
    if not os.path.isdir(root):
        return
    for config_id in sorted(os.listdir(root)):
        proto_dir = os.path.join(root, config_id, protocol_id)
        if not os.path.isdir(proto_dir):
            continue
        for leaf in sorted(os.listdir(proto_dir)):
            leaf_dir = os.path.join(proto_dir, leaf)
            if os.path.isdir(leaf_dir):
                yield config_id, leaf_dir


def _load_json(path):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rows_from_sim_leaf(leaf_dir, policy_map=None, skips=None):
    """One row per TERM_ORDER term for a single sim_results leaf dir."""
    meta_path = os.path.join(leaf_dir, "sim_meta.json")
    states_path = os.path.join(leaf_dir, "sim_states.csv")
    if not (os.path.isfile(meta_path) and os.path.isfile(states_path)):
        print(f"[build] skip (missing sim_meta.json/sim_states.csv): {leaf_dir}")
        return []
    meta = _load_json(meta_path)
    if policy_map:
        want = policy_map.get(meta["config_id"])
        got = meta.get("policy_run_id")
        if want and got != want:
            if skips is not None:
                skips["sim_wrong_policy"].append((leaf_dir, got))
            return []
        # A D23-policy rollout initialized from a LEGACY real run mirrors a
        # run the robot executed under a DIFFERENT policy -- not a valid pair.
        if _is_legacy_leafname(meta.get("measured_init_path") and
                                os.path.dirname(meta["measured_init_path"])):
            if skips is not None:
                skips["sim_mirrors_legacy"].append((leaf_dir, got))
            return []
    try:
        df = pd.read_csv(states_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if len(df) == 0:
        # Sim never aborts (run_gap_protocol_sim.py's stop_reason is always
        # "horizon_complete", D16) -- an empty sim_states.csv here would be
        # an actual pipeline bug, not an expected outcome. Still don't crash
        # the whole build over one bad folder: emit zero-return rows and
        # print loudly so it gets noticed.
        print(f"[build] WARNING: empty sim_states.csv (unexpected -- sim "
              f"never aborts), emitting zero-return rows: {leaf_dir}")
    seed = _extract_seed(meta.get("policy_run_id"))
    rows = []
    for term in TERM_ORDER:
        total = float(df[term].sum()) if term in df.columns else 0.0
        rows.append(dict(
            config_id=meta["config_id"], seed=seed,
            episode_id=meta["episode_id"], repeat=meta["repeat"],
            cube_size=meta["cube_size"], term=term, domain="sim",
            scaled_return_H=total,
            policy_run_id=meta.get("policy_run_id"),
            achieved_hz=meta.get("achieved_hz"),
            overrun_count=meta.get("overrun_count"),
            stop_reason=meta.get("stop_reason"),
            protocol_hash=meta.get("poses_sha256"),
        ))
    return rows


def _rows_from_real_leaf(leaf_dir, replay_dataframe, xml_path,
                          policy_map=None, skips=None):
    """One row per TERM_ORDER term for a single real_robot_results leaf dir.

    `replay_dataframe` is passed in (not imported at module top) so that
    gap_metrics.py's own import graph stays numpy/pandas-only unless --build
    is actually used -- ur3_reward_replay additionally needs matplotlib and
    mujoco (plain FK, not MJX; still heavier than this module normally
    needs).
    """
    meta_path = os.path.join(leaf_dir, "ur3_pick_meta.json")
    states_path = os.path.join(leaf_dir, "ur3_pick_states.csv")
    if not (os.path.isfile(meta_path) and os.path.isfile(states_path)):
        print(f"[build] skip (missing ur3_pick_meta.json/ur3_pick_states.csv): "
              f"{leaf_dir}")
        return []
    meta = _load_json(meta_path)
    if policy_map:
        want = policy_map.get(meta["config_id"])
        got = meta.get("policy_run_id")
        if want and got != want:
            if skips is not None:
                skips["real_wrong_policy"].append((leaf_dir, got))
            return []
    # 0-step aborts (stop_step: 0, e.g. the hang watchdog fires before a
    # single control loop iteration is logged) leave ur3_pick_states.csv
    # empty or containing not even a header row -- pd.read_csv raises
    # EmptyDataError in that case rather than returning a 0-row frame. These
    # are NOT a data error to skip: they are a genuine (config, cube_size)
    # abort that abort_rate() (D16/D17) needs to count, so still emit rows
    # here, just with scaled_return_H=0.0 for every term (zero steps ->
    # zero accrued reward) and stop_reason carried through from the meta.
    try:
        df = pd.read_csv(states_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if len(df) == 0:
        print(f"[build] 0-step run (stop_reason={meta.get('stop_reason')}), "
              f"emitting zero-return rows: {leaf_dir}")
        reward_df = pd.DataFrame(columns=TERM_ORDER)
    else:
        # target=None -> replay_dataframe reads per-row target_x/y/z from the
        # log itself (present in ur3_pick_states.csv), so this can never
        # disagree with what the run was actually scored against.
        #
        # cfg / box_half: score each run with the reward THAT policy trained
        # under, not with ur3_pick.default_config(). Before the v3 reward fix
        # every deployed policy shared the defaults so this was a no-op, but a
        # v3 policy trains with align_mode="axis_aware", a raised
        # grasp_align_thresh, and reduced action_rate / robot_target_qpos --
        # replaying it under v2 reward would change WHICH steps latch `grasped`
        # and silently corrupt the grasp/lift/transport stages of every
        # retention number. config_for_policy falls back to the defaults (with a
        # warning) when the policy's metadata.json is absent, so the archived
        # D23 campaign replays exactly as before.
        from ur3_reward_replay import (  # local import, see docstring above
            CUBE_HALF_EXTENTS, config_for_policy,
        )
        reward_df = replay_dataframe(
            df, xml_path=xml_path, target=None, contact_col="grasped",
            cfg=config_for_policy(meta.get("policy_run_id")),
            box_half=CUBE_HALF_EXTENTS.get(meta.get("cube_size", "3cm")),
        )
    seed = _extract_seed(meta.get("policy_run_id"))
    rows = []
    for term in TERM_ORDER:
        total = float(reward_df[term].sum()) if term in reward_df.columns \
            else 0.0
        rows.append(dict(
            config_id=meta["config_id"], seed=seed,
            episode_id=meta["episode_id"], repeat=meta["repeat"],
            cube_size=meta["cube_size"], term=term, domain="real",
            scaled_return_H=total,
            policy_run_id=meta.get("policy_run_id"),
            achieved_hz=meta.get("achieved_hz"),
            overrun_count=meta.get("overrun_count"),
            stop_reason=meta.get("stop_reason"),
            protocol_hash=meta.get("poses_sha256"),
        ))
    return rows


def build_long_csv(
    real_root=os.path.join("robots", "UR3e", "real_robot_results", "gap_protocol"),
    sim_root=os.path.join("robots", "UR3e", "sim_results", "gap_protocol"),
    protocol_id="gap_v1",
    xml_path=None,
    out_csv=None,
):
    """Merge the raw real + sim run folders into one long-format DataFrame
    matching LONG_COLUMNS, and (if out_csv is given) write it to disk.

    `real_root`/`sim_root` are relative to the repo root by default (run this
    from the repo root, e.g. `python evaluation/gap_metrics.py --build`).
    Returns the merged DataFrame either way, so callers (e.g. a notebook)
    can also just inspect it without writing a file.
    """
    from ur3_reward_replay import replay_dataframe  # local import, see above

    policy_map = load_policy_map()
    if policy_map:
        print(f"[build] campaign filter ON -- keeping only runs whose "
              f"policy_run_id matches gap_protocol_policy_map.json:")
        for k, v in sorted(policy_map.items()):
            print(f"         {k}: {v}")
    else:
        print(f"[build] WARNING: no policy map at {POLICY_MAP_PATH} -- campaign "
              f"filter OFF. The output MAY MIX the superseded 2026-07-22 "
              f"campaign with the D23 one; see the campaign-filter note above.")
    skips = {"sim_wrong_policy": [], "sim_mirrors_legacy": [],
             "real_wrong_policy": [], "real_superseded_rerun": []}

    rows = []
    n_sim_leaves = n_real_leaves = 0
    for config_id, leaf_dir in _iter_run_dirs(sim_root, protocol_id):
        n_sim_leaves += 1
        rows.extend(_rows_from_sim_leaf(leaf_dir, policy_map, skips))

    # Within-D23 re-runs: L0_none episodes 3-5 were executed twice on
    # 2026-07-29 -- once before the condition-prefix naming convention
    # (`ep3_rep1_3cm`, 17:04-17:10) and again after it (`B1_ep3_rep1_3cm`,
    # 17:19-17:24). Same policy, same episode/repeat/cube, so they collide on
    # the run-identity key exactly like the cross-campaign case. Keep the
    # CONDITION-PREFIXED run: it is the canonical D23 board labelling every
    # other config uses, it is the later execution, and the earlier batch has
    # aborts where the re-run completed all 400 steps.
    real_leaves = list(_iter_run_dirs(real_root, protocol_id))
    _by_key = {}
    for _cfg, leaf_dir in real_leaves:
        mp = os.path.join(leaf_dir, "ur3_pick_meta.json")
        if not os.path.isfile(mp):
            continue
        m = _load_json(mp)
        if policy_map and policy_map.get(m["config_id"]) not in (
                None, m.get("policy_run_id")):
            continue  # dropped by the campaign filter anyway
        _by_key.setdefault(
            (m["config_id"], m["episode_id"], m["repeat"], m.get("cube_size")),
            []).append((leaf_dir, m.get("timestamp") or ""))
    superseded = set()
    for key, cands in _by_key.items():
        if len(cands) < 2:
            continue
        # prefer condition-prefixed, then latest timestamp
        ranked = sorted(cands, key=lambda c: (
            _is_legacy_leafname(c[0]), c[1]), reverse=False)
        keep = ranked[0]
        for leaf_dir, ts in ranked[1:]:
            superseded.add(leaf_dir)
            skips["real_superseded_rerun"].append((leaf_dir, f"kept {os.path.basename(keep[0])}"))

    for config_id, leaf_dir in real_leaves:
        n_real_leaves += 1
        if leaf_dir in superseded:
            continue
        rows.extend(_rows_from_real_leaf(leaf_dir, replay_dataframe, xml_path,
                                          policy_map, skips))

    for reason, dropped in skips.items():
        if dropped:
            pols = sorted({p for _, p in dropped})
            print(f"[build] DROPPED {len(dropped)} runs ({reason}); "
                  f"policy_run_id(s): {pols}")

    df = pd.DataFrame(rows, columns=LONG_COLUMNS + ["policy_run_id"])

    # Hard guard: after filtering, no two runs may share a run-identity key.
    # If they do, _run_totals would SUM them into one fabricated run (this is
    # exactly the 2026-07-22/D23 collision the campaign filter exists to
    # prevent) -- fail loudly rather than emit a corrupted table.
    if len(df):
        kcols = ["config_id", "domain", "seed", "episode_id", "repeat",
                 "cube_size", "term"]
        n_dup = int((df.groupby(kcols, dropna=False).size() > 1).sum())
        if n_dup:
            raise ValueError(
                f"{n_dup} run-identity keys are duplicated after filtering -- "
                f"_run_totals would sum them into fabricated runs. Inspect the "
                f"policy_run_id column; this means two distinct runs still "
                f"share (config, domain, seed, episode, repeat, cube, term)."
            )
    n_runs = len(rows) // max(len(TERM_ORDER), 1)
    print(f"[build] {n_sim_leaves} sim leaf dirs, {n_real_leaves} real leaf "
          f"dirs scanned -> {n_runs} runs, {len(df)} rows "
          f"({df['config_id'].nunique()} configs, "
          f"domains={sorted(df['domain'].unique().tolist()) if len(df) else []})")
    if len(df):
        print(df.groupby(["config_id", "domain"]).size()
              .div(len(TERM_ORDER)).rename("n_runs").astype(int))

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[build] wrote {out_csv}")
    return df


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
    ap.add_argument("--build", action="store_true",
                    help="Merge raw real_robot_results/gap_protocol + "
                         "sim_results/gap_protocol run folders into a long-"
                         "format CSV (default: evaluation/gap_metrics.csv). "
                         "Plain MuJoCo FK replay only, no MJX -- safe to run "
                         "locally.")
    ap.add_argument("--real_root", default=os.path.join(
        "robots", "UR3e", "real_robot_results", "gap_protocol"))
    ap.add_argument("--sim_root", default=os.path.join(
        "robots", "UR3e", "sim_results", "gap_protocol"))
    ap.add_argument("--protocol_id", default="gap_v1")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cube_size", default=None, choices=[None, "3cm", "4cm"],
                    help="D17: pin the cube-size grouping key. Required if the "
                         "CSV mixes 3cm and 4cm rows -- see _filter_cube_size.")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    if args.build:
        out_csv = args.csv or os.path.join(_EVAL_DIR, "gap_metrics.csv")
        build_long_csv(real_root=args.real_root, sim_root=args.sim_root,
                        protocol_id=args.protocol_id, out_csv=out_csv)
        return

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
