"""Nine sim-to-real gap figures over evaluation/gap_metrics.csv (+ a few
sibling artifacts: dynamics-replay error, timing logs, LOO sim returns).

Style mirrors the repo's existing plotting conventions (evaluation/
ur3_reward_replay.py's save_reward_terms_plot, evaluation/
compare_reward_train_sim_real.py's _MK_FORMATTER, evaluation/
run_robustness_eval*.py's dpi=300 savefig): Agg backend, tab10/tab20
categorical colors, png+pdf at dpi=300, `_MK_FORMATTER` for large step counts,
`bbox_inches="tight"`, bold titles, `alpha=0.3` grids.

No metric math lives here -- every number plotted is read from
evaluation/gap_metrics.py's functions (retention, noise_floor, term_retention,
sim_selection_regret, abort_rate, gate_against_noise_floor) or from a sibling
module (evaluation/ur3_dynamics_replay.time_to_divergence). This module only
arranges axes.

D17 cube_size: F1's headline rendering (`f1_dose_response_cube_split`)
overlays the 3cm in-distribution result (solid) and the 4cm OOD probe
(dashed) on one figure; F9 (`f9_abort_rate_bar`) is a new grouped bar chart
of abort rate by (policy, cube_size). Every 4cm-cube panel/figure/caption
carries an explicit OOD label (`OOD_CAPTION_4CM`) -- 4cm is never averaged
with 3cm anywhere (gap_metrics.py's `_filter_cube_size` guard already
enforces this at the data layer; the figures additionally make it visible).

--demo fabricates a plausible synthetic gap_metrics.csv (reusing
gap_metrics.make_synthetic_long -- the SAME fabrication gap_metrics.py's own
--selftest uses, not a second one) plus small synthetic inputs for the 3
figures gap_metrics.csv alone cannot feed (F3's per-step curves, F6's replay-
error series, F7's timing logs), and renders all 9 figures end to end so the
layout is inspectable before any real data exists.

Local usage:
    python evaluation/plots_gap.py --demo [--out_dir DIR]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import gap_metrics as gm  # noqa: E402
from evaluation.ur3_dynamics_replay import (  # noqa: E402
    time_to_divergence, DIVERGENCE_TCP_TOL_M,
)

DEFAULT_OUT_DIR = os.path.join(_THIS_DIR, "gap_plots_demo")

# ===========================================================================
# Shared style -- mirrors the conventions found in ur3_reward_replay.py /
# compare_reward_train_sim_real.py / run_robustness_eval*.py.
# ===========================================================================

_MK_FORMATTER = plt.FuncFormatter(
    lambda x, _: f"{x / 1e6:.0f}M" if x >= 1e6 else (
        f"{x / 1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"
    )
)
_TAB10 = plt.cm.tab10.colors
_TAB20 = plt.cm.tab20.colors
_SIM_COLOR = _TAB10[0]    # blue
_REAL_COLOR = _TAB10[3]   # red
_FAILED_MARKER = dict(marker="x", s=90, color="black", zorder=5)

# D14: the 4 real-deployed DR-ladder configs (L4_full/L5_full_obs never
# trained -- excluded everywhere downstream). Default panel set for the F3
# per-step trace (D15's 4-panel small multiple).
D14_REAL_POLICIES = ["L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot"]


def _config_color(i: int):
    return _TAB20[i % len(_TAB20)]


def _savefig(fig, out_path: str):
    """png+pdf @ dpi=300, bbox_inches="tight" -- the repo's existing pattern."""
    base, _ext = os.path.splitext(out_path)
    png_path, pdf_path = base + ".png", base + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ===========================================================================
# F1 -- dose-response curve.
# ===========================================================================

# D17: the size-DR axis (on in L2/L3) is hard-clamped to a 0.018 half-extent
# = 3.6cm cross-section for graspability -- the 4x4x4cm physical cube's 4cm
# cross-section (half-extent 0.020) sits OUTSIDE that clamp even for the
# configs that randomize size. Every 4cm-cube panel/figure/caption in this
# module states this explicitly (Task 6).
OOD_CAPTION_4CM = (
    "4cm cube is OUT-OF-DISTRIBUTION: outside the trained cube_size DR range "
    "(hard-clamped to 0.018 half-extent / 3.6cm cross-section, D17) -- not "
    "averaged with the 3cm in-distribution result."
)


def _f1_series(df, levels, failed_configs, cube_size):
    """Shared data prep for f1_dose_response / f1_dose_response_cube_split:
    per-level (sim_mean, sim_lo, sim_hi, real_mean, real_err) arrays for one
    cube_size. No plotting here -- see the module docstring's "no metric math
    lives here" (the arrays are means/min/max/D9-floor read straight from
    gap_metrics.py, not new statistics).
    """
    sub_all = gm._filter_cube_size(df, cube_size, "_f1_series")
    sim_mean, sim_lo, sim_hi, real_mean, real_err = [], [], [], [], []
    for c in levels:
        if c in failed_configs:
            sim_mean.append(np.nan); sim_lo.append(np.nan); sim_hi.append(np.nan)
            real_mean.append(np.nan); real_err.append(0.0)
            continue
        sub = sub_all[(sub_all["config_id"] == c) & (sub_all["domain"] == "sim")]
        per_seed = gm._run_totals(sub).groupby("seed")["total"].mean()
        sim_mean.append(float(per_seed.mean()))
        sim_lo.append(float(per_seed.min()))
        sim_hi.append(float(per_seed.max()))
        r = gm.retention(sub_all, c, cube_size=cube_size)
        real_mean.append(r["R_real"])
        real_err.append(gm.noise_floor(sub_all, c, cube_size=cube_size)["noise_floor"])
    return tuple(map(np.array, (sim_mean, sim_lo, sim_hi, real_mean, real_err)))


def f1_dose_response(df, levels, out_path, failed_configs=(), cube_size=None):
    """x = DR level (levels, in order), two lines: mean sim / mean real
    return, seed-spread shaded band (sim, min-max across seeds), noise-floor
    error bars on the real line. `failed_configs`: levels that never trained
    (marked with an 'x', excluded from both lines so they don't fake a dip).

    `cube_size` (D17): pin to "3cm"/"4cm" -- required if `df` mixes both (see
    gap_metrics._filter_cube_size). Single-cube only -- see
    `f1_dose_response_cube_split` for the 3cm-solid/4cm-dashed overlay (D17)
    that is the actual headline F1 figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(levels))

    sim_mean, sim_lo, sim_hi, real_mean, real_err = _f1_series(
        df, levels, failed_configs, cube_size)
    ok_idx = [i for i, c in enumerate(levels) if c not in failed_configs]

    ax.plot(x[ok_idx], sim_mean[ok_idx], "-o", color=_SIM_COLOR, label="sim (mean)")
    ax.fill_between(x[ok_idx], sim_lo[ok_idx], sim_hi[ok_idx], color=_SIM_COLOR,
                     alpha=0.2, label="sim seed spread (min-max)")
    ax.errorbar(x[ok_idx], real_mean[ok_idx], yerr=real_err[ok_idx], fmt="-s",
                color=_REAL_COLOR, capsize=4, label="real (mean +/- noise floor)")

    knee = int(np.nanargmax(real_mean)) if np.any(np.isfinite(real_mean)) else None
    if knee is not None:
        ax.axvline(x[knee], color=_REAL_COLOR, linestyle=":", alpha=0.5,
                   label=f"knee = {levels[knee]}")

    fail_idx = [i for i, c in enumerate(levels) if c in failed_configs]
    if fail_idx:
        ax.scatter(x[fail_idx], np.zeros(len(fail_idx)), **_FAILED_MARKER,
                   label="failed to train (excluded)")

    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=30, ha="right")
    ax.set_xlabel("DR level")
    ax.set_ylabel("episode return")
    title = "F1 -- dose-response: return vs. DR level"
    if cube_size == "4cm":
        title += "  [OOD -- 4cm cube]"
    ax.set_title(title, fontweight="bold")
    if cube_size == "4cm":
        ax.text(0.5, -0.16, OOD_CAPTION_4CM, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, style="italic")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, out_path)


def f1_dose_response_cube_split(df, levels, out_path, failed_configs=(),
                                cube_sizes=("3cm", "4cm")):
    """D17: the headline F1 rendering -- 3cm (in-distribution, solid) and
    4cm (OOD probe, dashed) overlaid on ONE dose-response figure, so the
    reader sees both the primary result and the OOD generalization probe on
    the same axes. 4cm is never averaged into 3cm anywhere (gap_metrics.py's
    `_filter_cube_size` guard, D17) -- this figure OVERLAYS the two, it does
    not pool them.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(levels))
    ok_idx = [i for i, c in enumerate(levels) if c not in failed_configs]
    fail_idx = [i for i, c in enumerate(levels) if c in failed_configs]
    linestyles = {"3cm": "-", "4cm": "--"}
    markers = {"3cm": "o", "4cm": "^"}

    for cube in cube_sizes:
        sim_mean, sim_lo, sim_hi, real_mean, real_err = _f1_series(
            df, levels, failed_configs, cube)
        ls = linestyles.get(cube, "-")
        mk = markers.get(cube, "o")
        ood_tag = " [OOD]" if cube == "4cm" else ""

        ax.plot(x[ok_idx], sim_mean[ok_idx], linestyle=ls, marker=mk,
               color=_SIM_COLOR, alpha=1.0 if cube == "3cm" else 0.75,
               label=f"sim, {cube}{ood_tag}")
        if cube == "3cm":  # seed-spread band only for the primary cube -- a
            # dashed+shaded 4cm band on top would clutter the overlay.
            ax.fill_between(x[ok_idx], sim_lo[ok_idx], sim_hi[ok_idx],
                            color=_SIM_COLOR, alpha=0.15,
                            label="sim seed spread (3cm, min-max)")
        ax.errorbar(x[ok_idx], real_mean[ok_idx], yerr=real_err[ok_idx],
                   fmt=ls + mk, color=_REAL_COLOR, capsize=4,
                   alpha=1.0 if cube == "3cm" else 0.75,
                   label=f"real, {cube}{ood_tag} (mean +/- noise floor)")

    if fail_idx:
        ax.scatter(x[fail_idx], np.zeros(len(fail_idx)), **_FAILED_MARKER,
                  label="failed to train (excluded)")

    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=30, ha="right")
    ax.set_xlabel("DR level")
    ax.set_ylabel("episode return")
    ax.set_title(
        "F1 -- dose-response: return vs. DR level, 3cm (solid) vs. 4cm (dashed, OOD)",
        fontweight="bold",
    )
    ax.text(0.5, -0.16, OOD_CAPTION_4CM, transform=ax.transAxes,
           ha="center", va="top", fontsize=7.5, style="italic")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, out_path)


# ===========================================================================
# F2 -- per-term / per-stage retention (diagnostic).
# ===========================================================================


def f2_term_retention(df, configs, out_path, cube_size=None):
    """Grouped bars, one group per term (gm.TERM_ORDER, which already starts
    gripper_box -> approach_open -> gripper_align -> grasp -> lift ->
    box_target), one bar per config within a group. Proxy terms
    (gm.PROXY_TERMS: grasp, no_floor_collision) are hatched.

    `cube_size` (D17): pin to "3cm"/"4cm" -- required if `df` mixes both.
    """
    n_cfg = len(configs)
    terms = gm.TERM_ORDER
    width = 0.8 / max(n_cfg, 1)
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, c in enumerate(configs):
        tr = gm.term_retention(df, c, cube_size=cube_size)["per_term"].reindex(terms)
        xpos = np.arange(len(terms)) + i * width - 0.4 + width / 2
        hatches = ["///" if t in gm.PROXY_TERMS else None for t in terms]
        for j, (xp, val, h) in enumerate(zip(xpos, tr["retention"].to_numpy(), hatches)):
            ax.bar(xp, val, width=width, color=_config_color(i), hatch=h,
                   edgecolor="black", linewidth=0.4,
                   label=c if j == 0 else None)

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(np.arange(len(terms)))
    ax.set_xticklabels(terms, rotation=30, ha="right")
    ax.set_ylabel("retention (real / sim)")
    ax.set_title("F2 -- per-term retention, ordered by pipeline stage\n"
                 "(DIAGNOSTIC / proposed -- no precedent for this figure; "
                 "hatched = proxy term, no direct real sensor)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, ncol=min(n_cfg, 4))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _savefig(fig, out_path)


# ===========================================================================
# F3 -- per-step exact-parity reward trace (D15; replaces the old cumulative
# grid, formerly `f3_grid`).
# ===========================================================================


def f3_per_step_trace(step_reward, out_path, policies=None):
    """Per-step EXACT-PARITY reward vs. timestep (D15's respec of the old
    cumulative-return grid) -- 4-panel small multiple, one panel per policy,
    sim vs. real overlaid: mean line + shaded +/-1 std band across the
    protocol's episodes. Cumulative return R = sum_t r_t (mean +/- std across
    episodes -- the RECORDED scalar, D15) is annotated as text per panel, so
    the per-step "where does the gap open" story and the per-episode
    magnitude both read off the same figure without conflating the two.

    step_reward: {config_id: {"sim": (n_episodes, H) array, "real":
    (n_episodes, H) array}} of PER-STEP (NOT cumulative) exact-parity reward
    -- i.e. each row already sums gap_metrics.EXACT_PARITY_TERMS (D8) for
    that step; this function does no term selection or summation of its own
    (no metric math lives here, per the module docstring). sim and real may
    have different n_episodes (aborted real repeats are excluded upstream,
    D16) but must share H.

    `policies`: ordered config_ids, one panel each -- default
    D14_REAL_POLICIES (the 4 real-deployed ladder rungs), restricted to keys
    actually present in `step_reward`.
    """
    if policies is None:
        policies = [c for c in D14_REAL_POLICIES if c in step_reward]
    if not policies:
        raise ValueError("f3_per_step_trace: no policies to plot (step_reward "
                          "is empty or shares no keys with D14_REAL_POLICIES / "
                          "the given `policies`).")

    fig, axes = plt.subplots(1, len(policies), figsize=(4.6 * len(policies), 4.4),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for ax, cid in zip(axes, policies):
        for dom, color, y_txt in (("sim", _SIM_COLOR, 0.96), ("real", _REAL_COLOR, 0.86)):
            arr = np.asarray(step_reward[cid][dom], dtype=float)  # (n_ep, H)
            H = arr.shape[1]
            x = np.arange(H)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(x, mean, color=color, linewidth=2.0, label=dom)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

            cum_R = arr.sum(axis=1)  # per-episode cumulative return
            R_mean, R_std = float(cum_R.mean()), float(cum_R.std())
            ax.text(0.03, y_txt, f"R_{dom} = {R_mean:.1f} +/- {R_std:.1f}",
                   transform=ax.transAxes, fontsize=8, color=color,
                   va="top", ha="left")

        ax.set_title(cid, fontweight="bold", fontsize=10)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("per-step exact-parity reward")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "F3 -- per-step exact-parity reward trace, sim vs. real "
        "(mean +/- 1 std across episodes; D15) -- shows WHEN in the episode "
        "the gap opens; R annotated per panel is the recorded cumulative scalar",
        fontweight="bold", fontsize=10, y=1.06,
    )
    plt.tight_layout()
    return _savefig(fig, out_path)


# ===========================================================================
# F4 -- paired scatter (sim vs real episode return).
# ===========================================================================


def f4_paired_scatter(df, config, out_path, cube_size=None, n_boot=10000, seed=0):
    """`cube_size` (D17): pin to "3cm"/"4cm" -- required if `df` mixes both."""
    piv = gm.per_episode_returns(df, config, cube_size=cube_size)
    sim, real = piv["sim"].to_numpy(), piv["real"].to_numpy()
    floor = gm.noise_floor(df, config, cube_size=cube_size)["noise_floor"]

    rng = np.random.default_rng(seed)
    idx = gm._boot_episode_indices(len(piv), n_boot, rng)
    gap_samples = np.mean(sim[idx] - real[idx], axis=1)
    gap_mean = float(np.mean(sim - real))
    gap_ci = gm._percentile_ci(gap_samples)

    fig, ax = plt.subplots(figsize=(7, 7))
    lo = min(sim.min(), real.min()) * 0.95
    hi = max(sim.max(), real.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.errorbar(sim, real, yerr=floor, fmt="o", color=_TAB10[2], capsize=3,
               markersize=6, alpha=0.85, label="episode (real +/- noise floor)")

    ax.text(
        0.05, 0.95,
        f"mean signed gap (sim-real) = {gap_mean:.2f}\n"
        f"95% CI [{gap_ci[0]:.2f}, {gap_ci[1]:.2f}]",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("sim episode return"); ax.set_ylabel("real episode return")
    ax.set_title(f"F4 -- paired sim/real return, {config}\n"
                 "(no correlation coefficient -- see gap_metrics.py's rationale)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    return _savefig(fig, out_path)


# ===========================================================================
# F5 -- sim-selection regret (two panels).
# ===========================================================================


def f5_selection_regret(df, configs, out_path, cube_size=None, n_boot=10000, seed=0):
    """`cube_size` (D17): pin to "3cm"/"4cm" -- required if `df` mixes both."""
    res = gm.sim_selection_regret(df, configs=configs, cube_size=cube_size,
                                  n_boot=n_boot, seed=seed)
    floor = float(np.mean([
        gm.noise_floor(df, c, cube_size=cube_size)["noise_floor"] for c in configs
    ]))

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 3], hspace=0.55, wspace=0.3)
    ax_bars = fig.add_subplot(gs[0, :])
    ax_freq = fig.add_subplot(gs[1, :], sharex=ax_bars)
    ax_hist = fig.add_subplot(gs[2, :])

    table = res["table"].set_index("config_id").loc[configs]
    x = np.arange(len(configs))
    w = 0.35
    ax_bars.bar(x - w / 2, table["R_sim"], width=w, color=_SIM_COLOR, label="R_sim")
    ax_bars.bar(x + w / 2, table["R_real"], width=w, color=_REAL_COLOR, label="R_real")
    chat_i = configs.index(res["chat_sim"])
    cstar_i = configs.index(res["c_star"])
    ax_bars.annotate("chat_sim", (chat_i - w / 2, table["R_sim"].iloc[chat_i]),
                     textcoords="offset points", xytext=(0, 8), ha="center",
                     fontweight="bold", color=_SIM_COLOR)
    ax_bars.annotate("c*", (cstar_i + w / 2, table["R_real"].iloc[cstar_i]),
                     textcoords="offset points", xytext=(0, 8), ha="center",
                     fontweight="bold", color=_REAL_COLOR)
    r_chat_real = float(table["R_real"].iloc[chat_i])
    r_cstar_real = float(table["R_real"].iloc[cstar_i])
    ax_bars.plot([chat_i + w / 2, cstar_i + w / 2], [r_chat_real, r_cstar_real],
                color="black", linewidth=1.5, linestyle=":")
    ax_bars.annotate(f"R = {res['regret']:.2f}",
                     ((chat_i + cstar_i) / 2 + w / 2,
                      (r_chat_real + r_cstar_real) / 2),
                     ha="center", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax_bars.set_xticks(x); ax_bars.set_xticklabels(configs, rotation=20, ha="right")
    ax_bars.set_ylabel("episode return")
    ax_bars.set_title("F5a -- sim-selection regret: R_sim vs R_real per config",
                      fontweight="bold")
    ax_bars.legend(fontsize=8)
    ax_bars.grid(True, alpha=0.3, axis="y")

    freqs = [res["selection_freq"][c] for c in configs]
    ax_freq.bar(x, freqs, color=[_config_color(i) for i in range(len(configs))])
    ax_freq.set_ylabel("P(chat_sim)", fontsize=8)
    ax_freq.set_ylim(0, 1)
    plt.setp(ax_freq.get_xticklabels(), visible=False)
    ax_freq.grid(True, alpha=0.3, axis="y")

    samples = res["regret_samples"]
    ax_hist.hist(samples, bins=40, color=_TAB10[4], alpha=0.85, density=True)
    ax_hist.axvspan(-floor, floor, color="gray", alpha=0.25, label=f"noise floor (+/-{floor:.2f})")
    ax_hist.axvline(res["regret"], color="black", linewidth=1.5, label=f"point est. R={res['regret']:.2f}")
    ax_hist.set_xlabel("bootstrap regret R (resampled-argmax)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("F5b -- bootstrap distribution of R (no minimum-n gate)",
                      fontweight="bold")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    return _savefig(fig, out_path)


# ===========================================================================
# F6 -- dynamics-replay error vs step.
# ===========================================================================


def f6_replay_error(one_step_df, open_loop_df, out_path,
                    tol_m=DIVERGENCE_TCP_TOL_M):
    ttd = time_to_divergence(open_loop_df, tol_m=tol_m)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(one_step_df["step"], one_step_df["joint_sq_err"], color=_TAB10[0],
             label="one-step (re-synced every tick)")
    ax1.plot(open_loop_df["step"], open_loop_df["joint_sq_err"], color=_TAB10[3],
             label="open-loop (free-running)")
    ax1.set_ylabel("E_replay (joint, rad^2)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    ax1.set_title("F6 -- dynamics-replay error vs. step "
                 "(Aljalbout Sec. 5.1 style; time-to-divergence NOT standard "
                 "terminology -- defined in ur3_dynamics_replay.py)",
                 fontweight="bold", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(open_loop_df["step"], open_loop_df["tcp_err_m"], color=_TAB10[3])
    ax2.axhline(tol_m, color="black", linestyle="--", alpha=0.6,
               label=f"divergence threshold ({tol_m * 100:.0f} cm)")
    if ttd == ttd:  # not NaN
        ax2.axvline(ttd, color="black", linewidth=1.5,
                   label=f"time-to-divergence = step {ttd}")
    ax2.set_xlabel("step"); ax2.set_ylabel("open-loop TCP error (m)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, out_path)


# ===========================================================================
# F7 -- timing honesty (before/after gripper-thread decoupling).
# ===========================================================================


def f7_timing_honesty(timing_by_condition, out_path):
    """timing_by_condition: {"before": {"hz": array, "dt": array}, "after": {...}}."""
    conds = list(timing_by_condition.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {conds[0]: _TAB10[3], conds[-1]: _TAB10[0]} if len(conds) >= 2 else {}

    for cond in conds:
        hz = timing_by_condition[cond]["hz"]
        ax1.hist(hz, bins=30, alpha=0.6, label=cond,
                color=colors.get(cond, _config_color(conds.index(cond))))
    ax1.axvline(50.0, color="black", linestyle="--", alpha=0.6, label="nominal 50 Hz")
    ax1.set_xlabel("achieved control-loop Hz"); ax1.set_ylabel("count")
    ax1.set_title("Achieved Hz", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for cond in conds:
        dt = timing_by_condition[cond]["dt"]
        ax2.plot(np.arange(len(dt)), dt * 1000, alpha=0.8, label=cond,
                color=colors.get(cond, _config_color(conds.index(cond))))
    ax2.axhline(20.0, color="black", linestyle="--", alpha=0.6, label="nominal 20 ms")
    ax2.set_xlabel("step"); ax2.set_ylabel("loop_dt_true_s (ms)")
    ax2.set_title("Per-step loop duration", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("F7 -- timing honesty: before vs. after gripper-thread decoupling",
                fontweight="bold", y=1.03)
    plt.tight_layout()
    return _savefig(fig, out_path)


# ===========================================================================
# F8 -- LOO attribution.
# ===========================================================================


def f8_loo_attribution(df, full_config, loo_configs, out_path, cube_size=None):
    """Sim-return drop for each leave-one-cluster-out config vs. the full
    config. SIM ONLY -- see the caption (no real-robot LOO data collected).

    `cube_size` (D17): pin to "3cm"/"4cm" -- required if `df` mixes both (the
    LOO sim rollouts are cube-agnostic in practice, but the guard still
    applies if the table happens to carry both).
    """
    r_full = gm.retention(df, full_config, cube_size=cube_size)["R_sim"]
    drops, labels = [], []
    for c in loo_configs:
        r_c = gm.retention(df, c, cube_size=cube_size)["R_sim"]
        drops.append(r_full - r_c)
        labels.append(c)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_config_color(i) for i in range(len(labels))]
    ax.bar(labels, drops, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel(f"sim-return drop vs. {full_config}")
    ax.set_title("F8 -- leave-one-cluster-out attribution (SIM ONLY)",
                fontweight="bold")
    ax.text(
        0.5, -0.28,
        "Attributes the SIMULATED policy's dependence on each DR cluster.\n"
        "Does NOT claim to attribute the real-world sim-to-real gap -- that "
        "would require real-robot LOO data, not collected by default.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, style="italic",
    )
    plt.xticks(rotation=20, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _savefig(fig, out_path)


# ===========================================================================
# F9 -- abort rate by (policy, cube_size) (D16/D17, Task 6).
# ===========================================================================


def f9_abort_rate_bar(df, configs, out_path, cube_sizes=("3cm", "4cm")):
    """Grouped bar chart, one group per policy, one bar per cube_size:
    fraction of real runs with stop_reason == "abort" (gap_metrics.
    abort_rate, D16/D17). D17 explicitly predicts this will be HIGH for the
    4cm cube (Hand-E clearance drops to ~0.5cm/side, right at the mechanical
    grasp limit) -- expect it and report it as a finding, not a bug.
    """
    n_cube = len(cube_sizes)
    width = 0.8 / max(n_cube, 1)
    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, cube in enumerate(cube_sizes):
        rates = [gm.abort_rate(df, c, cube_size=cube)["abort_rate"] for c in configs]
        xpos = x + i * width - 0.4 + width / 2
        hatch = "///" if cube == "4cm" else None
        label = f"{cube}" + (" [OOD]" if cube == "4cm" else "")
        ax.bar(xpos, rates, width=width, color=_config_color(i), hatch=hatch,
              edgecolor="black", linewidth=0.5, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("abort rate (fraction of real runs, stop_reason == abort)")
    ax.set_title("F9 -- abort rate by policy x cube_size", fontweight="bold")
    ax.text(
        0.5, -0.32,
        "D16/D17: abort = protective stop / RTDE error / mocap loss / hang watchdog\n"
        "(real robot only -- sim never aborts). 4cm is OOD (hatched, outside the\n"
        "trained cube_size DR range, D17); a high 4cm abort rate is an expected,\n"
        "reportable finding, not a bug.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, style="italic",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _savefig(fig, out_path)


# ===========================================================================
# --demo: fabricate everything and render all 9 figures.
# ===========================================================================

_DEMO_LADDER = ["L0_none", "L1_pos", "L2_pos_cube", "L3_pos_cube_robot",
               "L4_full", "L5_full_obs"]
_DEMO_LOO = ["LOO_no_pos", "LOO_no_cube", "LOO_no_robot", "LOO_no_env"]
_DEMO_FAILED = {"L3_pos_cube_robot"}  # marked visually distinct in F1


_DEMO_RETENTION_FRAC_3CM = {
    "L0_none": 0.85, "L1_pos": 0.75, "L2_pos_cube": 0.65,
    "L3_pos_cube_robot": 0.5, "L4_full": 0.55, "L5_full_obs": 0.60,
    "LOO_no_pos": 0.62, "LOO_no_cube": 0.58, "LOO_no_robot": 0.64,
    "LOO_no_env": 0.61,
}
# D17 OOD probe: sim_total is UNCHANGED from 3cm (the policy has no size
# channel in obs and cannot perceive which physical cube it faces -- same
# simulated behaviour either way). Real retention drops further everywhere,
# but less so for the size-DR-on configs (L2/L3 saw +/-10% size jitter in
# training) than the position-DR-only ones -- the story this probe tests.
_DEMO_RETENTION_FRAC_4CM = {
    "L0_none": 0.35, "L1_pos": 0.30, "L2_pos_cube": 0.45,
    "L3_pos_cube_robot": 0.38, "L4_full": 0.20, "L5_full_obs": 0.22,
    "LOO_no_pos": 0.25, "LOO_no_cube": 0.22, "LOO_no_robot": 0.24,
    "LOO_no_env": 0.26,
}
_DEMO_BASE_SIM = {
    "L0_none": 140, "L1_pos": 150, "L2_pos_cube": 155,
    "L3_pos_cube_robot": 60, "L4_full": 160, "L5_full_obs": 158,
    "LOO_no_pos": 145, "LOO_no_cube": 150, "LOO_no_robot": 148,
    "LOO_no_env": 152,
}


# D17: "expect a high 4cm failure/abort rate -- that is itself a result, not
# a bug" (Hand-E clearance drops to ~0.5cm/side at 4cm, right at the
# mechanical grasp limit). The demo injects a low background abort rate for
# 3cm and a much higher one for 4cm, consistently across configs -- just
# enough structure that F9's bar chart isn't flat.
_DEMO_ABORT_FRAC_3CM = 0.05
_DEMO_ABORT_FRAC_4CM = 0.35


def _make_demo_gap_metrics_csv(rng):
    """Reuses gap_metrics.make_synthetic_long -- the SAME fabrication
    gap_metrics.py's own --selftest uses, not a second one. Builds BOTH cube
    sizes (D17): a 3cm (in-distribution) table and a 4cm (OOD probe) table,
    concatenated -- exercises gap_metrics.py's cube_size grouping guard end
    to end, the same shape a real merged gap_metrics.csv will have. Real rows
    get a synthetic abort rate (low for 3cm, high for 4cm, D17) so F9's
    abort-rate bar chart has something to show.
    """
    specs_3cm = {
        cid: {"sim_total": float(s), "real_total": float(s * _DEMO_RETENTION_FRAC_3CM[cid])}
        for cid, s in _DEMO_BASE_SIM.items()
    }
    specs_4cm = {
        cid: {"sim_total": float(s), "real_total": float(s * _DEMO_RETENTION_FRAC_4CM[cid])}
        for cid, s in _DEMO_BASE_SIM.items()
    }
    df_3cm = gm.make_synthetic_long(
        specs_3cm, n_episodes=10, seeds=(0, 1, 2), repeats=(1, 2, 3),
        ep_noise=6.0, repeat_noise=2.5, seed_noise=3.0, cube_size="3cm",
        abort_frac=_DEMO_ABORT_FRAC_3CM, rng=rng,
    )
    df_4cm = gm.make_synthetic_long(
        specs_4cm, n_episodes=10, seeds=(0, 1, 2), repeats=(1, 2, 3),
        ep_noise=6.0, repeat_noise=4.0, seed_noise=3.0, cube_size="4cm",
        abort_frac=_DEMO_ABORT_FRAC_4CM, rng=rng,
    )
    return pd.concat([df_3cm, df_4cm], ignore_index=True)


def _make_demo_step_curves(rng, configs, n_episodes=10, H=400):
    """PER-STEP (NOT cumulative) synthetic exact-parity reward for
    f3_per_step_trace: a single approach/hold-shaped bump (so the trace has
    structure instead of being flat noise), real a scaled-down and noisier
    version of sim -- a plausible per-step sim-to-real gap. n_episodes/H
    match the frozen protocol (10 episodes, H=400).
    """
    t = np.linspace(0.0, 1.0, H)
    shape = np.exp(-((t - 0.55) ** 2) / (2 * 0.12 ** 2))
    out = {}
    for cid in configs:
        curves = {}
        for dom, scale, noise in (("sim", 1.0, 0.05), ("real", 0.75, 0.09)):
            base = shape * scale
            curves[dom] = base[None, :] + rng.normal(0, noise, size=(n_episodes, H))
        out[cid] = curves
    return out


def _make_demo_replay_dfs(rng, T=400):
    step = np.arange(T)
    one_step = pd.DataFrame({
        "step": step,
        "joint_sq_err": np.abs(rng.normal(1e-4, 3e-5, T)),
        "tcp_err_m": np.abs(rng.normal(0.002, 0.0005, T)),
    })
    # Open-loop error grows roughly monotonically (compounding), crossing 1cm
    # partway through -- a plausible divergence curve.
    growth = 0.0008 * step + rng.normal(0, 0.0006, T).cumsum() * 0.02
    tcp_err = np.clip(np.abs(growth), 0, None)
    open_loop = pd.DataFrame({
        "step": step,
        "joint_sq_err": np.clip(1e-5 * step + rng.normal(0, 1e-5, T).cumsum() * 0.01, 0, None),
        "tcp_err_m": tcp_err,
    })
    return one_step, open_loop


def _make_demo_timing(rng, T=400):
    return {
        "before (coupled gripper thread)": {
            "hz": rng.normal(46.0, 4.0, 500),
            "dt": np.clip(rng.normal(0.0217, 0.006, T), 0.005, None),
        },
        "after (decoupled)": {
            "hz": rng.normal(50.3, 0.6, 500),
            "dt": np.clip(rng.normal(0.0199, 0.0008, T), 0.005, None),
        },
    }


def run_demo(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    df = _make_demo_gap_metrics_csv(rng)
    all_configs = _DEMO_LADDER + _DEMO_LOO
    # D17: the demo table now carries both 3cm and 4cm rows (gap_metrics.py's
    # cube_size guard would otherwise refuse to run any of these). f2/f4/f5/f8
    # are not cube-split figures (only F1 + the abort-rate bar are, D17/Task
    # 6) -- pin to the primary in-distribution cube here.
    PRIMARY_CUBE = "3cm"

    outputs = []
    # D17/Task 6: F1's headline rendering is now the 3cm/4cm cube-split
    # overlay (solid vs. dashed, OOD-labelled) -- supersedes a single-cube
    # f1_dose_response() call as the "f1_dose_response" output.
    outputs.append(f1_dose_response_cube_split(
        df, _DEMO_LADDER, os.path.join(out_dir, "f1_dose_response"),
        failed_configs=_DEMO_FAILED))
    outputs.append(f2_term_retention(df, _DEMO_LADDER[:4], os.path.join(out_dir, "f2_term_retention"),
                                     cube_size=PRIMARY_CUBE))
    step_curves = _make_demo_step_curves(rng, D14_REAL_POLICIES)
    outputs.append(f3_per_step_trace(step_curves, os.path.join(out_dir, "f3_per_step_trace")))
    outputs.append(f4_paired_scatter(df, "L1_pos", os.path.join(out_dir, "f4_paired_scatter"),
                                     cube_size=PRIMARY_CUBE))
    outputs.append(f5_selection_regret(df, _DEMO_LADDER, os.path.join(out_dir, "f5_selection_regret"),
                                       cube_size=PRIMARY_CUBE))
    one_step, open_loop = _make_demo_replay_dfs(rng)
    outputs.append(f6_replay_error(one_step, open_loop, os.path.join(out_dir, "f6_replay_error")))
    timing = _make_demo_timing(rng)
    outputs.append(f7_timing_honesty(timing, os.path.join(out_dir, "f7_timing_honesty")))
    outputs.append(f8_loo_attribution(df, "L5_full_obs", _DEMO_LOO, os.path.join(out_dir, "f8_loo_attribution"),
                                      cube_size=PRIMARY_CUBE))
    # Task 6: new abort-rate-by-(policy, cube_size) grouped bar chart (D16/D17).
    outputs.append(f9_abort_rate_bar(df, _DEMO_LADDER[:4],
                                     os.path.join(out_dir, "f9_abort_rate_bar")))

    print(f"\n--demo wrote {len(outputs)} figures (png+pdf each) to {out_dir}:")
    for png in outputs:
        pdf = png.replace(".png", ".pdf")
        for p in (png, pdf):
            size = os.path.getsize(p)
            print(f"  {p}  ({size:,} bytes)")
    return outputs


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if not args.demo:
        print("Nothing to do -- pass --demo (this module has no non-demo CLI "
              "entry point; call the f1..f8 functions directly from a driver "
              "script once real gap_metrics.csv / replay / timing data exist).")
        return
    run_demo(args.out_dir)


if __name__ == "__main__":
    _cli()
