"""Sim-to-real gap metrics over a tidy long-format results table.

The table (gap_metrics.csv) is LONG: one row per
(config_id, seed, episode_id, repeat, term, domain) with the episode-summed
scaled reward `scaled_return_H` for that term, plus per-run bookkeeping
(achieved_hz, overrun_count, stop_reason, protocol_hash). `domain` is "sim" or
"real". A run's total return is the sum of its term rows; a term's stage is
fixed by STAGE_MAP.

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

LONG_COLUMNS = [
    "config_id", "seed", "episode_id", "repeat", "term", "domain",
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
    """Sum term rows -> one total per (config, domain, seed, episode, repeat)."""
    keys = ["config_id", "domain", "seed", "episode_id", "repeat"]
    keys = [k for k in keys if k in df.columns]
    return (
        df.groupby(keys, dropna=False)["scaled_return_H"].sum()
        .reset_index().rename(columns={"scaled_return_H": "total"})
    )


def per_episode_returns(
    df: pd.DataFrame, config: str, episodes=None
) -> pd.DataFrame:
    """Paired per-episode returns for one config.

    Returns a DataFrame indexed by episode_id with columns ['sim','real'] --
    each the mean total return over that episode's seeds (sim) / repeats (real).
    Only episodes present in BOTH domains are kept (pairing requirement).
    """
    tot = _run_totals(df[df["config_id"] == config])
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


def retention(df, config, subset="exact", episodes=None, n_boot=10000, seed=0):
    """Real/sim return retention for one config + paired-bootstrap 95% CI.

    ratio = mean_episode(real) / mean_episode(sim), on RAW returns (no
    normalisation -- see module docstring). CI resamples episodes.
    `subset` is a label recorded in the result (e.g. "exact" = exact-init
    matched episodes); pass `episodes` to restrict to a subset's episode_ids.
    """
    piv = per_episode_returns(df, config, episodes=episodes)
    real = piv["real"].to_numpy()
    sim = piv["sim"].to_numpy()
    n = len(piv)
    point = float(np.mean(real) / np.mean(sim))

    rng = np.random.default_rng(seed)
    idx = _boot_episode_indices(n, n_boot, rng)
    ratios = np.mean(real[idx], axis=1) / np.mean(sim[idx], axis=1)
    lo, hi = _percentile_ci(ratios)
    return {
        "config_id": config, "subset": subset, "n_episodes": n,
        "R_sim": float(np.mean(sim)), "R_real": float(np.mean(real)),
        "retention": point, "ci95": (lo, hi),
    }


def noise_floor(df, config, episodes=None):
    """Within-episode real spread across the k repeats (D9's measurement floor).

    For each episode: std of the real total return over its repeats. The floor
    is the mean of those per-episode stds (return units); nothing smaller than
    it is reportable. Returns per-episode detail too.
    """
    tot = _run_totals(df[(df["config_id"] == config) & (df["domain"] == "real")])
    if episodes is not None:
        tot = tot[tot["episode_id"].isin(episodes)]
    per_ep = (
        tot.groupby("episode_id")["total"]
        .agg(["mean", "std", "count", "min", "max"])
        .rename(columns={"std": "repeat_std"})
    )
    per_ep["repeat_std"] = per_ep["repeat_std"].fillna(0.0)
    floor = float(per_ep["repeat_std"].mean())
    return {
        "config_id": config,
        "noise_floor": floor,                 # mean within-episode repeat std
        "noise_floor_band": (-floor, floor),  # symmetric reportability band
        "per_episode": per_ep,
        "mean_repeats": float(per_ep["count"].mean()),
    }


def term_retention(df, config, episodes=None):
    """Per-term AND per-stage real/sim retention ratios for one config.

    Each ratio = mean real term (stage) return / mean sim term (stage) return,
    over episodes and seeds/repeats. Raw returns, no normalisation.
    """
    sub = df[df["config_id"] == config]
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
        "config_id": config,
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


def sim_selection_regret(df, configs=None, n_boot=10000, seed=0):
    """Sim-selection regret R = R_real(c*) - R_real(argmax_c R_sim).

    c*      = argmax_c R_real (the config you WOULD have picked with real data)
    chat_sim = argmax_c R_sim (the config you DO pick from sim alone)
    R >= 0; it is the real-return you lose by selecting on sim.

    Paired bootstrap resamples episodes and RECOMPUTES BOTH ARGMAXES inside each
    resample -- fixing chat_sim at the point estimate drops selection
    uncertainty (which dominates at ~6 configs) and understates the CI. Both the
    resampled-argmax CI and the (narrower) fixed-chat_sim CI are returned so the
    understatement is visible; the reportable CI is the resampled one.
    """
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
        "configs": configs,
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
# Synthetic long-format data (shared with the plots' --demo and the selftest)
# ===========================================================================


def make_synthetic_long(
    config_specs, n_episodes=10, seeds=(0, 1, 2), repeats=(1, 2, 3),
    ep_noise=0.0, repeat_noise=0.0, seed_noise=0.0, protocol_hash="demo_hash",
    rng=None,
):
    """Fabricate a plausible long-format table.

    config_specs: {config_id: {"sim_total": float, "real_total": float,
                               "term_frac": {term: frac} (optional)}}.
    Per-episode base drawn N(total, ep_noise); sim seeds jitter by seed_noise;
    real repeats jitter by repeat_noise. Totals are split across terms by
    term_frac (default: an approach-heavy split summing to 1).
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
                        term=term, domain="sim", scaled_return_H=tot * f,
                        achieved_hz=np.nan, overrun_count=0,
                        stop_reason="rollout", protocol_hash=protocol_hash,
                    ))
            for rp in repeats:
                tot = ep_real + rng.normal(0, repeat_noise)
                for term, f in frac.items():
                    rows.append(dict(
                        config_id=cid, seed=-1, episode_id=ep, repeat=rp,
                        term=term, domain="real", scaled_return_H=tot * f,
                        achieved_hz=50.4, overrun_count=0,
                        stop_reason="timeout", protocol_hash=protocol_hash,
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

    # ---- gate_against_noise_floor sanity ----
    print("gate_against_noise_floor ...", end=" ")
    assert gate_against_noise_floor(10.0, (6.0, 14.0), floor=5.0) == "reportable"
    assert gate_against_noise_floor(3.0, (-1.0, 7.0), floor=5.0) == "within_noise_floor"
    assert gate_against_noise_floor(-10.0, (-14.0, -6.0), 5.0) == "reportable"
    print("OK")

    print("\nSELFTEST OK (no MJX, no robot, no correlation, no normalisation).")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    if args.selftest or not args.csv:
        _selftest()
        return

    df = load_long(args.csv)
    configs = sorted(df["config_id"].unique())
    print(f"Loaded {len(df)} rows, configs={configs}")
    if args.config:
        r = retention(df, args.config, n_boot=args.n_boot)
        nf = noise_floor(df, args.config)
        print(f"\nretention({args.config}) = {r['retention']:.3f} "
              f"CI{tuple(round(x,3) for x in r['ci95'])}  "
              f"[R_sim={r['R_sim']:.1f} R_real={r['R_real']:.1f}]")
        print(f"noise_floor({args.config}) = {nf['noise_floor']:.3f} (return units)")
        print(f"gate: {gate_against_noise_floor(r['R_real']-r['R_sim'], r['ci95'], nf['noise_floor'])}")
        print("\nterm_retention:")
        print(term_retention(df, args.config)["per_stage"])
    if len(configs) >= 2:
        res = sim_selection_regret(df, n_boot=args.n_boot)
        print(f"\nsim_selection_regret: R={res['regret']:.2f} "
              f"CI{tuple(round(x,2) for x in res['ci95_resampled_argmax'])}  "
              f"chat_sim={res['chat_sim']} c*={res['c_star']}")
        print(res["table"])


if __name__ == "__main__":
    _cli()
