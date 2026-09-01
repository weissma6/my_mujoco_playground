"""Pin WindowedTrendTracker's stop/no-stop verdict against real archived W&B curves.

This is the regression guard `replay_real_curve.py` is not: that script is a
report meant to be read by a human before choosing defaults. This test PINS a
result that has already been verified by hand, so a future change to
`WindowedTrendTracker` (or its defaults) that silently flips a verdict fails
loudly here instead of only showing up as a worse ladder weeks later.

Motivation (see `early_stop.py`'s `WindowedTrendTracker` docstring and the
curriculum-v2 plan): `PatienceTracker`'s running-max ratchet died twice on
noise on SLURM job 50874 --

  * Curr_L3_pos_cube_robot_s0_... was still climbing (eval reward 5356 ->
    6613) when PatienceTracker killed it at step 19,660,800.
  * Curr_L4_full_s0_... died on a 0.03% near-miss: eval 6123.0 against a
    6125.1 threshold.

Replaying both curves through WindowedTrendTracker at the same window=4 /
patience=3 / min_delta=0.02 must show it does NOT fire on either -- that is
the whole point of moving to a windowed-mean comparison. Replaying three
from-scratch velocity-DR-ladder curves that ran to a full, sane budget must
show it DOES fire on all three, so the new tracker is not simply "never
stops" in disguise.

Curves are fetched once from W&B (entity
weissma6-zhaw-school-of-engineering, project UR3_pick_ppo) and cached to
`_windowed_curve_cache.json` next to this file, so re-runs are OFFLINE and
the pinned numbers are reproducible without a W&B login. Delete that file and
re-run with network access (and `wandb login` done) to refresh it.

Two W&B gotchas, already paid for -- do not rediscover them:

  1. `run.history(keys=[...], samples=N)` returns EMPTY in this project's
     wandb version (0.23.1). Use `run.history(samples=500, pandas=False)`
     with NO keys filter, then pull the metric and `_step` out of the
     returned dicts yourself, skipping rows where the metric is absent/None.
  2. Do NOT use `scan_history` -- it walks this project's ~25M-wide step
     index and takes >15 minutes PER RUN. The sampled endpoint returns the
     same eval points (these runs only ever log 20-40 of them) in about a
     second.

Run:
    python -m pytest learning/curriculum/tests/test_windowed_replay.py -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from learning.curriculum.early_stop import WindowedTrendTracker  # noqa: E402

ENTITY = "weissma6-zhaw-school-of-engineering"
PROJECT = "UR3_pick_ppo"
METRIC = "eval/episode_reward"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_windowed_curve_cache.json")
# Fallback source only -- test_early_stop's sibling replay tool already fetched
# the three *_vel_s1 curves into this file. We do not own it and never write
# to it; a run missing from our own cache may still be recoverable from here
# without a network round-trip.
FALLBACK_CACHE = os.path.join(HERE, "_curve_cache.json")

# Curves that should NEVER stop: still-learning rungs from the (scale-bugged
# but otherwise real) curriculum run, killed early by PatienceTracker's noise
# sensitivity rather than by genuine convergence.
NEVER_STOP_RUNS = [
    "Curr_L3_pos_cube_robot_s0_20260831_193912_6311",
    "Curr_L4_full_s0_20260831_195449_6311",
]

# Curves that SHOULD stop: from-scratch velocity-DR-ladder rungs that ran to
# a full ~32M-step budget, well past any reasonable convergence point.
DOES_STOP_RUNS = [
    "L0_none_vel_s1_20260729_104930_2201",
    "L3_pos_cube_robot_vel_s1_20260729_115408_2201",
    "L4_full_vel_s1_20260729_122736_2201",
]

ALL_RUNS = NEVER_STOP_RUNS + DOES_STOP_RUNS


def _load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _fetch_curve(run_id, cache, fallback):
    """Return [[step, reward], ...] for one run: our cache, then the sibling
    cache as a fallback, then W&B. Raises (not skips) if none of those work --
    a missing curve here is a hard failure, not something to silently drop.
    """
    if run_id in cache:
        return cache[run_id]
    if run_id in fallback:
        pts = fallback[run_id]
        cache[run_id] = pts
        return pts

    import wandb  # local import: keep this module importable with no wandb

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    # See module docstring gotcha (1): no `keys=` filter, samples=500, pull
    # the fields out by hand. See gotcha (2): never scan_history.
    hist = run.history(samples=500, pandas=False)
    pts = sorted(
        [int(row["_step"]), float(row[METRIC])]
        for row in hist
        if row.get(METRIC) is not None
    )
    cache[run_id] = pts
    return pts


def _build_curves():
    cache = _load_json(CACHE)
    fallback = _load_json(FALLBACK_CACHE)
    curves = {}
    errors = {}
    for run_id in ALL_RUNS:
        try:
            pts = _fetch_curve(run_id, cache, fallback)
        except Exception as e:  # noqa: BLE001 - collected below, not swallowed
            errors[run_id] = f"{type(e).__name__}: {e}"
            continue
        if not pts:
            errors[run_id] = "fetched but no eval points"
            continue
        curves[run_id] = pts
    if cache:
        _save_cache(cache)
    return curves, errors


_CURVES, _FETCH_ERRORS = _build_curves()


def _replay(pts, window=4, patience=3, min_delta=0.02, min_steps=0):
    """Feed a [[step, reward], ...] curve through a fresh tracker.

    min_steps=0: these are all real, long, past-convergence-region curves,
    so the floor is not what is under test here and is set low enough to
    never be the reason a curve does or does not stop (window=4 already
    requires 8 evals of history before any verdict is possible at all).
    """
    t = WindowedTrendTracker(
        window=window, patience=patience, min_delta=min_delta, min_steps=min_steps
    )
    for step, reward in pts:
        if t.update(step, {METRIC: reward}):
            return t, step
    return t, None


@pytest.fixture(autouse=True)
def _require_all_curves_fetched():
    if _FETCH_ERRORS:
        pytest.fail(
            "could not obtain the following real W&B curves (hard failure, "
            "not a skip -- see test_windowed_replay.py module docstring for "
            "the caching/fallback path):\n"
            + "\n".join(f"  {run_id}: {err}" for run_id, err in _FETCH_ERRORS.items())
        )


@pytest.mark.parametrize("run_id", NEVER_STOP_RUNS)
def test_still_learning_curves_never_stop(run_id):
    """Curr_L3/Curr_L4 were killed by PatienceTracker's noise sensitivity while
    still improving. WindowedTrendTracker at the same window=4/patience=3/
    min_delta=0.02 must never fire on either.
    """
    pts = _CURVES[run_id]
    t, stop = _replay(pts, window=4, patience=3, min_delta=0.02, min_steps=0)
    assert stop is None, (
        f"{run_id} was expected to NEVER stop (it was still learning when "
        f"PatienceTracker killed it) but WindowedTrendTracker stopped it at "
        f"step {stop:,}"
    )
    assert t.stopped is False


@pytest.mark.parametrize("run_id", DOES_STOP_RUNS)
def test_converged_curves_do_stop(run_id):
    """The three from-scratch *_vel_s1 rungs ran a full ~32M-step budget well
    past convergence. WindowedTrendTracker at window=4/patience=3/min_delta=0.02
    must fire on all three.
    """
    pts = _CURVES[run_id]
    t, stop = _replay(pts, window=4, patience=3, min_delta=0.02, min_steps=0)
    assert stop is not None, (
        f"{run_id} was expected to stop (it ran a full budget well past "
        f"convergence) but WindowedTrendTracker never fired across "
        f"{len(pts)} evals up to step {pts[-1][0]:,}"
    )
    assert t.stopped is True


def test_signal_reports_the_true_peak_not_a_window_mean():
    """For every curve that does stop, the ConvergedSignal's best_reward must
    equal the curve's true max eval, not a window average -- a window mean
    would understate what the rung actually reached.
    """
    for run_id in DOES_STOP_RUNS:
        pts = _CURVES[run_id]
        t, stop = _replay(pts, window=4, patience=3, min_delta=0.02, min_steps=0)
        assert stop is not None, f"{run_id} unexpectedly never stopped"
        # true max among evals observed up to and including the stop step
        seen = [(s, r) for s, r in pts if s <= stop]
        true_max_step, true_max_reward = max(seen, key=lambda sr: sr[1])
        sig = t.signal(stop)
        assert sig.best_reward == pytest.approx(true_max_reward), (
            f"{run_id}: signal best_reward {sig.best_reward} != true observed "
            f"peak {true_max_reward} (at step {true_max_step})"
        )
        assert sig.best_step == true_max_step
