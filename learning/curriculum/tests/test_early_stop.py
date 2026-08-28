"""Tests for the curriculum early-stop tracker.

Pure Python -- no jax, no brax, no MJX. Runs on the MacBook.

Run:
    python -m pytest batch_runs/curriculum/tests/test_early_stop.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from learning.curriculum.early_stop import ConvergedSignal, PatienceTracker  # noqa: E402


EVAL = "eval/episode_reward"


def feed(tracker, rewards, step0=0, dstep=1_000_000):
    """Push a reward sequence through the tracker; return the step it stopped at."""
    for i, r in enumerate(rewards):
        step = step0 + i * dstep
        if tracker.update(step, {EVAL: r}):
            return step
    return None


# --- 1. a curve that keeps improving must never stop ------------------------

def test_monotonically_rising_never_stops():
    t = PatienceTracker(patience=3, min_delta=0.02, min_steps=0)
    rewards = [100.0 * (1.5 ** i) for i in range(40)]
    assert feed(t, rewards) is None
    assert t.stopped is False
    assert t.strikes == 0


# --- 2. patience boundary ---------------------------------------------------

def test_flat_curve_stops_at_exactly_patience_non_improvements():
    patience = 5
    # 1 seeding eval + `patience` non-improvements -> stop on the last one.
    t = PatienceTracker(patience=patience, min_delta=0.02, min_steps=0)
    assert feed(t, [100.0] * (1 + patience)) is not None
    assert t.stopped is True
    assert t.strikes == patience


def test_one_short_of_patience_does_not_stop():
    patience = 5
    t = PatienceTracker(patience=patience, min_delta=0.02, min_steps=0)
    assert feed(t, [100.0] * patience) is None      # 1 seeding + patience-1 strikes
    assert t.stopped is False
    assert t.strikes == patience - 1


# --- 3. the minimum-step floor ---------------------------------------------

def test_never_stops_before_min_steps_even_on_a_dead_flat_curve():
    t = PatienceTracker(patience=2, min_delta=0.02, min_steps=6_000_000)
    # 20 identical evals, all below the floor.
    stop = feed(t, [100.0] * 20, step0=0, dstep=100_000)   # max step 1.9M < 6M
    assert stop is None
    assert t.stopped is False
    assert max(s for s, _ in t.history) < t.min_steps


def test_staleness_does_not_accrue_below_the_floor():
    """The floor must defer the EVIDENCE, not just the verdict.

    UR3Pick's eval curve is flat for its first several million steps -- the
    policy only learns reach/grasp by ~5M. If non-improvements counted below
    the floor, patience would already be spent when the floor was crossed and
    every rung would die at exactly min_steps. Regression guard for that.
    """
    t = PatienceTracker(patience=2, min_delta=0.02, min_steps=6_000_000)
    feed(t, [100.0] * 20, step0=0, dstep=100_000)          # all below the floor
    assert t.strikes == 0, "strikes must not accrue below min_steps"

    # opting in restores the old behaviour, so the gate is a real switch
    t2 = PatienceTracker(patience=2, min_delta=0.02, min_steps=6_000_000,
                         count_stale_before_floor=True)
    feed(t2, [100.0] * 20, step0=0, dstep=100_000)
    assert t2.strikes >= t2.patience


def test_stop_is_deferred_by_a_full_patience_window_past_the_floor():
    patience, dstep, floor = 2, 1_000_000, 6_000_000
    t = PatienceTracker(patience=patience, min_delta=0.02, min_steps=floor)
    stop = feed(t, [100.0] * 20, step0=0, dstep=dstep)
    # first at-or-past-floor eval is the 1st strike, so the stop lands
    # (patience-1) evals later -- NOT on the floor itself.
    assert stop == floor + (patience - 1) * dstep
    assert stop > floor
    assert t.stopped is True


def test_production_cadence_guarantees_a_real_training_budget():
    """At the planned num_evals=30 over 24M, a fully flat rung must still get
    well past the floor before it is allowed to stop.

    Worked through: brax runs num_evals-1 = 29 iterations after init, so one
    eval is 24M/29 = 827,586 steps. The floor at 6M falls between evals 7
    (5.79M) and 8 (6.62M), so eval 8 takes the first strike and the fifth lands
    at eval 12 = 9,931,032 steps -- 41% of the budget, not the 25% the floor
    alone would allow. The floor does NOT coincide with an eval boundary, which
    is why this is 9.93M and not 6M + 5*827,586 = 10.14M.
    """
    per_eval = 24_000_000 // 29
    floor, patience = 6_000_000, 5
    t = PatienceTracker(patience=patience, min_delta=0.02, min_steps=floor)
    stop = feed(t, [100.0] * 30, step0=0, dstep=per_eval)
    assert stop is not None

    first_at_floor = -(-floor // per_eval) * per_eval      # ceil to the eval grid
    assert stop == first_at_floor + (patience - 1) * per_eval
    assert stop == 9_931_032
    assert stop > 1.5 * floor, f"stopped too close to the floor at {stop:,}"


# --- 4. noise tolerance -----------------------------------------------------

def test_single_dip_between_improvements_does_not_stop():
    t = PatienceTracker(patience=3, min_delta=0.02, min_steps=0)
    # rise, dip, rise, dip, rise -- never 3 strikes in a row
    assert feed(t, [100.0, 200.0, 150.0, 300.0, 250.0, 400.0]) is None
    assert t.stopped is False


# --- 5. the negative-reward regime -----------------------------------------

def test_improvement_test_does_not_invert_when_reward_is_negative():
    t = PatienceTracker(patience=10, min_delta=0.02, min_steps=0)
    t.update(0, {EVAL: -100.0})
    # 2% of magnitude above -100 is -98.
    assert t.improvement_threshold() == pytest.approx(-98.0)

    # -99 is genuinely better than -100 but by only 1% -> not enough, a strike.
    assert t.update(1, {EVAL: -99.0}) is False
    assert t.strikes == 1
    assert t.best_reward == -100.0          # the bar did not move

    # -97 clears the 2% bar -> improvement, patience resets.
    t.update(2, {EVAL: -97.0})
    assert t.strikes == 0
    assert t.best_reward == -97.0


def test_naive_multiplicative_threshold_would_have_been_wrong():
    """Guards the choice of `best + abs(best)*d` over `best * (1 + d)`.

    With best = -100 and d = 0.02 the naive form gives -102, i.e. it would
    accept a *worse* reward as an improvement and never stop. Pin that we do
    not do that.
    """
    t = PatienceTracker(patience=10, min_delta=0.02, min_steps=0)
    t.update(0, {EVAL: -100.0})
    naive = -100.0 * (1 + 0.02)
    assert naive == pytest.approx(-102.0)
    assert t.improvement_threshold() > naive
    # a reward of -101 is worse than best; it must be a strike, not an improvement
    assert t.update(1, {EVAL: -101.0}) is False
    assert t.strikes == 1


# --- 6. the signal reports the peak, not the last eval ----------------------

def test_converged_signal_carries_the_best_not_the_last():
    t = PatienceTracker(patience=2, min_delta=0.02, min_steps=0)
    stop = feed(t, [10.0, 100.0, 50.0, 50.0], dstep=1)
    assert stop == 3

    sig = t.signal(stop)
    assert isinstance(sig, ConvergedSignal)
    assert sig.step == 3                 # where we stopped
    assert sig.best_reward == 100.0      # the peak, NOT the trailing 50.0
    assert sig.best_step == 1
    assert "100" in str(sig)


# --- extras: the bar vs the true max ---------------------------------------

def test_sub_min_delta_creep_still_records_a_true_max():
    """A rise too small to reset patience must still be reported as the max.

    Otherwise the rung hands forward a peak it did not actually reach.
    """
    t = PatienceTracker(patience=10, min_delta=0.10, min_steps=0)
    t.update(0, {EVAL: 100.0})
    t.update(1, {EVAL: 105.0})           # +5%, below the 10% bar
    assert t.strikes == 1                # not an improvement for patience
    assert t.best_reward == 100.0        # bar unmoved
    assert t.max_reward == 105.0         # but the max is honest
    assert t.max_step == 1


def test_metric_absent_is_ignored_not_counted_as_a_strike():
    t = PatienceTracker(patience=2, min_delta=0.02, min_steps=0)
    t.update(0, {EVAL: 100.0})
    assert t.update(1, {"training/walltime": 12.0}) is False
    assert t.strikes == 0
    assert len(t.history) == 1


def test_constructor_rejects_nonsense():
    with pytest.raises(ValueError):
        PatienceTracker(patience=0)
    with pytest.raises(ValueError):
        PatienceTracker(min_delta=-0.1)
    with pytest.raises(ValueError):
        PatienceTracker(min_steps=-1)


def test_summary_is_json_safe_and_reports_the_peak():
    import json
    t = PatienceTracker(patience=2, min_delta=0.02, min_steps=0)
    feed(t, [10.0, 100.0, 50.0, 50.0], dstep=1)
    s = t.summary()
    json.dumps(s)                        # must not raise
    assert s["stopped"] is True
    assert s["best_reward"] == 100.0
    assert s["best_step"] == 1
    assert s["n_evals"] == 4
