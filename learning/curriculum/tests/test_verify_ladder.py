"""Tests for verify_ladder's v3 pass criteria.

v3 trains every rung to a fixed budget instead of stopping early, so an
early stop is now a FAULT rather than a required success signal -- the
opposite of what today's check() asserts. This file exercises check()
directly with hand-built by_rung fixtures, matching the shape fetch()
produces ({rung_id: {"run": <obj with .state and .name>, "summary": {...}}})
without importing wandb anywhere.

Run:
    python -m pytest learning/curriculum/tests/test_verify_ladder.py -q
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from batch_runs.curriculum.verify_ladder import RUNG_ORDER, check  # noqa: E402

CAP = 30_000_000


class FakeRun:
    """Stands in for a wandb Run: check() only ever reads .state and .name."""

    def __init__(self, name, state="finished"):
        self.name = name
        self.state = state


def full_ladder():
    """Six chained rungs: cold L0, no stops, every rung at the step floor.

    training/num_steps is the field to floor-check: run_experiment.py:236-249
    logs it explicitly on every eval (`log_dict = {"training/num_steps": ...}`,
    then `wandb.log(log_dict, step=...)`), so it is a value this codebase
    deliberately writes, not wandb's own `_step` bookkeeping of the `step=`
    kwarg. Every rung is built at exactly the passing shape (RUNG_ORDER
    order, Curr_v5_<id>_s0 names, an unbroken sha256 chain, nothing stopped
    early, training/num_steps at the 30M floor) so each test below mutates
    exactly one field off that baseline and the resulting problem can be
    attributed to it.
    """
    by_rung = {}
    prev_pub = None
    for rung_id in RUNG_ORDER:
        init_sha = prev_pub if prev_pub is not None else "cold"
        pub_sha = f"pub_{rung_id}"
        by_rung[rung_id] = {
            "run": FakeRun(f"Curr_v5_{rung_id}_s0"),
            "summary": {
                "curriculum/params_sha256_at_init": init_sha,
                "curriculum/published_sha256": pub_sha,
                "curriculum/stopped_early": False,
                "training/num_steps": CAP,
            },
        }
        prev_pub = pub_sha
    return by_rung


# --- criterion 1: unchanged --------------------------------------------------

def test_missing_rung_is_a_hard_failure():
    by_rung = full_ladder()
    del by_rung["L2_pos_cube"]
    _, problems, _ = check(by_rung)
    assert any("L2_pos_cube" in p for p in problems)


# --- criterion 2: unchanged ---------------------------------------------------

def test_broken_chain_link_is_reported():
    by_rung = full_ladder()
    by_rung["L2_pos_cube"]["summary"]["curriculum/params_sha256_at_init"] = "wrong"
    _, problems, _ = check(by_rung)
    assert any("L2_pos_cube" in p for p in problems)


def test_intact_chain_produces_no_chain_problem():
    by_rung = full_ladder()
    _, problems, _ = check(by_rung)
    chain_problems = [p for p in problems if "predecessor" in p.lower()]
    assert chain_problems == []


# --- criterion 3: inverted ----------------------------------------------------

def test_a_rung_that_stopped_early_is_now_a_failure():
    by_rung = full_ladder()
    by_rung["L3_pos_cube_robot"]["summary"]["curriculum/stopped_early"] = True
    _, problems, _ = check(by_rung)
    assert any("L3_pos_cube_robot" in p for p in problems)


def test_early_stop_summary_stopped_flag_is_also_a_failure():
    by_rung = full_ladder()
    by_rung["L3_pos_cube_robot"]["summary"]["curriculum/early_stop_summary"] = {
        "stopped": True
    }
    _, problems, _ = check(by_rung)
    assert any("L3_pos_cube_robot" in p for p in problems)


def test_no_rung_stopping_early_is_not_a_problem():
    """The old behaviour inverted: v1/v2 complained when NOTHING stopped
    early. v3 trains every rung to the fixed budget, so a clean ladder with
    zero early stops is now the only correct outcome -- the old "no rung
    reports an early stop" complaint must be gone."""
    by_rung = full_ladder()
    _, problems, _ = check(by_rung)
    assert problems == []


def test_a_rung_below_the_step_floor_is_reported():
    by_rung = full_ladder()
    by_rung["L4_full"]["summary"]["training/num_steps"] = 29_999_999
    _, problems, _ = check(by_rung)
    assert any("L4_full" in p for p in problems)


def test_step_floor_boundary_at_exactly_30m_passes():
    by_rung = full_ladder()
    by_rung["L4_full"]["summary"]["training/num_steps"] = 30_000_000
    _, problems, _ = check(by_rung)
    assert problems == []


def test_step_floor_is_inclusive_at_the_true_quantized_cap():
    """The real cap overshoots to 30,474,240 -- the bound must be >=, never
    ==, or every real rung would fail this check."""
    by_rung = full_ladder()
    by_rung["L4_full"]["summary"]["training/num_steps"] = 30_474_240
    _, problems, _ = check(by_rung)
    assert problems == []


def test_step_floor_falls_back_to_underscore_step_when_num_steps_is_absent():
    """training/num_steps is primary; wandb's own `_step` mirror of the
    `step=` kwarg is the fallback, not an equal alternative -- see the
    module-level note on full_ladder() for why the two are not
    interchangeable in general."""
    by_rung = full_ladder()
    del by_rung["L4_full"]["summary"]["training/num_steps"]
    by_rung["L4_full"]["summary"]["_step"] = CAP
    _, problems, _ = check(by_rung)
    assert not any("L4_full" in p for p in problems)


def test_step_floor_is_a_failure_when_neither_key_is_present():
    """A rung whose step count cannot be established must never pass
    silently -- absence is not evidence the budget was met."""
    by_rung = full_ladder()
    del by_rung["L4_full"]["summary"]["training/num_steps"]
    _, problems, _ = check(by_rung)
    assert any("L4_full" in p for p in problems)


def test_a_run_not_named_curr_v5_is_reported():
    """A stale-spec run name (the pre-migration Curr_v4_ prefix) must be
    flagged just like any other wrong-prefix name -- the ladder spec moved
    from Curr_v4_ to Curr_v5_ and check() must not still accept the old
    prefix."""
    by_rung = full_ladder()
    by_rung["L1_pos"]["run"].name = "Curr_v4_L1_pos_s0"
    _, problems, _ = check(by_rung)
    assert any("L1_pos" in p for p in problems)


def test_fully_correct_v5_ladder_yields_empty_problems():
    by_rung = full_ladder()
    rows, problems, _ = check(by_rung)
    assert problems == []
    assert len(rows) == 6


# --- adversarial: reviewer-added, not part of the frozen WP1 set -------------
#
# These probe edge cases the frozen set above does not cover: a falsy-but-
# present step count, and non-int/malformed shapes for the two fields
# check() reads off raw W&B summaries (which are not schema-enforced, so any
# JSON-serialisable value can show up there in practice).

def test_step_floor_zero_is_present_not_absent_and_correctly_fails():
    """training/num_steps: 0 is falsy but PRESENT -- `steps is None` must
    stay False so it is compared against the floor (and fails), rather than
    falling through to the _step fallback and possibly passing on stale
    data."""
    by_rung = full_ladder()
    by_rung["L4_full"]["summary"]["training/num_steps"] = 0
    _, problems, _ = check(by_rung)
    assert any("L4_full" in p and "0" in p for p in problems)


def test_step_floor_string_value_does_not_crash_the_verifier():
    """DEFECT: W&B summaries are not schema-enforced -- training/num_steps
    logged as a numeric string (e.g. "30000000") is a realistic shape, not a
    contrived one. `elif steps < STEP_FLOOR` compares str < int, which
    raises TypeError instead of reporting a problem or coercing the value.
    A single malformed field in one run's summary should not take down the
    whole ladder verification."""
    by_rung = full_ladder()
    by_rung["L4_full"]["summary"]["training/num_steps"] = str(CAP)
    check(by_rung)  # must not raise
