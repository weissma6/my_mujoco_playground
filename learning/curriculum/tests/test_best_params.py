"""Tests for the best-checkpoint pairing fix.

Pure Python. The headline test replays brax's REAL callback order and shows the
old logic picking the wrong params -- without that, "we fixed an off-by-one" is
an unverifiable claim.

Run:
    python -m pytest learning/curriculum/tests/test_best_params.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from learning.curriculum.best_params import BestParamsRecorder  # noqa: E402


def brax_callback_order(n_evals):
    """The exact order brax 0.13.0 fires the two callbacks.

    ppo/train.py:689 progress(0) -> :697 policy_params(0)
    then per iteration :727 policy_params(s_k) -> :748 progress(s_k)
    """
    yield ("prg", 0)
    yield ("ppf", 0)
    for k in range(1, n_evals):
        yield ("ppf", k)
        yield ("prg", k)


def old_logic(rewards):
    """The pre-fix implementation, verbatim in behaviour.

    `prog_state["last_eval_reward"]` is read inside policy_params_fn.
    """
    prog_state = {}
    best = {"reward": float("-inf"), "step": -1, "params": None}
    for kind, k in brax_callback_order(len(rewards)):
        if kind == "prg":
            prog_state["last_eval_reward"] = rewards[k]
        else:
            r = prog_state.get("last_eval_reward")
            if r is not None and r > best["reward"]:
                best = {"reward": float(r), "step": k, "params": f"p{k}"}
    return best


def new_logic(rewards):
    rec = BestParamsRecorder()
    for kind, k in brax_callback_order(len(rewards)):
        if kind == "prg":
            rec.on_progress(k, rewards[k])
        else:
            rec.on_policy_params(k, f"p{k}")
    return rec


# --- the headline: the bug was real, and it is gone -------------------------

def test_old_logic_picks_the_params_one_eval_past_the_peak():
    rewards = [10.0, 40.0, 90.0, 250.0, 120.0, 60.0, 30.0]   # peak at index 3
    old = old_logic(rewards)
    assert old["reward"] == 250.0            # it found the right NUMBER...
    assert old["step"] == 4                  # ...but attached the WRONG params
    assert old["params"] == "p4"


def test_new_logic_picks_the_params_that_earned_the_peak():
    rewards = [10.0, 40.0, 90.0, 250.0, 120.0, 60.0, 30.0]
    rec = new_logic(rewards)
    assert rec.best["reward"] == 250.0
    assert rec.best["step"] == 3
    assert rec.best["params"] == "p3"


def test_the_two_disagree_which_is_the_whole_point():
    rewards = [10.0, 40.0, 90.0, 250.0, 120.0, 60.0, 30.0]
    assert old_logic(rewards)["params"] != new_logic(rewards).best["params"]


@pytest.mark.parametrize("peak", [1, 2, 3, 5, 8])
def test_new_logic_is_right_wherever_the_peak_sits(peak):
    rewards = [float(10 * i) for i in range(10)]
    rewards[peak] = 10_000.0
    rec = new_logic(rewards)
    assert rec.best["step"] == peak
    assert rec.best["params"] == f"p{peak}"


def test_step_zero_is_where_the_old_logic_accidentally_agreed():
    """Explains why the bug was easy to miss: at step 0 progress fires FIRST."""
    rewards = [999.0, 1.0, 2.0]
    assert old_logic(rewards)["step"] == 0
    assert new_logic(rewards).best["step"] == 0


# --- order independence -----------------------------------------------------

def test_pairing_is_correct_under_the_reversed_callback_order():
    """Matching on step, not arrival, so a brax ordering change cannot break it."""
    rewards = [10.0, 300.0, 20.0]
    rec = BestParamsRecorder()
    for k, r in enumerate(rewards):          # progress BEFORE params, every time
        rec.on_progress(k, r)
        rec.on_policy_params(k, f"p{k}")
    assert rec.best["step"] == 1 and rec.best["params"] == "p1"


def test_everything_pairs_up_leaving_nothing_pending():
    rec = new_logic([1.0, 2.0, 3.0, 4.0])
    assert rec.unpaired() == (0, 0)          # no leaked params references


def test_latest_tracks_the_most_recent_params_for_early_stop():
    rec = new_logic([5.0, 4.0, 3.0])
    assert rec.latest["step"] == 2 and rec.latest["params"] == "p2"


# --- robustness -------------------------------------------------------------

def test_a_reward_with_no_params_never_becomes_best():
    rec = BestParamsRecorder()
    rec.on_progress(7, 10_000.0)
    assert rec.best["params"] is None and rec.best["step"] == -1


def test_missing_and_unparseable_rewards_are_ignored():
    rec = BestParamsRecorder()
    rec.on_policy_params(0, "p0")
    rec.on_progress(0, None)
    rec.on_progress(0, "not-a-number")
    assert rec.best["params"] is None


def test_negative_rewards_still_select_a_best():
    rec = new_logic([-500.0, -100.0, -900.0])
    assert rec.best["step"] == 1 and rec.best["reward"] == -100.0


# --- the disk snapshot is unchanged behaviour -------------------------------

def test_snapshots_are_written_once_per_eval_with_the_original_filename(tmp_path):
    written = []

    def ser(p):
        written.append(p)
        return f"BYTES:{p}".encode()

    rec = BestParamsRecorder(ckpt_dir=str(tmp_path), serialize=ser)
    for k in (0, 1_200_000, 24_000_000):
        rec.on_policy_params(k, f"p{k}")

    names = sorted(os.listdir(tmp_path))
    assert names == [
        "params_step000000000.msgpack",
        "params_step001200000.msgpack",
        "params_step024000000.msgpack",
    ]
    assert (tmp_path / "params_step001200000.msgpack").read_bytes() == b"BYTES:p1200000"
    assert rec.n_snapshots == 3
    assert written == ["p0", "p1200000", "p24000000"]


def test_a_failing_snapshot_never_kills_the_run(tmp_path):
    errors = []

    def boom(_):
        raise IOError("disk full")

    rec = BestParamsRecorder(ckpt_dir=str(tmp_path), serialize=boom,
                             on_error=lambda s, e: errors.append((s, e)))
    rec.on_policy_params(5, "p5")
    rec.on_progress(5, 100.0)
    assert errors and errors[0][0] == 5
    assert rec.best["step"] == 5             # tracking survives a write failure


def test_best_info_shape_matches_what_best_info_json_publishes():
    rec = new_logic([1.0, 9.0, 2.0])
    assert rec.best_info() == {"best_eval_reward": 9.0, "best_step": 1}
