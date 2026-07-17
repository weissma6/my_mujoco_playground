"""Local, robot-free smoke test for _GripperWorker (Commit 1, 2026-07-17).

Proves the two properties the gripper-thread decoupling exists for, without
touching RTDE, XML-RPC, or the physical robot:

  1. command() / latest_state() never block, even while the worker's own
     XML-RPC call is stalled (simulated with a slow fake gripper_fn).
  2. The worker still rate-limits to <=~10 Hz and only re-sends on a
     meaningful (>1e-3) change, matching the pre-Commit-1 semantics.

This is NOT the Commit-1 acceptance test -- that requires the real robot
(confirm overrun_count ~= 0 and loop_hz_true ~= 50 on a real run, per
"Plan - Sim-to-Real Gap Protocol"). This only proves the threading design
itself is sound before it ever reaches hardware.

Run: .venv/bin/python robots/UR3e/test_gripper_worker.py
"""

import sys
import time

sys.path.insert(0, ".")
from robots.UR3e.ur3_realrobot_dependencies import _GripperWorker  # noqa: E402

FAKE_XMLRPC_STALL_S = 0.15  # worse than the measured 116 ms mean / 202 ms max


def make_fake_gripper():
    calls = []

    def gripper_fn(norm_cmd):
        time.sleep(FAKE_XMLRPC_STALL_S)  # simulate the blocking HTTP round-trip
        calls.append(norm_cmd)

    def gripper_state_fn():
        time.sleep(FAKE_XMLRPC_STALL_S / 2)  # simulate getCurrentPosition()
        return {
            "sim_finger": 0.5 * (calls[-1] if calls else 0.0) * 0.025,
            "pos_pct": 50.0,
            "grasped": bool(calls) and calls[-1] > 0.5,
            "obj_flag": 1 if (calls and calls[-1] > 0.5) else 0,
        }

    return gripper_fn, gripper_state_fn, calls


def test_command_never_blocks():
    gripper_fn, gripper_state_fn, calls = make_fake_gripper()
    worker = _GripperWorker(gripper_fn, gripper_state_fn, gripper_min_dt=0.1,
                             debug_print=False)
    worker.start()
    try:
        # Force at least one in-flight XML-RPC call before timing.
        worker.command(0.1)
        time.sleep(0.05)

        t0 = time.perf_counter()
        for i in range(50):
            worker.command(0.1 + 0.001 * i)  # tiny deltas, mostly under the 1e-3 gate
            worker.latest_state()
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.05, (
            f"command()/latest_state() took {elapsed*1000:.1f} ms for 50 calls "
            f"-- should be sub-millisecond; the arm loop must never block on "
            f"the gripper channel."
        )
        print(f"[PASS] 50x command()+latest_state() took {elapsed*1000:.3f} ms "
              f"(non-blocking, even with a {FAKE_XMLRPC_STALL_S*1000:.0f} ms "
              f"fake XML-RPC stall in flight)")
    finally:
        worker.stop()


def test_rate_limit_and_change_gate():
    gripper_fn, gripper_state_fn, calls = make_fake_gripper()
    worker = _GripperWorker(gripper_fn, gripper_state_fn, gripper_min_dt=0.1,
                             debug_print=False)
    worker.start()
    try:
        worker.command(0.5)
        time.sleep(0.6)  # ~6 worker ticks at 0.1s pacing, but value never changes
        n_after_static = len(calls)
        assert n_after_static == 1, (
            f"expected exactly 1 send for an unchanged command, got "
            f"{n_after_static} -- the >1e-3 change gate is not holding."
        )

        worker.command(0.9)  # a real change
        time.sleep(0.5)  # >= one full round-trip (0.15s call + 0.075s readback)
        n_after_change = len(calls)
        assert n_after_change == 2, (
            f"expected exactly 1 additional send after a real change, got "
            f"{n_after_change - n_after_static}."
        )
        print(f"[PASS] rate limit + change gate: {n_after_static} send while "
              f"static, {n_after_change - n_after_static} send on change "
              f"(fake channel stalls {FAKE_XMLRPC_STALL_S*1000:.0f} ms/call)")
    finally:
        worker.stop()


def test_state_carries_forward():
    gripper_fn, gripper_state_fn, calls = make_fake_gripper()
    worker = _GripperWorker(gripper_fn, gripper_state_fn, gripper_min_dt=0.1,
                             debug_print=False)
    worker.start()
    try:
        s0 = worker.latest_state()
        assert s0["grasped"] is False and s0["obj_flag"] == 0, (
            "initial state must default False/0, matching the pre-Commit-1 "
            "sentinels, not raise or block."
        )
        worker.command(0.9)
        time.sleep(0.5)  # >= one full round-trip (0.15s call + 0.075s readback)
        s1 = worker.latest_state()
        assert s1["grasped"] is True and s1["obj_flag"] == 1

        # No new command for a while -- latest_state() must keep returning
        # the same (carried-forward) values, not go stale/None.
        time.sleep(0.3)
        s2 = worker.latest_state()
        assert s2 == s1, "state should carry forward unchanged, not reset"
        print("[PASS] latest_state() carry-forward semantics match the "
              "pre-Commit-1 behaviour")
    finally:
        worker.stop()


def test_stop_is_clean_and_idempotent():
    gripper_fn, gripper_state_fn, _ = make_fake_gripper()
    worker = _GripperWorker(gripper_fn, gripper_state_fn, gripper_min_dt=0.1,
                             debug_print=False)
    worker.start()
    t0 = time.perf_counter()
    worker.stop()
    worker.stop()  # must not raise / hang on a second call (finally-block safety)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, f"stop() took {elapsed:.2f}s -- should join quickly"
    print(f"[PASS] stop() is clean and idempotent ({elapsed*1000:.0f} ms)")


if __name__ == "__main__":
    test_command_never_blocks()
    test_rate_limit_and_change_gate()
    test_state_carries_forward()
    test_stop_is_clean_and_idempotent()
    print("\nAll _GripperWorker smoke tests passed.")
    print("Reminder: this does NOT replace the on-robot acceptance check --")
    print("confirm overrun_count ~= 0 and loop_hz_true ~= 50 on a real run.")
