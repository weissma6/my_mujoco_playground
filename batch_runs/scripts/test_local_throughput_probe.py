"""Tests for the guarded CPU-MJX local throughput probe.

The module under test does not exist yet (batch_runs/scripts/local_throughput_probe.py).
These tests are expected to be red today: the module-loading fixture raises
FileNotFoundError/ModuleNotFoundError, and the subprocess-based tests fail their
needle-in-stderr assertions because Python's own "can't open file" also happens
to exit 2, without the expected message.

Machine: MacBook (Darwin). No test here imports jax, brax, mujoco.mjx or
mujoco_playground, and none builds an env. Subprocess-spawned children are the
module under test itself (run as a script) or plain `python -c` snippets that
only touch stdlib.
"""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "batch_runs" / "scripts" / "local_throughput_probe.py"
DEFENCE_JSONL = REPO / "batch_runs" / "sweeps" / "UR3Pick_snappy2.jsonl"
DEFENCE_RUN_ID = "Snappy2_as04_ar70_g01_s1"


@pytest.fixture(scope="module")
def probe():
    """Load local_throughput_probe.py by file path (no package __init__.py).

    Expected to raise today (module file does not exist yet) -- that failure
    is the intended red state for every test that depends on this fixture.
    """
    spec = importlib.util.spec_from_file_location("local_throughput_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_without_slurm():
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    return env


# ---------------------------------------------------------------------------
# 1. extrapolate
# ---------------------------------------------------------------------------


def test_extrapolate(probe):
    assert probe.extrapolate(1000.0, 24_903_680) == pytest.approx(24903.68)
    with pytest.raises(ValueError):
        probe.extrapolate(0.0, 10)
    with pytest.raises(ValueError):
        probe.extrapolate(10.0, 0)


# ---------------------------------------------------------------------------
# 2. total_steps reproduces the campaign budgets
# ---------------------------------------------------------------------------


def test_total_steps_reproduces_the_campaign_budgets(probe):
    defence = probe.total_steps(24_000_000, 20, 512, 10, 32)
    assert defence == 24_903_680
    assert defence == probe.DEFENCE_TOTAL_STEPS

    rung = probe.total_steps(30_000_000, 32, 512, 10, 32)
    assert rung == 30_474_240
    assert rung == probe.RUNG_TOTAL_STEPS


# ---------------------------------------------------------------------------
# 3. ppo_stage_kwargs is the brief
# ---------------------------------------------------------------------------


def test_ppo_stage_kwargs_is_the_brief(probe):
    kw = probe.ppo_stage_kwargs()
    assert kw["num_envs"] == 32
    assert kw["batch_size"] == 32
    assert kw["num_minibatches"] == 4
    assert kw["unroll_length"] == 10
    assert kw["num_updates_per_batch"] == 1
    assert kw["num_evals"] == 0
    assert kw["run_evals"] is False
    assert kw["max_devices_per_host"] == 1
    assert kw["num_timesteps"] == 3 * 32 * 10 * 4
    assert (kw["batch_size"] * kw["num_minibatches"]) % kw["num_envs"] == 0
    assert kw["num_envs"] <= probe.NUM_ENVS_MAX


# ---------------------------------------------------------------------------
# 4. format_report: ok record and watchdog record
# ---------------------------------------------------------------------------


def test_format_report_ok_and_watchdog(probe):
    ok_record = {
        "stage": 3,
        "num_envs": 32,
        "steps": 200,
        "outcome": "ok",
        "env_steps_per_s": 1000.0,
        "peak_rss_gb": 2.5,
        "compile_s": 40.0,
        "run_s": 6.4,
        "rss_cap_gb": 8.0,
    }
    text = probe.format_report(ok_record)
    assert "Stage 3" in text
    assert "32" in text
    assert "1000" in text
    assert "2.5" in text
    assert "40" in text
    for key in probe.TARGETS:
        assert key in text
    assert "6.9" in text
    assert "8.5" in text
    assert "extrapolat" in text
    assert "per-env" in text

    watchdog_record = {
        "stage": 2,
        "num_envs": 8,
        "outcome": "watchdog",
        "peak_rss_gb": 8.1,
        "rss_cap_gb": 8.0,
    }
    wtext = probe.format_report(watchdog_record)
    assert "watchdog" in wtext
    assert "8.1" in wtext
    assert "n/a" in wtext
    assert " h (" not in wtext


# ---------------------------------------------------------------------------
# 5. load_defence_cfg
# ---------------------------------------------------------------------------


def test_load_defence_cfg(probe):
    cfg = probe.load_defence_cfg(DEFENCE_JSONL, DEFENCE_RUN_ID)
    assert cfg["action_scale"] == 0.04
    assert cfg["action_rate"] == -0.7
    assert cfg["num_timesteps"] == 24000000
    assert cfg["network_factory"]["policy_hidden_layer_sizes"] == [256, 256, 256]

    with pytest.raises(KeyError):
        probe.load_defence_cfg(DEFENCE_JSONL, "does_not_exist")


# ---------------------------------------------------------------------------
# 6. rss_bytes
# ---------------------------------------------------------------------------


def test_rss_bytes(probe):
    assert probe.rss_bytes(os.getpid()) > 1_000_000
    assert probe.rss_bytes(2**22 - 1) == 0


# ---------------------------------------------------------------------------
# 7 & 8. RssWatchdog behaviour
# ---------------------------------------------------------------------------


def test_watchdog_kills_a_fat_child(probe):
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import time\n"
                "b = bytearray(400 * 1024 * 1024)\n"
                "b[::4096] = b'x' * len(b[::4096])\n"
                "time.sleep(60)\n"
            ),
        ]
    )
    try:
        wd = probe.RssWatchdog(p.pid, cap_bytes=64 * 1024 * 1024, interval_s=0.1)
        wd.start()
        p.wait(timeout=20)
        wd.stop()
        assert p.returncode == -signal.SIGKILL
        assert wd.tripped is True
        assert wd.peak_bytes > 64 * 1024 * 1024
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)


def test_watchdog_leaves_a_thin_child_alone(probe):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    try:
        wd = probe.RssWatchdog(p.pid, cap_bytes=2 * 1024**3, interval_s=0.1)
        wd.start()
        rc = p.wait(timeout=20)
        wd.stop()
        assert rc == 0
        assert wd.tripped is False
        assert 0 < wd.peak_bytes < 2 * 1024**3
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)


# ---------------------------------------------------------------------------
# 9. stage4_allowed
# ---------------------------------------------------------------------------


def test_stage4_allowed(probe, tmp_path):
    allowed, reason = probe.stage4_allowed(tmp_path)
    assert allowed is False
    assert isinstance(reason, str) and reason

    (tmp_path / "a.jsonl").write_text(
        json.dumps({"stage": 3, "outcome": "ok", "peak_rss_gb": 3.5}) + "\n"
    )
    allowed, reason = probe.stage4_allowed(tmp_path)
    assert allowed is True
    assert isinstance(reason, str) and reason

    # A later file wins: b.jsonl's stage-3 record exceeds the prior-peak cap.
    (tmp_path / "b.jsonl").write_text(
        json.dumps({"stage": 3, "outcome": "ok", "peak_rss_gb": 4.5}) + "\n"
    )
    allowed, reason = probe.stage4_allowed(tmp_path)
    assert allowed is False
    assert isinstance(reason, str) and reason

    watchdog_dir = tmp_path / "watchdog_only"
    watchdog_dir.mkdir()
    (watchdog_dir / "w.jsonl").write_text(
        json.dumps({"stage": 3, "outcome": "watchdog", "peak_rss_gb": 1.0}) + "\n"
    )
    allowed, reason = probe.stage4_allowed(watchdog_dir)
    assert allowed is False
    assert isinstance(reason, str) and reason

    stage2_dir = tmp_path / "stage2_only"
    stage2_dir.mkdir()
    (stage2_dir / "s2.jsonl").write_text(
        json.dumps({"stage": 2, "outcome": "ok", "peak_rss_gb": 1.0}) + "\n"
    )
    allowed, reason = probe.stage4_allowed(stage2_dir)
    assert allowed is False
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# 10. CLI refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,needs_slurm_env,needle",
    [
        (["--stage", "1"], False, "Never train or smoke-test locally"),
        (["--stage", "1", "--acknowledge-cpu"], True, "SLURM"),
        (["--stage", "1", "--acknowledge-cpu", "--rss-cap-gb", "13"], False, "cap"),
        (["--stage", "1", "--acknowledge-cpu", "--num-envs", "65"], False, "envs"),
        (["--stage", "1", "--acknowledge-cpu", "--timeout-s", "901"], False, "timeout"),
    ],
    ids=["no_ack", "slurm_present", "cap_too_high", "envs_too_high", "timeout_too_high"],
)
def test_cli_refusals(args, needs_slurm_env, needle):
    env = dict(os.environ)
    if needs_slurm_env:
        env["SLURM_JOB_ID"] = "1"
    else:
        env.pop("SLURM_JOB_ID", None)

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    elapsed = time.perf_counter() - start

    assert result.returncode == 2
    # Guard against a false pass: the interpreter path itself
    # (miniconda3/envs/mujoco/...) can coincidentally contain a needle
    # like "envs", so a Python "can't open file" error must not count.
    assert "can't open file" not in result.stderr
    assert needle in result.stderr
    assert elapsed < 15


# ---------------------------------------------------------------------------
# 11. Importing the module must not pull in jax/brax/mjx/mujoco_playground
# ---------------------------------------------------------------------------


def test_module_import_is_lazy():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('local_throughput_probe', {str(SCRIPT)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "banned = [m for m in ('jax', 'brax', 'mujoco.mjx', 'mujoco_playground') if m in sys.modules]\n"
        "assert not banned, banned\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 12. Stage 4 refuses without a qualifying stage-3 record (end to end, CLI)
# ---------------------------------------------------------------------------


def test_stage4_refuses_without_a_qualifying_s3_record(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--stage",
            "4",
            "--acknowledge-cpu",
            "--results-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_env_without_slurm(),
    )
    assert result.returncode == 2
    assert "stage 3" in result.stderr


# ---------------------------------------------------------------------------
# 13. stage4_allowed tolerates a trailing unparsable line in the winning file
# ---------------------------------------------------------------------------


def test_stage4_allowed_trailing_invalid_json_line(probe, tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"stage": 3, "outcome": "ok", "peak_rss_gb": 1.0}) + "\n"
        "{not valid json\n"
    )
    allowed, reason = probe.stage4_allowed(tmp_path)
    assert allowed is True
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# 14. format_report on an "ok" record missing env_steps_per_s must not crash
# (and must not fabricate an extrapolation from a zero/absent rate)
# ---------------------------------------------------------------------------


def test_format_report_ok_missing_env_steps_per_s(probe):
    record = {
        "stage": 3,
        "num_envs": 32,
        "steps": 200,
        "outcome": "ok",
        "peak_rss_gb": 2.5,
        "rss_cap_gb": 8.0,
    }
    text = probe.format_report(record)
    assert "n/a" in text
    for name in probe.TARGETS:
        assert f"extrapolation → {name}: n/a (ok)" in text


# ---------------------------------------------------------------------------
# 15. rss_bytes parses the whitespace-padded output `ps -o rss=` actually emits
# ---------------------------------------------------------------------------


def test_rss_bytes_parses_whitespace_padded_ps_output(probe, monkeypatch):
    class FakeResult:
        stdout = "   4096\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        assert cmd[:3] == ["ps", "-o", "rss="]
        return FakeResult()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    assert probe.rss_bytes(12345) == 4096 * 1024
