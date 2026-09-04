"""Guard gen_satsweep.py against regressions. Run with:

cd /Users/matthiasweiss/Claude/Projects/worktrees/sweep_env_envelope && \
/Users/matthiasweiss/miniconda3/envs/mujoco/bin/python -m pytest -p no:cacheprovider -q \
batch_runs/sweeps/test_gen_sweep_env_envelope.py batch_runs/sweeps/test_gen_satsweep.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gen_satsweep
import gen_rewardgate
import gen_snappy2


def test_fingerprint_default_is_unchanged():
    result = gen_satsweep.init_pose_library_fingerprint()
    assert result == ("mid", 60, "993d8e56cf2e")


def test_fingerprint_for_an_explicit_light_level():
    result = gen_satsweep.init_pose_library_fingerprint(level="light")
    assert result == ("light", 30, "970cf46b1ac8")


def test_fingerprint_for_an_explicit_mid_level_equals_the_default():
    explicit = gen_satsweep.init_pose_library_fingerprint(level="mid")
    default = gen_satsweep.init_pose_library_fingerprint()
    assert explicit == default


def test_fingerprint_none_and_unknown_levels():
    none_result = gen_satsweep.init_pose_library_fingerprint(level="none")
    assert none_result == ("none", None, None)

    unknown_result = gen_satsweep.init_pose_library_fingerprint(level="nosuchlevel")
    assert unknown_result == ("nosuchlevel", None, None)


def test_tier_a_regenerates_byte_identical():
    lines, _ = gen_satsweep.build_lines("A", [0])
    jsonl_path = os.path.join(_HERE, "UR3Pick_satsweep_tierA.jsonl")
    expected = open(jsonl_path, encoding="utf-8").read()
    actual = "\n".join(lines) + "\n"
    assert actual == expected


def test_rewardgate_regenerates_byte_identical():
    lines, _ = gen_rewardgate.build_lines([0, 1])
    jsonl_path = os.path.join(_HERE, "UR3Pick_rewardgate.jsonl")
    expected = open(jsonl_path, encoding="utf-8").read()
    actual = "\n".join(lines) + "\n"
    assert actual == expected


def test_snappy2_regenerates_byte_identical():
    lines, _ = gen_snappy2.build_lines([0, 1])
    jsonl_path = os.path.join(_HERE, "UR3Pick_snappy2.jsonl")
    expected = open(jsonl_path, encoding="utf-8").read()
    actual = "\n".join(lines) + "\n"
    assert actual == expected
