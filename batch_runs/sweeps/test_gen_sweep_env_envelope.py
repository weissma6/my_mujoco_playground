"""Guard gen_sweep_env_envelope.py against regressions. Run with:

cd /Users/matthiasweiss/Claude/Projects/worktrees/sweep_env_envelope && \
/Users/matthiasweiss/miniconda3/envs/mujoco/bin/python -m pytest -p no:cacheprovider -q \
batch_runs/sweeps/test_gen_sweep_env_envelope.py batch_runs/sweeps/test_gen_satsweep.py
"""

import os
import sys
import json
import ast
import subprocess
import importlib
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gen_satsweep
import gen_snappy2
import gen_rewardgate
import gen_dr_ladder

JSONL = os.path.join(_HERE, "UR3Pick_sweep_env_envelope.jsonl")
SNAPPY2 = os.path.join(_HERE, "UR3Pick_snappy2.jsonl")
GEN = os.path.join(_HERE, "gen_sweep_env_envelope.py")
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
RUN_EXPERIMENT = os.path.join(REPO, "learning", "notebooks", "run_experiment.py")


@pytest.fixture(scope="module")
def gen():
    return importlib.import_module("gen_sweep_env_envelope")


def data_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.startswith("#")]


def header_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.startswith("#")]


def test_build_lines_emits_eight_data_lines_at_the_ladder_budget(gen):
    lines, total = gen.build_lines([0, 1])
    assert len(lines) - 1 == 8
    assert total == 30_474_240


def test_run_ids_are_the_eight_expected(gen):
    lines, _ = gen.build_lines([0, 1])
    data = [json.loads(l) for l in lines[1:]]
    run_ids = [row["run_id"] for row in data]
    expected_set = {f"SweepC_{lv}_{xy}_s{s}"
                    for lv in ("light", "mid")
                    for xy in ("narrow", "wide")
                    for s in (0, 1)}
    assert set(run_ids) == expected_set
    expected_order = [
        "SweepC_light_narrow_s0", "SweepC_light_narrow_s1",
        "SweepC_light_wide_s0", "SweepC_light_wide_s1",
        "SweepC_mid_narrow_s0", "SweepC_mid_narrow_s1",
        "SweepC_mid_wide_s0", "SweepC_mid_wide_s1",
    ]
    assert run_ids == expected_order


def test_committed_jsonl_regenerates_byte_identical(gen):
    lines, _ = gen.build_lines([0, 1])
    expected = open(JSONL, encoding="utf-8").read()
    actual = "\n".join(lines) + "\n"
    assert actual == expected


def test_check_lines_accepts_the_committed_file(gen):
    rows = gen.check_lines(JSONL, 8, [0, 1])
    assert len(rows) == 8


def test_each_cell_matches_its_run_id_and_tags(gen):
    for row in data_rows(JSONL):
        _, lv, xy, s = row["run_id"].split("_")
        assert row["init_start_random"] == lv
        if xy == "narrow":
            assert row["box_xy_jitter"] == [0.085, 0.12]
        else:
            assert row["box_xy_jitter"] == [0.17, 0.24]
        assert row["seed"] == int(s[1:])
        assert row["wandb_tags"] == ["sweepC", f"pose{lv}", f"xy{xy}", s, "anchora1b0c1d1"]


def test_base_is_the_defence_line_from_the_snappy2_jsonl(gen):
    sweepc_rows = {r["run_id"]: r for r in data_rows(JSONL)}
    snappy_rows = {r["run_id"]: r for r in data_rows(SNAPPY2)}

    for seed in (0, 1):
        sweepc = sweepc_rows[f"SweepC_mid_wide_s{seed}"]
        snappy = snappy_rows[f"Snappy2_as04_ar70_g01_s{seed}"]

        A = {k: v for k, v in sweepc.items()
             if k not in {"run_id", "wandb_tags", "num_timesteps", "num_evals",
                          "num_eval_envs", "lifter_tilt_max", "init_start_random",
                          "box_xy_jitter"}}
        B = {k: v for k, v in snappy.items()
             if k not in {"run_id", "wandb_tags", "num_timesteps", "num_evals"}}
        assert A == B


def test_pinned_values_on_every_line(gen):
    for row in data_rows(JSONL):
        assert row["lifter_tilt_max"] == 0.12 == gen_dr_ladder.LIFTER_TILT_MAX
        assert row["num_timesteps"] == 30_000_000
        assert row["num_evals"] == 32
        assert row["num_eval_envs"] == 256
        assert row["num_resets_per_eval"] == 1
        assert row["episode_length"] == 400
        assert row["box_z_rot_range"] == 6.283185307179586
        assert row["obs_include_velocity"] is True
        assert row["action_scale"] == 0.04
        assert row["action_rate"] == -0.7
        assert row["gripper_action_scale"] == 0.02
        assert row["gate_gripper_box_on_lift"] is False
        assert row["gate_gripper_align_on_lift"] is True
        assert row["entropy_cost"] == 0.02
        assert row["learning_rate"] == 0.0003
        assert row["reward_scaling"] == 0.03
        assert row["network_factory"] == {
            "policy_hidden_layer_sizes": [256, 256, 256],
            "value_hidden_layer_sizes": [256, 256, 256, 256, 256]
        }
        assert row["domain_rand.enable"] is True
        assert row["domain_rand.cube_mass.enable"] is True
        assert row["domain_rand.cube_mass.min"] == 0.85
        assert row["domain_rand.cube_mass.max"] == 1.15
        for k in ("lifter_height_abs_min", "lifter_height_abs_max", "init_qpos_noise"):
            assert k not in row


def test_lines_identical_outside_the_swept_keys(gen):
    lines, _ = gen.build_lines([0, 1])
    rows = [json.loads(l) for l in lines[1:]]
    varying = {"seed", "run_id", "wandb_tags", "init_start_random", "box_xy_jitter"}
    base = {k: v for k, v in rows[0].items() if k not in varying}
    for row in rows[1:]:
        other = {k: v for k, v in row.items() if k not in varying}
        assert other == base


def test_header_stamps_both_pose_libraries_and_the_array_size(gen):
    hdr = "\n".join(header_lines(JSONL))
    assert "level='light' n_poses=30 sha256=970cf46b1ac8" in hdr
    assert "level='mid' n_poses=60 sha256=993d8e56cf2e" in hdr
    assert "ACTUAL TOTAL_STEPS=30474240" in hdr
    assert "--array=1-8%4" in hdr


def test_check_lines_rejects_a_tampered_action_scale(gen, tmp_path):
    rows = data_rows(JSONL)
    tampered = rows.copy()
    tampered[0]["action_scale"] = 0.03
    p = tmp_path / "t.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in tampered:
            f.write(json.dumps(row) + "\n")
    with pytest.raises(AssertionError):
        gen.check_lines(str(p), 8, [0, 1])


def test_check_lines_rejects_a_duplicated_run_id(gen, tmp_path):
    rows = data_rows(JSONL)
    tampered = rows.copy()
    tampered[-1]["run_id"] = tampered[0]["run_id"]
    p = tmp_path / "t.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in tampered:
            f.write(json.dumps(row) + "\n")
    with pytest.raises(AssertionError):
        gen.check_lines(str(p), 8, [0, 1])


def test_check_lines_rejects_a_missing_swept_key(gen, tmp_path):
    rows = data_rows(JSONL)
    tampered = rows.copy()
    del tampered[0]["box_xy_jitter"]
    p = tmp_path / "t.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in tampered:
            f.write(json.dumps(row) + "\n")
    with pytest.raises((AssertionError, KeyError)):
        gen.check_lines(str(p), 8, [0, 1])


def test_narrow_spawn_corners_are_inside_the_reachable_annulus(gen):
    jx, jy = gen.XY_JITTER["narrow"]
    cx = gen_dr_ladder._BOX_CENTER_X
    for x, y in [(cx - jx, jy), (cx - jx, -jy), (cx + jx, jy), (cx + jx, -jy)]:
        r = (x * x + y * y) ** 0.5
        assert gen_dr_ladder._BOX_XY_R_MIN <= r <= gen_dr_ladder._UR3_WORKING_RADIUS_M


def test_run_experiment_forwards_the_swept_env_keys():
    tree = ast.parse(open(RUN_EXPERIMENT, encoding="utf-8").read())
    run_exp = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "run_experiment")

    forwarded = set()
    for node in ast.walk(run_exp):
        if isinstance(node, ast.Assign):
            if (isinstance(node.targets[0], ast.Subscript) and
                isinstance(node.targets[0].value, ast.Name) and
                node.targets[0].value.id == "env_overrides" and
                isinstance(node.targets[0].slice, ast.Constant) and
                isinstance(node.targets[0].slice.value, str)):
                forwarded.add(node.targets[0].slice.value)

    assert {"init_start_random", "box_xy_jitter", "lifter_tilt_max"} <= forwarded


def test_extract_ppo_overrides_reserves_env_keys_and_forwards_num_eval_envs():
    tree = ast.parse(open(RUN_EXPERIMENT, encoding="utf-8").read())
    extract = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_extract_ppo_overrides")

    reserved = None
    for node in ast.walk(extract):
        if isinstance(node, ast.Assign):
            if (isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == "reserved" and
                isinstance(node.value, ast.Set)):
                reserved = {e.value for e in node.value.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)}
                break

    assert reserved is not None
    assert {"init_start_random", "box_xy_jitter", "lifter_tilt_max"} <= reserved
    assert "num_eval_envs" not in reserved


def test_cli_refuses_to_overwrite_without_force(gen, tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text("keep\n")
    r = subprocess.run(
        [sys.executable, GEN, "--out", str(p)],
        capture_output=True, text=True
    )
    assert r.returncode != 0
    assert p.read_text() == "keep\n"

    r2 = subprocess.run(
        [sys.executable, GEN, "--out", str(p), "--force"],
        capture_output=True, text=True
    )
    assert r2.returncode == 0
    assert p.read_text() == open(JSONL, encoding="utf-8").read()
