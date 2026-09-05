import copy
import json
import pathlib
import re
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_sweep_anchor_budget as ga
import gen_satsweep as gs
import gen_rewardgate as rg

REPO_ROOT = HERE.parents[1]
JSONL = HERE / "UR3Pick_sweep_anchor_budget.jsonl"
SNAPPY2 = HERE / "UR3Pick_snappy2.jsonl"


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f
                if l.strip() and not l.lstrip().startswith("#")]


def read_header_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.lstrip().startswith("#")]


def _write_variant(path, headers, rows):
    with open(path, "w", encoding="utf-8") as f:
        for h in headers:
            f.write(h + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_six_data_lines_with_expected_ids():
    rows = read_rows(JSONL)
    assert len(rows) == 6

    run_ids = [r["run_id"] for r in rows]
    expected_ids = {
        "SweepA_anchor_b24M_s0", "SweepA_anchor_b24M_s1",
        "SweepA_anchor_b48M_s0", "SweepA_anchor_b48M_s1",
        "SweepA_anchor_b72M_s0", "SweepA_anchor_b72M_s1"
    }
    assert set(run_ids) == expected_ids
    assert run_ids == [
        "SweepA_anchor_b24M_s0", "SweepA_anchor_b24M_s1",
        "SweepA_anchor_b48M_s0", "SweepA_anchor_b48M_s1",
        "SweepA_anchor_b72M_s0", "SweepA_anchor_b72M_s1"
    ]


def test_run_id_and_tags_agree_with_budget_and_seed():
    rows = read_rows(JSONL)
    for r in rows:
        nts = r["num_timesteps"]
        seed = r["seed"]
        expected_run_id = f"SweepA_anchor_{ga.budget_tag(nts)}_s{seed}"
        assert r["run_id"] == expected_run_id

        expected_tags = [
            "sweepA", "anchor", ga.budget_tag(nts), f"s{seed}", "anchora1b0c1d1"
        ]
        assert r["wandb_tags"] == expected_tags


def test_budget_cells_and_eval_cadence():
    rows = read_rows(JSONL)

    for r in rows:
        assert r["num_timesteps"] in ga.BUDGETS
        assert r["num_evals"] == ga.num_evals_for(r["num_timesteps"])

        nts = r["num_timesteps"]
        num_evals = r["num_evals"]
        assert gs.solve_budget(nts, num_evals)[0] == 8

    multiset = [(r["num_timesteps"], r["seed"]) for r in rows]
    expected_multiset = [(b, s) for b in ga.BUDGETS for s in ga.SEEDS]
    assert sorted(multiset) == sorted(expected_multiset)

    assert gs.solve_budget(24_000_000, 20)[1] == 24_903_680
    assert gs.solve_budget(48_000_000, 40)[1] == 51_118_080
    assert gs.solve_budget(72_000_000, 60)[1] == 77_332_480


def test_pinned_keys_on_every_line():
    rows = read_rows(JSONL)
    for r in rows:
        assert r["lifter_tilt_max"] == 0.05
        assert isinstance(r["lifter_tilt_max"], float)
        assert r["box_z_rot_range"] == 6.283185307179586
        assert r["episode_length"] == 400
        assert r["obs_include_velocity"] is True
        assert r["action_scale"] == 0.04
        assert r["action_rate"] == -0.7
        assert r["gripper_action_scale"] == 0.02
        assert r["gate_gripper_box_on_lift"] is False
        assert r["gate_gripper_align_on_lift"] is True
        assert r["entropy_cost"] == rg.ANCHOR["entropy_cost"]
        assert r["learning_rate"] == rg.ANCHOR["learning_rate"]
        assert r["reward_scaling"] == rg.ANCHOR["reward_scaling"]
        assert r["network_factory"]["policy_hidden_layer_sizes"] == [256, 256, 256]
        assert r["network_factory"]["value_hidden_layer_sizes"] == [256, 256, 256, 256, 256]
        assert r["num_resets_per_eval"] == 1
        assert r["video_every_evals"] == 5
        assert r["env_name"] == "UR3Pick"
        assert r["wandb_project"] == "UR3_pick_ppo"
        assert r["domain_rand.enable"] is True
        assert r["domain_rand.cube_mass.enable"] is True
        assert r["domain_rand.cube_mass.min"] == 0.85
        assert r["domain_rand.cube_mass.max"] == 1.15
        assert "num_eval_envs" not in r


def test_identical_outside_the_varying_keys():
    rows = read_rows(JSONL)
    varying = {"seed", "run_id", "wandb_tags", "num_timesteps", "num_evals"}

    base = {k: v for k, v in rows[0].items() if k not in varying}
    for r in rows[1:]:
        other = {k: v for k, v in r.items() if k not in varying}
        assert other == base


def test_no_stray_reward_keys():
    rows = read_rows(JSONL)
    for r in rows:
        stray = (gs.REWARD_KEYS - {"action_rate"}) & set(r)
        assert not stray


def test_24M_lines_equal_snappy2_lines():
    rows = read_rows(JSONL)
    snappy2_rows = read_rows(SNAPPY2)

    snappy2_by_id = {r["run_id"]: r for r in snappy2_rows}

    for seed in (0, 1):
        sweep_run_id = f"SweepA_anchor_b24M_s{seed}"
        snappy2_run_id = f"Snappy2_as04_ar70_g01_s{seed}"

        sweep_row = next(r for r in rows if r["run_id"] == sweep_run_id)
        snappy2_row = snappy2_by_id[snappy2_run_id]

        sweep_copy = copy.deepcopy(sweep_row)
        snappy2_copy = copy.deepcopy(snappy2_row)

        sweep_copy.pop("run_id", None)
        sweep_copy.pop("wandb_tags", None)
        sweep_copy.pop("lifter_tilt_max", None)

        snappy2_copy.pop("run_id", None)
        snappy2_copy.pop("wandb_tags", None)
        snappy2_copy.pop("lifter_tilt_max", None)

        assert sweep_copy == snappy2_copy


def test_regeneration_is_byte_identical():
    lines, _ = ga.build_lines(ga.SEEDS)
    generated = "\n".join(lines) + "\n"
    committed = JSONL.read_text(encoding="utf-8")
    assert generated == committed


def test_check_lines_accepts_committed_file_and_rejects_drift(tmp_path):
    rows = ga.check_lines(str(JSONL), 6, ga.SEEDS)
    assert len(rows) == 6

    headers = read_header_lines(JSONL)

    bad_tilt = copy.deepcopy(rows)
    bad_tilt[0]["lifter_tilt_max"] = 0.12
    path_a = tmp_path / "a.jsonl"
    _write_variant(path_a, headers, bad_tilt)
    with pytest.raises(AssertionError):
        ga.check_lines(str(path_a), 6, ga.SEEDS)

    duplicated = copy.deepcopy(rows) + [copy.deepcopy(rows[0])]
    path_b = tmp_path / "b.jsonl"
    _write_variant(path_b, headers, duplicated)
    with pytest.raises(AssertionError):
        ga.check_lines(str(path_b), 7, ga.SEEDS)

    extra_key = copy.deepcopy(rows)
    extra_key[0]["lift"] = 1.0
    path_c = tmp_path / "c.jsonl"
    _write_variant(path_c, headers, extra_key)
    with pytest.raises(AssertionError):
        ga.check_lines(str(path_c), 6, ga.SEEDS)

    bad_evals = copy.deepcopy(rows)
    for r in bad_evals:
        if r["run_id"] == "SweepA_anchor_b48M_s0":
            r["num_evals"] = 20
            break
    path_d = tmp_path / "d.jsonl"
    _write_variant(path_d, headers, bad_evals)
    with pytest.raises(AssertionError):
        ga.check_lines(str(path_d), 6, ga.SEEDS)


def test_header_states_totals_and_array():
    headers = read_header_lines(JSONL)
    header_text = "\n".join(headers)

    assert "TOTAL_STEPS=24903680" in header_text
    assert "TOTAL_STEPS=51118080" in header_text
    assert "TOTAL_STEPS=77332480" in header_text
    assert "--array=1-6%4" in header_text


def test_base_env_default_makes_the_pin_load_bearing():
    ur3_pick_path = REPO_ROOT / "mujoco_playground/_src/manipulation/my_ur3/ur3_pick.py"
    ur3_pick_text = ur3_pick_path.read_text(encoding="utf-8")
    assert re.search(r"lifter_tilt_max\s*=\s*0\.12", ur3_pick_text)


def test_run_experiment_forwards_lifter_tilt_max():
    run_exp_path = REPO_ROOT / "learning/notebooks/run_experiment.py"
    run_exp_text = run_exp_path.read_text(encoding="utf-8")
    assert '"lifter_tilt_max",' in run_exp_text
    assert 'env_overrides["lifter_tilt_max"]' in run_exp_text


def test_check_lines_rejects_uniform_video_every_evals_drift(tmp_path):
    rows = read_rows(JSONL)
    headers = read_header_lines(JSONL)

    drifted = copy.deepcopy(rows)
    for r in drifted:
        r["video_every_evals"] = 10
    path = tmp_path / "video_drift.jsonl"
    _write_variant(path, headers, drifted)
    with pytest.raises(AssertionError):
        ga.check_lines(str(path), 6, ga.SEEDS)
