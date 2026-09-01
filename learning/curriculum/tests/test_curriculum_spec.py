"""Guards on the curriculum rung spec and the ladder driver.

The spec assertions deliberately re-read the WRITTEN FILE rather than the
in-memory object build_spec() returns: what the cluster consumes is the file, and
a generator that builds the right dict but serialises it wrongly would pass an
in-memory check.

The driver is exercised with `run_experiment` stubbed and the clock injected --
no brax, no MJX, no GPU, no wall-clock waiting.

Run:
    python -m pytest learning/curriculum/tests/test_curriculum_spec.py -q
    python -m pytest learning/curriculum/tests/test_curriculum_spec.py -q -k driver
"""

import importlib
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "batch_runs", "scripts"))

from batch_runs.curriculum.gen_curriculum import (  # noqa: E402
    EXPECTED_ORDER,
    L0_5_LIGHT_OVERRIDES,
    PHYSICS_AXES,
    build_spec,
    dumps,
)

SPEC_PATH = os.path.join(REPO, "batch_runs", "curriculum", "UR3Pick_curriculum.json")


@pytest.fixture(scope="module")
def spec():
    """The file on disk -- what the cluster actually reads."""
    assert os.path.exists(SPEC_PATH), (
        f"{SPEC_PATH} missing. Generate it: "
        f"python batch_runs/curriculum/gen_curriculum.py --seed 0"
    )
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ladder():
    mod = importlib.import_module("batch_runs.sweeps.gen_dr_ladder")
    return mod._CONFIGS


# --- the spec matches the ladder -------------------------------------------

def test_exactly_six_rungs_in_curriculum_order(spec):
    assert [r["config_id"] for r in spec["rungs"]] == list(EXPECTED_ORDER)


def test_each_ladder_rungs_overrides_equal_the_ladder_entry_exactly(spec, ladder):
    """No drift from gen_dr_ladder._CONFIGS, which is the source of truth.

    Exact dict equality, not a subset check: an extra key here would mean the
    curriculum trains a rung the DR study never measured.

    L0_5_light has no entry in _CONFIGS (it is curriculum-internal, injected
    by gen_curriculum.py) -- it is checked separately below, against the
    generator's own L0_5_LIGHT_OVERRIDES constant.
    """
    by_id = {c[0]: c[1] for c in ladder}
    for rung in spec["rungs"]:
        if rung["config_id"] == "L0_5_light":
            continue
        assert rung["overrides"] == by_id[rung["config_id"]], (
            f"{rung['config_id']} drifted from _CONFIGS"
        )


def test_l0_5_overrides_equal_the_generators_own_constant(spec):
    """Load-bearing: catches a sparse/hand-retyped L0_5_LIGHT_OVERRIDES.

    The highest-consequence trap in this change is an L0.5 overrides dict
    that omits keys -- an omitted key falls back to the baked env default,
    not to L0, silently training L0.5 WIDER than L1. This test pins the
    on-disk spec's L0.5 overrides to the generator's own module constant by
    exact dict equality, so a mutation that sparsifies
    L0_5_LIGHT_OVERRIDES flips this test red.
    """
    r = {x["config_id"]: x for x in spec["rungs"]}
    assert r["L0_5_light"]["overrides"] == L0_5_LIGHT_OVERRIDES


def test_l0_5_key_set_matches_l0_and_differs_in_exactly_two_keys(spec):
    r = {x["config_id"]: x for x in spec["rungs"]}
    l0 = r["L0_none"]["overrides"]
    l0_5 = r["L0_5_light"]["overrides"]
    assert len(l0) == 13
    assert set(l0_5) == set(l0)
    diff_keys = {k for k in l0 if l0[k] != l0_5[k]}
    assert diff_keys == {"init_start_random", "box_xy_jitter"}, diff_keys


def test_every_rung_carries_episode_length_400(spec):
    """ur3_pick.default_config()'s 250 is stale; inheriting it once trained 30
    runs at the wrong horizon."""
    for rung in spec["rungs"]:
        effective = {**spec["defaults"], **rung["overrides"]}
        assert effective["episode_length"] == 400, rung["config_id"]


def test_physics_rungs_also_set_the_master_switch(spec):
    """domain_rand.enable gates ONLY the _randomize_physics axes, and it is not
    implied by setting one of them (gen_dr_ladder.py:41-46). A rung that
    randomises cube mass without it silently trains as L1."""
    # How many _randomize_physics axes each rung is known to carry. Pinned per
    # rung, not just as a total: PHYSICS_AXES once held the bare axis names
    # while the ladder writes them with a `.enable` suffix, so `touched` was
    # empty everywhere and this test passed no matter what the spec said. A
    # count of matched axes catches a single renamed axis too, which a mere
    # "some rung matched" liveness check does not.
    EXPECTED_HITS = {
        "L0_none": 0, "L0_5_light": 0,  # L0.5 touches no PHYSICS_AXES
        "L1_pos": 0,
        "L2_pos_cube": 4,            # cube mass/friction/size_xy/size_z
        "L3_pos_cube_robot": 6,      # + arm stiffness/damping (action_delay is
                                     #   gated independently, not a physics axis)
        "L4_full": 7,                # + gravity (cube_force / joint_torque /
                                     #   obs_noise are also gated independently)
    }
    for rung in spec["rungs"]:
        ov = rung["overrides"]
        touched = [a for a in PHYSICS_AXES if a in ov]
        assert len(touched) == EXPECTED_HITS[rung["config_id"]], (
            f"{rung['config_id']} matched {len(touched)} physics axes, expected "
            f"{EXPECTED_HITS[rung['config_id']]} -- PHYSICS_AXES no longer "
            f"matches the ladder's key names, so this guard is dead"
        )
        if touched:
            assert ov.get("domain_rand.enable") is True, (
                f"{rung['config_id']} sets {touched} without "
                f"domain_rand.enable -- the randomisation would do nothing"
            )


def test_warm_start_chain_has_no_gaps_and_no_self_reference(spec):
    ids = [r["config_id"] for r in spec["rungs"]]
    assert spec["rungs"][0]["warm_start_from"] is None
    for i, rung in enumerate(spec["rungs"][1:], start=1):
        assert rung["warm_start_from"] == ids[i - 1]
        assert rung["warm_start_from"] != rung["config_id"]


def test_run_ids_match_the_names_wp7_expects(spec):
    assert [r["run_id"] for r in spec["rungs"]] == [
        f"Curr_{c}_s{spec['seed']}" for c in EXPECTED_ORDER
    ]


def test_defaults_carry_the_curriculum_keys(spec):
    d = spec["defaults"]
    assert d["num_resets_per_eval"] == 1          # sweep rule
    assert d["num_evals"] >= 15                   # sweep rule
    assert d["obs_include_velocity"] is True      # obs stays 33D
    assert d["normalizer_count_reset"] is None    # documented, off by default
    assert set(d["early_stop"]) == {
        "strategy", "window", "patience", "min_delta", "min_steps",
    }
    assert d["early_stop"]["strategy"] == "windowed"


def test_defaults_carry_the_defence_hyperparameters_and_v2_budget(spec):
    """L0 peaked at eval/episode_success 0.016 in v1 (env-default
    action_scale=0.015) against the archived L0_none_vel_s1's 1.000 -- see
    gen_curriculum.py's _DEFENCE_* block. These must be in `defaults`, not
    left to the per-rung overrides or the env default."""
    d = spec["defaults"]
    assert d["num_timesteps"] == 40_000_000
    assert d["num_evals"] == 42
    assert d["action_scale"] == 0.04
    assert d["gripper_action_scale"] == 0.02
    assert d["action_rate"] == -0.7
    assert d["entropy_cost"] == 0.02
    assert d["learning_rate"] == 3e-4
    assert d["reward_scaling"] == 0.03
    assert d["network_factory"] == {
        "policy_hidden_layer_sizes": [256, 256, 256],
        "value_hidden_layer_sizes": [256, 256, 256, 256, 256],
    }
    assert d["gate_gripper_align_on_lift"] is True
    assert d["gate_gripper_box_on_lift"] is False
    assert d["num_eval_envs"] == 256


def test_defaults_own_no_dr_or_position_keys(spec):
    """DR and position keys stay owned by the per-rung overrides. A
    domain_rand.* or box_z_rot_range key in `defaults` would apply to every
    rung including L0/L0.5, which must stay deterministic."""
    d = spec["defaults"]
    assert not any(k.startswith("domain_rand.") for k in d)
    assert "box_z_rot_range" not in d
    assert "cube_mass" not in d


# --- the generator is reproducible -----------------------------------------

def test_regenerating_is_byte_identical(spec):
    on_disk = open(SPEC_PATH, encoding="utf-8").read()
    assert dumps(build_spec(seed=spec["seed"])) == on_disk


def test_generator_refuses_to_overwrite_without_force(tmp_path):
    from batch_runs.curriculum.gen_curriculum import main
    out = tmp_path / "spec.json"
    assert main(["--seed", "0", "--out", str(out)]) == 0
    # identical content -> no-op, no error
    assert main(["--seed", "0", "--out", str(out)]) == 0
    # different content -> refuses
    out.write_text('{"rungs": []}', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--seed", "0", "--out", str(out)])
    # ...unless forced
    assert main(["--seed", "0", "--out", str(out), "--force"]) == 0


# --- driver ----------------------------------------------------------------

@pytest.fixture(scope="module")
def drv():
    return importlib.import_module("run_curriculum_ur3")


class FakeClock:
    """Deterministic monotonic clock: every read advances by `step`."""

    def __init__(self, step=100.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def make_params(tag):
    """A brax-shaped 3-tuple of real arrays, so flax can serialise it."""
    return (
        {"count": np.float32(1.0), "mean": np.zeros(3, np.float32)},
        {"params": {"h": np.full((2, 2), float(tag), np.float32)}},
        {"params": {"v": np.full((2,), float(tag), np.float32)}},
    )


def make_runner(record, best=True):
    def runner(cfg, out_dir):
        i = len(record)
        record.append({"cfg": cfg, "out_dir": out_dir,
                       "warm": cfg.get("warm_start_params")})
        p = make_params(i)
        return {"params": make_params(100 + i),
                "best_params": p if best else None,
                "wandb_run_id": cfg["run_id"], "steps_completed": 1000,
                "stopped_early": False, "best_reward": 1.0, "best_step": 1}
    return runner


def test_driver_runs_all_six_rungs_in_order(spec, drv, tmp_path):
    rec = []
    s = drv.run_curriculum(spec, str(tmp_path), wall_budget_s=1e9,
                           runner=make_runner(rec), clock=FakeClock(),
                           group="g")
    assert [r["cfg"]["curriculum_rung"] for r in rec] == list(EXPECTED_ORDER)
    assert s["n_completed"] == 6
    assert s["stopped_reason"] is None


def test_driver_hands_the_exact_object_forward(spec, drv, tmp_path):
    """Identity, not shape: the next rung must receive the very tree the
    previous rung returned, or the warm start is not what it claims."""
    rec = []
    drv.run_curriculum(spec, str(tmp_path), wall_budget_s=1e9,
                       runner=make_runner(rec), clock=FakeClock(), group="g")
    assert rec[0]["warm"] is None, "L0 must be a cold start"
    for i in range(1, 6):
        assert rec[i]["warm"] is not None

    # The chain itself: rung i's warm params ARE the object rung i-1 returned.
    seen = []
    rec2 = []

    def runner(cfg, out_dir):
        p = make_params(len(rec2))
        rec2.append(cfg.get("warm_start_params"))
        seen.append(p)
        return {"params": make_params(99), "best_params": p,
                "wandb_run_id": cfg["run_id"], "steps_completed": 1,
                "stopped_early": False, "best_reward": 1.0, "best_step": 1}

    drv.run_curriculum(spec, str(tmp_path / "b"), wall_budget_s=1e9,
                       runner=runner, clock=FakeClock(), group="g")
    for i in range(1, 6):
        assert rec2[i] is seen[i - 1], f"rung {i} did not get rung {i-1}'s object"


def test_driver_falls_back_to_final_params_when_there_is_no_best(spec, drv, tmp_path):
    rec = []
    s = drv.run_curriculum(spec, str(tmp_path), wall_budget_s=1e9,
                           runner=make_runner(rec, best=False),
                           clock=FakeClock(), group="g")
    assert s["n_completed"] == 6
    assert rec[1]["warm"] is not None


def test_driver_writes_each_rungs_params_before_the_next_starts(spec, drv, tmp_path):
    """A timeout must cost one rung, not the ladder."""
    order = []

    def writer(path, params):
        order.append(("write", os.path.basename(os.path.dirname(
            os.path.dirname(path)))))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(b"x")
        return 1

    def runner(cfg, out_dir):
        order.append(("run", cfg["curriculum_rung"]))
        return {"params": make_params(0), "best_params": make_params(1),
                "wandb_run_id": cfg["run_id"], "steps_completed": 1,
                "stopped_early": False, "best_reward": 1.0, "best_step": 1}

    drv.run_curriculum(spec, str(tmp_path), wall_budget_s=1e9, runner=runner,
                       clock=FakeClock(), group="g", writer=writer)
    assert order == [
        x for c in EXPECTED_ORDER for x in (("run", c), ("write", c))
    ], order


def test_driver_really_serialises_the_handoff_to_disk(spec, drv, tmp_path):
    """The default writer, not a stub -- flax must accept the tree."""
    rec = []
    drv.run_curriculum(spec, str(tmp_path), wall_budget_s=1e9,
                       runner=make_runner(rec), clock=FakeClock(), group="g")
    for cid in EXPECTED_ORDER:
        p = tmp_path / "g" / cid / "trained_policy" / "handoff_params.msgpack"
        assert p.exists() and p.stat().st_size > 0, p


def test_driver_stops_before_rung_one_on_a_zero_wall_budget(spec, drv, tmp_path):
    rec = []
    s = drv.run_curriculum(spec, str(tmp_path), wall_budget_s=0.0,
                           runner=make_runner(rec), clock=FakeClock(),
                           group="g")
    assert s["n_completed"] == 1, "the first rung must always be attempted"
    assert len(rec) == 1
    assert s["stopped_reason"] is not None
    marker = tmp_path / "g" / "_resume.json"
    assert marker.exists()
    m = json.loads(marker.read_text())
    assert m["next_index"] == 1 and m["next_config_id"] == "L0_5_light"


def test_driver_exits_zero_when_the_budget_stops_it(spec, drv, tmp_path, monkeypatch):
    """SIGKILL vs clean stop: SLURM must record COMPLETED, not TIMEOUT."""
    monkeypatch.setattr(drv, "_default_runner", make_runner([]))
    argv = ["--spec", SPEC_PATH, "--out-root", str(tmp_path),
            "--wall-budget-s", "0", "--group", "g"]
    assert drv.main(argv) == 0


def test_group_is_stamped_when_the_spec_leaves_it_null(spec, drv):
    assert spec["wandb_group"] is None
    g = drv.resolve_group(spec)
    assert g.startswith("curriculum_") and len(g) > len("curriculum_")
    assert drv.resolve_group(spec, "pinned") == "pinned"


def test_rung_cfg_carries_what_run_experiment_needs(spec, drv):
    # Looked up by config_id, not positionally -- inserting L0.5 at index 1
    # is exactly the kind of change a positional spec["rungs"][2] would break.
    rung = next(r for r in spec["rungs"] if r["config_id"] == "L2_pos_cube")
    cfg = drv.build_rung_cfg(spec, rung, "grp", warm_params=("a", "b", "c"))
    assert cfg["curriculum_rung"] == rung["config_id"]
    assert cfg["warm_start_from"] == rung["warm_start_from"]
    assert cfg["warm_start_params"] == ("a", "b", "c")
    assert cfg["wandb_group"] == "grp"
    assert cfg["episode_length"] == 400
    assert cfg["run_id"] == rung["run_id"]
    # L0 gets no warm params at all
    cold = drv.build_rung_cfg(spec, spec["rungs"][0], "grp")
    assert "warm_start_params" not in cold
