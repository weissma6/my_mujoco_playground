"""WP1: test gen_sweep_dr_dose.py generator. Run command:
cd /Users/matthiasweiss/Claude/Projects/worktrees/sweep_dr_dose && \
JAX_PLATFORM_NAME=cpu /Users/matthiasweiss/miniconda3/envs/mujoco/bin/python \
  -m pytest batch_runs/sweeps/test_gen_sweep_dr_dose.py -q
"""

import os
import sys
import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.join(str(REPO), "batch_runs", "sweeps"))

import gen_sweep_dr_dose as g
import gen_dr_ladder as dl
import gen_rewardgate as rg
import gen_satsweep as gs
import gen_snappy as sn

from learning.notebooks.run_experiment import (
    _extract_ppo_overrides, apply_validated_overrides)
from mujoco_playground.config import manipulation_params

JSONL = REPO / "batch_runs" / "sweeps" / "UR3Pick_sweep_dr_dose.jsonl"
CURRICULUM = REPO / "batch_runs" / "curriculum" / "UR3Pick_curriculum.json"
SNAPPY2_JSONL = REPO / "batch_runs" / "sweeps" / "UR3Pick_snappy2.jsonl"


def ladder_entry(cid):
    """Return (config_id, overrides, tags) tuple from dl._CONFIGS matching cid."""
    return next((c, o, t) for c, o, t in dl._CONFIGS if c == cid)


@pytest.fixture(scope="module")
def rows():
    """Rows from the committed UR3Pick_sweep_dr_dose.jsonl."""
    return g.check_lines(str(JSONL), 8, [0, 1])


@pytest.fixture(scope="module")
def curriculum():
    """Curriculum from UR3Pick_curriculum.json."""
    with open(CURRICULUM, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    """Load data rows from JSONL, skipping comments and blanks."""
    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                result.append(json.loads(line))
    return result


def get_config_id_from_run_id(run_id):
    """Extract config_id from run_id: SweepB_<config_id>_s<seed>."""
    return run_id[len("SweepB_"):].rsplit("_s", 1)[0]


class TestShape:
    """Test 1: JSONL structure and line counts."""

    def test_jsonl_has_8_data_lines(self, rows):
        assert len(rows) == 8

    def test_build_lines_returns_9_lines(self):
        lines, total = g.build_lines([0, 1])
        assert len(lines) == 9
        assert lines[0].startswith("#")

    def test_total_steps_matches_budget(self):
        lines, total = g.build_lines([0, 1])
        assert total == 30_474_240

    def test_solve_budget_consistency(self):
        nts, total = gs.solve_budget(30_000_000, 32)
        assert nts == 6
        assert total == 30_474_240


class TestIds:
    """Test 2: run_id structure and uniqueness."""

    def test_run_ids_ordered_by_config_then_seed(self, rows):
        run_ids = [r["run_id"] for r in rows]
        expected = [
            f"SweepB_{cid}_s{s}"
            for cid in g.CONFIG_IDS for s in (0, 1)
        ]
        assert run_ids == expected

    def test_all_run_ids_unique(self, rows):
        run_ids = [r["run_id"] for r in rows]
        assert len(set(run_ids)) == len(run_ids)

    def test_config_ids_constant(self):
        assert g.CONFIG_IDS == ("L1_pos", "L2_pos_cube", "L3_pos_cube_robot", "L4_full")


class TestFrozenDefenceValues:
    """Test 3: frozen defence values on every row."""

    @pytest.mark.parametrize("row_idx", range(8))
    def test_frozen_defence_values(self, rows, row_idx):
        r = rows[row_idx]
        rid = r["run_id"]

        assert r["entropy_cost"] == rg.ANCHOR["entropy_cost"], rid
        assert r["learning_rate"] == rg.ANCHOR["learning_rate"], rid
        assert r["reward_scaling"] == rg.ANCHOR["reward_scaling"], rid

        nf = r["network_factory"]
        assert tuple(nf["policy_hidden_layer_sizes"]) == \
            rg.ANCHOR["policy_hidden_layer_sizes"], rid
        assert tuple(nf["value_hidden_layer_sizes"]) == \
            gs.VALUE_HIDDEN_LAYER_SIZES, rid

        assert r["episode_length"] == 400, rid
        assert r["box_z_rot_range"] == 6.283185307179586, rid
        assert r["obs_include_velocity"] is True, rid
        assert r["action_scale"] == 0.04, rid
        assert r["action_rate"] == -0.7, rid
        assert r["gripper_action_scale"] == sn.GRIPPER_ACTION_SCALE, rid
        assert r[rg.GATE_BOX_KEY] is False, rid
        assert r[rg.GATE_ALIGN_KEY] is True, rid
        assert r["lifter_tilt_max"] == dl.LIFTER_TILT_MAX, rid
        assert r["lifter_tilt_max"] == 0.12, rid
        assert r["num_timesteps"] == 30_000_000, rid
        assert r["num_evals"] == 32, rid
        assert r["num_eval_envs"] == 256, rid
        assert r["num_resets_per_eval"] == 1, rid
        assert r["domain_rand.enable"] is True, rid
        assert r["domain_rand.cube_mass.enable"] is True, rid
        assert r["domain_rand.cube_mass.min"] == 0.85, rid
        assert r["domain_rand.cube_mass.max"] == 1.15, rid
        assert r["env_name"] == "UR3Pick", rid
        assert r["wandb_project"] == "UR3_pick_ppo", rid
        assert r["video_every_evals"] == 5, rid

        assert not (gs.REWARD_KEYS - {"action_rate"} & set(r)), rid


class TestTags:
    """Test 4: wandb_tags structure."""

    def test_tags_structure(self, rows):
        for r in rows:
            seed = r["seed"]
            cid = get_config_id_from_run_id(r["run_id"])
            tags = r["wandb_tags"]

            assert "sweepB" in tags, r["run_id"]
            assert "scratch" in tags, r["run_id"]
            assert f"s{seed}" in tags, r["run_id"]
            assert "anchora1b0c1d1" in tags, r["run_id"]

            for tag in ladder_entry(cid)[2]:
                assert tag in tags, f"{r['run_id']} missing tag {tag}"


class TestDrSetEqualsCurriculumRung:
    """Test 5: DR set matches curriculum rung."""

    def test_dr_overrides_match_curriculum(self, rows, curriculum):
        for r in rows:
            cid = get_config_id_from_run_id(r["run_id"])

            rung = next((rng for rng in curriculum["rungs"]
                        if rng["config_id"] == cid), None)
            assert rung is not None, f"No rung for {cid}"

            rung_overrides = rung["overrides"]

            dr_keys_on_row = {k for k in r if k.startswith("domain_rand.")}
            dr_keys_in_rung = {
                k for k in rung_overrides if k.startswith("domain_rand.")
            }

            for k in dr_keys_in_rung:
                assert r[k] == rung_overrides[k], \
                    f"{r['run_id']}: {k} mismatch"

            expected_dr_keys = dr_keys_in_rung | g.BASE_DR_KEYS
            assert dr_keys_on_row == expected_dr_keys, \
                f"{r['run_id']}: DR key mismatch"

            assert r["lifter_tilt_max"] == \
                rung_overrides.get("lifter_tilt_max", dl.LIFTER_TILT_MAX), \
                r["run_id"]

    def test_dl_configs_subset_of_row(self, rows):
        for r in rows:
            cid = get_config_id_from_run_id(r["run_id"])
            _, dl_overrides, _ = ladder_entry(cid)

            for k, v in dl_overrides.items():
                assert r[k] == v, f"{r['run_id']}: {k} mismatch with dl._CONFIGS"


class TestL1IsEmptyOverride:
    """Test 6: L1_pos has empty override, L4_full rows have 14 DR keys."""

    def test_l1_pos_empty_override(self):
        _, overrides, _ = ladder_entry("L1_pos")
        assert overrides == {}

    def test_l1_pos_rows_have_base_dr_keys_only(self, rows):
        l1_rows = [r for r in rows if "L1_pos" in r["run_id"]]
        assert len(l1_rows) == 2

        for r in l1_rows:
            dr_keys = {k for k in r if k.startswith("domain_rand.")}
            assert dr_keys == g.BASE_DR_KEYS, r["run_id"]

    def test_l4_full_rows_have_14_dr_keys(self, rows):
        l4_rows = [r for r in rows if "L4_full" in r["run_id"]]
        assert len(l4_rows) == 2

        for r in l4_rows:
            dr_keys = {k for k in r if k.startswith("domain_rand.")}
            assert len(dr_keys) == 14, f"{r['run_id']}: {len(dr_keys)} keys"


class TestPairwiseIdentity:
    """Test 7: non-varying keys are identical across all rows."""

    def test_identical_except_varying(self, rows):
        varying = {"seed", "run_id", "wandb_tags"} | g.dr_keys()
        base = {k: v for k, v in rows[0].items() if k not in varying}

        for r in rows[1:]:
            other = {k: v for k, v in r.items() if k not in varying}
            assert other == base, \
                f"{r['run_id']} differs from {rows[0]['run_id']}"


class TestDefenceLineReused:
    """Test 8: defence line reused from snappy2, not retyped."""

    def test_defence_from_snappy2(self, rows):
        snappy2_rows = load_jsonl(str(SNAPPY2_JSONL))

        for seed in (0, 1):
            snappy_row = next(
                (r for r in snappy2_rows
                 if r["run_id"] == f"Snappy2_as04_ar70_g01_s{seed}"),
                None
            )
            assert snappy_row is not None, f"No Snappy2_as04_ar70_g01_s{seed}"
            assert "lifter_tilt_max" not in snappy_row
            assert "num_eval_envs" not in snappy_row

            snappy_stripped = {
                k: v for k, v in snappy_row.items()
                if k not in {"run_id", "wandb_tags", "num_timesteps", "num_evals"}
            }

            sweep_row = next(
                r for r in rows if r["run_id"] == f"SweepB_L1_pos_s{seed}"
            )

            sweep_stripped = {
                k: v for k, v in sweep_row.items()
                if k not in {"run_id", "wandb_tags", "num_timesteps", "num_evals",
                           "num_eval_envs", "lifter_tilt_max"} and
                k not in (g.dr_keys() - g.BASE_DR_KEYS)
            }

            assert sweep_stripped == snappy_stripped


class TestRunnerExtractionPath:
    """Test 9: every key survives runner's extraction path."""

    def test_ppo_overrides_extraction(self, rows):
        for r in rows:
            ppo = _extract_ppo_overrides(r)
            base = manipulation_params.brax_ppo_config("UR3Pick").to_dict()

            apply_validated_overrides(base, ppo, strict=True)

            assert ppo["num_timesteps"] == r["num_timesteps"], r["run_id"]
            assert ppo["num_evals"] == r["num_evals"], r["run_id"]
            assert ppo["num_eval_envs"] == r["num_eval_envs"], r["run_id"]
            assert ppo["num_resets_per_eval"] == r["num_resets_per_eval"], r["run_id"]
            assert ppo["entropy_cost"] == r["entropy_cost"], r["run_id"]
            assert ppo["learning_rate"] == r["learning_rate"], r["run_id"]
            assert ppo["reward_scaling"] == r["reward_scaling"], r["run_id"]
            assert ppo["network_factory"] == r["network_factory"], r["run_id"]

    def test_all_keys_forwarded(self, rows):
        src = Path(__file__).parents[2] / "learning" / "notebooks" / "run_experiment.py"
        with open(src, encoding="utf-8") as f:
            src_text = f.read()

        env_forwarded = set(re.findall(r'if "([A-Za-z_]+)" in cfg:', src_text))
        metadata_read = set(re.findall(r'cfg(?:\.get\(|\[)"([A-Za-z_]+)"', src_text))

        ppo_keys = {"num_timesteps", "num_evals", "num_eval_envs",
                    "num_resets_per_eval", "entropy_cost", "learning_rate",
                    "reward_scaling", "network_factory"}

        for r in rows:
            for k in r:
                is_domain_rand = k.startswith("domain_rand.")
                is_ppo = k in ppo_keys
                is_env = k in env_forwarded
                is_metadata = k in metadata_read

                assert is_domain_rand or is_ppo or is_env or is_metadata, \
                    f"{r['run_id']}: key {k} not forwarded"

        assert 'k.startswith("domain_rand.")' in src_text


class TestRegenerationByteIdentical:
    """Test 10: regeneration is byte-identical."""

    def test_byte_identical_regeneration(self):
        lines, _ = g.build_lines([0, 1])
        regenerated = "\n".join(lines) + "\n"

        with open(JSONL, encoding="utf-8") as f:
            committed = f.read()

        assert regenerated == committed


class TestHeaderContent:
    """Test 11: header contains required substrings."""

    def test_header_substrings(self):
        with open(JSONL, encoding="utf-8") as f:
            lines = [l for l in f if l.strip().startswith("#")]

        header = "\n".join(lines)

        assert "TOTAL_STEPS=30474240" in header
        assert "--array=1-8%4" in header
        assert "gen_sweep_dr_dose.py" in header
        assert "Curr_v3" in header
        assert "INIT-POSE LIBRARY: level='mid' n_poses=60" in header


class TestCheckLinesMutations:
    """Test 12: check_lines detects mutations."""

    def helper_mutate_and_check(self, mutation_func, should_fail=True):
        """Helper: mutate the JSONL and verify check_lines catches it."""
        rows = load_jsonl(str(JSONL))
        header_lines = []
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("#") or not line.strip():
                    header_lines.append(line.rstrip())
                else:
                    break

        mutation_func(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "mut.jsonl"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for line in header_lines:
                    f.write(line + "\n")
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            if should_fail:
                with pytest.raises(AssertionError):
                    g.check_lines(str(tmp_path), 8, [0, 1])
            else:
                g.check_lines(str(tmp_path), 8, [0, 1])

    def test_mutation_action_rate(self):
        def mutate(rows):
            rows[0]["action_rate"] = -0.1
        self.helper_mutate_and_check(mutate)

    def test_mutation_duplicate_run_id(self):
        def mutate(rows):
            rows[1]["run_id"] = rows[0]["run_id"]
        self.helper_mutate_and_check(mutate)

    def test_mutation_missing_row(self):
        def mutate(rows):
            rows.pop()
        self.helper_mutate_and_check(mutate)

    def test_mutation_lifter_tilt(self):
        def mutate(rows):
            rows[0]["lifter_tilt_max"] = 0.05
        self.helper_mutate_and_check(mutate)

    def test_mutation_num_eval_envs(self):
        def mutate(rows):
            rows[0]["num_eval_envs"] = 128
        self.helper_mutate_and_check(mutate)

    def test_mutation_gripper_box(self):
        def mutate(rows):
            rows[0]["gripper_box"] = 4.0
        self.helper_mutate_and_check(mutate)

    def test_mutation_missing_dr_key(self):
        def mutate(rows):
            l2_row = next(r for r in rows if "L2_pos_cube" in r["run_id"])
            del l2_row["domain_rand.cube_friction.enable"]
        self.helper_mutate_and_check(mutate)


class TestCliOverwriteGuard:
    """Test 13: CLI refuses to overwrite without --force."""

    def test_cli_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "x.jsonl"
            tmp_path.write_text("some content\n")

            gen_file = Path(__file__).parent / "gen_sweep_dr_dose.py"
            result = subprocess.run(
                [sys.executable, str(gen_file), "--out", str(tmp_path)],
                capture_output=True, text=True
            )
            assert result.returncode != 0

    def test_cli_overwrites_with_force(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "x.jsonl"
            tmp_path.write_text("some content\n")

            gen_file = Path(__file__).parent / "gen_sweep_dr_dose.py"
            result = subprocess.run(
                [sys.executable, str(gen_file), "--out", str(tmp_path),
                 "--force"],
                capture_output=True, text=True
            )
            assert result.returncode == 0
            assert "set --array=1-8%4" in result.stdout


class TestLadderCellsLeakGuard:
    """Test 14 (reviewer-added): ladder_cells()'s non-DR-key leak guard fires.

    gen_sweep_dr_dose.ladder_cells() asserts every override key on a used
    rung starts with "domain_rand." -- the module docstring calls this "would
    leak into the line". This is only a real guard if it actually raises when
    gen_dr_ladder._CONFIGS grows a stray non-DR key on one of the four rungs
    this sweep imports (e.g. a future edit adding a bare "lifter_tilt_max" or
    "wandb_tags" key inside a rung's override dict, which would silently
    override or bypass this generator's own values instead of failing loud).
    """

    def test_leak_guard_fires_on_stray_key(self, monkeypatch):
        patched = list(dl._CONFIGS)
        idx = next(i for i, (cid, _, _) in enumerate(patched)
                   if cid == "L2_pos_cube")
        cid, overrides, tags = patched[idx]
        bad_overrides = dict(overrides)
        bad_overrides["lifter_tilt_max"] = 0.05
        patched[idx] = (cid, bad_overrides, tags)
        monkeypatch.setattr(dl, "_CONFIGS", patched)

        with pytest.raises(AssertionError):
            g.ladder_cells()

    def test_leak_guard_silent_on_clean_configs(self):
        """Sanity check: the guard does NOT fire on the real, unmodified ladder."""
        g.ladder_cells()
