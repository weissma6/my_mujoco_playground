"""Tests for the curriculum warm-start loader.

CPU only -- no MJX, no env is ever built or stepped. Exercises brax's real
`restore_params` semantics against real archived checkpoints.

Run:
    python -m pytest learning/curriculum/tests/test_warm_start.py -q
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from flax import serialization

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from learning.curriculum.warm_start import (  # noqa: E402
    ArchitectureMismatch,
    build_template,
    load_metadata,
    load_params_3tuple,
    params_sha256,
    rescale_normalizer_count,
)

POLICIES = os.path.join(REPO, "evaluation", "downloaded_policies")
L0 = "L0_none_vel_s1_20260729_104930_2201"
LADDER = [
    "L0_none_vel_s1_20260729_104930_2201",
    "L1_pos_vel_s1_20260729_112044_2201",
    "L2_pos_cube_vel_s1_20260729_112246_2201",
    "L3_pos_cube_robot_vel_s1_20260729_115408_2201",
    "L4_full_vel_s1_20260729_122736_2201",
]
NF = {
    "policy_hidden_layer_sizes": [32, 32, 32, 32],
    "value_hidden_layer_sizes": [256, 256, 256, 256, 256],
    "policy_obs_key": "state",
    "value_obs_key": "state",
}


def pdir(run_id):
    return os.path.join(POLICIES, run_id)


def ppath(run_id):
    return os.path.join(pdir(run_id), "params.msgpack")


def require(run_id):
    """A missing fixture must FAIL, not skip -- a skipped check reads as passed."""
    p = ppath(run_id)
    assert os.path.exists(p), f"fixture missing: {p}"
    return p


def load_L0():
    return load_params_3tuple(require(L0), 33, 7, NF)


# --- the fixture itself -----------------------------------------------------

def test_metadata_matches_what_the_tests_assume():
    m = load_metadata(pdir(L0))
    assert m["obs_dim"] == 33
    assert m["act_dim"] == 7
    assert list(m["network_factory"]["policy_hidden_layer_sizes"]) == [32, 32, 32, 32]


# --- 1. bit-identical round trip -------------------------------------------

def test_loaded_policy_is_bit_identical_to_the_file():
    p = load_L0()
    raw = serialization.from_bytes(build_template(33, 7, NF), open(require(L0), "rb").read())
    got = jax.tree_util.tree_leaves(p[1])
    want = jax.tree_util.tree_leaves(raw["1"])
    assert len(got) == 10
    for a, b in zip(got, want):
        assert np.array_equal(np.asarray(a), np.asarray(b))   # exact, not allclose


def test_value_net_also_round_trips_and_the_tuple_is_ordered():
    p = load_L0()
    assert len(p) == 3
    assert type(p[0]).__name__ == "RunningStatisticsState"
    assert np.asarray(p[0].mean).shape == (33,)
    assert len(jax.tree_util.tree_leaves(p[2])) == 12   # 5 hidden + output, w+b


def test_normalizer_count_is_the_runs_total_env_steps():
    p = load_L0()
    assert float(np.asarray(p[0].count)) == 31948800.0


# --- 2. it differs from a random init, and it acts ---------------------------

def _net():
    return ppo_networks.make_ppo_networks(
        observation_size=33, action_size=7,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256,) * 5,
    )


def test_loaded_params_differ_from_a_random_init():
    p = load_L0()
    rand = _net().policy_network.init(jax.random.PRNGKey(999))
    got = jax.tree_util.tree_leaves(p[1])
    rnd = jax.tree_util.tree_leaves(rand)
    assert not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(got, rnd))


def test_random_init_gives_zero_action_but_the_warm_start_acts():
    """The behavioural proof that the weights took effect, no GPU required."""
    net = _net()
    inf = ppo_networks.make_inference_fn(net)
    obs = jnp.zeros((1, 33))
    key = jax.random.PRNGKey(1)

    fresh_norm = running_statistics.init_state(jax.ShapeDtypeStruct((33,), jnp.float32))
    rand = net.policy_network.init(jax.random.PRNGKey(999))
    a_cold, _ = inf((fresh_norm, rand), deterministic=True)(obs, key)

    p = load_L0()
    a_warm, _ = inf((p[0], p[1]), deterministic=True)(obs, key)

    assert np.allclose(np.asarray(a_cold), 0.0)          # cold: all zeros
    assert np.isfinite(np.asarray(a_warm)).all()
    assert not np.allclose(np.asarray(a_warm), 0.0)      # warm: a real action


# --- 3. brax's own restore lines install exactly these params ---------------

def test_brax_restore_semantics_install_the_loaded_params():
    """Replays brax/training/agents/ppo/train.py:628-635 verbatim."""
    p = load_L0()
    net = _net()
    init = ppo_losses.PPONetworkParams(
        policy=net.policy_network.init(jax.random.PRNGKey(999)),
        value=net.value_network.init(jax.random.PRNGKey(999)),
    )
    restored = init.replace(policy=p[1], value=p[2])      # restore_value_fn=True

    for a, b in zip(jax.tree_util.tree_leaves(restored.policy),
                    jax.tree_util.tree_leaves(p[1])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(jax.tree_util.tree_leaves(restored.value),
                    jax.tree_util.tree_leaves(p[2])):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# --- 4. the architecture guard ----------------------------------------------

def test_flax_does_not_validate_shapes_which_is_why_the_guard_exists():
    """Pins the reason `load_params_3tuple(strict=True)` cannot be removed.

    from_bytes with a WRONG 26-D template returns successfully and yields the
    FILE's 33-D normalizer. Nothing raises. brax does not check either, so the
    mismatch would surface only inside the first eval, after the GPU is
    allocated.
    """
    wrong = build_template(26, 7, NF)
    loaded = serialization.from_bytes(wrong, open(require(L0), "rb").read())
    assert np.asarray(loaded["0"].mean).shape == (33,)    # the file won, silently


@pytest.mark.parametrize(
    "obs,act,nf,needle",
    [
        (26, 7, NF, "normalizer"),
        (33, 3, NF, "policy"),
        (33, 7, {**NF, "policy_hidden_layer_sizes": [64, 64]}, "policy"),
        (33, 7, {**NF, "value_hidden_layer_sizes": [128, 128]}, "value"),
    ],
)
def test_architecture_mismatch_raises_with_a_useful_message(obs, act, nf, needle):
    with pytest.raises(ArchitectureMismatch) as e:
        load_params_3tuple(require(L0), obs, act, nf)
    msg = str(e.value)
    assert needle in msg
    assert "obs_include_velocity" in msg          # names the likely cause
    assert str(obs) in msg


def test_strict_false_is_an_explicit_opt_out():
    p = load_params_3tuple(require(L0), 26, 7, NF, strict=False)
    assert np.asarray(p[0].mean).shape == (33,)


def test_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_params_3tuple(os.path.join(POLICIES, "nope", "params.msgpack"), 33, 7, NF)


# --- 5. normalizer count rescaling (off by default) -------------------------

def test_rescale_keeps_mean_and_std_bit_identical():
    p = load_L0()
    out = rescale_normalizer_count(p[0], 1_000_000.0)
    assert np.array_equal(np.asarray(out.mean), np.asarray(p[0].mean))
    assert np.array_equal(np.asarray(out.std), np.asarray(p[0].std))
    assert float(np.asarray(out.count)) == 1_000_000.0
    assert float(np.asarray(p[0].count)) == 31948800.0    # source untouched


def test_rescale_none_is_a_no_op_object_identity():
    p = load_L0()
    assert rescale_normalizer_count(p[0], None) is p[0]


def test_rescale_rejects_nonpositive():
    p = load_L0()
    with pytest.raises(ValueError):
        rescale_normalizer_count(p[0], 0)


def test_frozen_normalizer_arithmetic_is_what_motivates_the_knob():
    """A carried count of 31.9M gives a fresh 20k batch ~0.06% weight."""
    count = 31_948_800.0
    batch = 2048 * 10
    assert batch / (count + batch) < 0.001


# --- 6. checksum + the whole ladder chain -----------------------------------

def test_params_sha256_is_stable_and_discriminating():
    a = params_sha256(load_L0())
    b = params_sha256(load_L0())
    assert a == b and len(a) == 64
    other = load_params_3tuple(require(LADDER[4]), 33, 7, NF)
    assert params_sha256(other) != a


@pytest.mark.parametrize("run_id", LADDER)
def test_chain_every_archived_rung_loads_and_acts(run_id):
    p = load_params_3tuple(require(run_id), 33, 7, NF)
    assert type(p[0]).__name__ == "RunningStatisticsState"
    inf = ppo_networks.make_inference_fn(_net())
    a, _ = inf((p[0], p[1]), deterministic=True)(jnp.zeros((1, 33)), jax.random.PRNGKey(0))
    assert np.isfinite(np.asarray(a)).all()
    assert np.asarray(a).shape == (1, 7)


# --- 7. the end-to-end proof: brax ITSELF installs the params ---------------
#
# Everything above replays brax's restore lines. This runs the real
# `ppo.train` and checks what brax actually put in the TrainingState at step 0.
# A tiny non-MuJoCo stub env carries UR3Pick's shapes, so there is no MJX, no
# GPU and no scene -- it completes in a couple of seconds on the Mac.

import functools  # noqa: E402

from brax.envs.base import Env, State  # noqa: E402
from brax.training.agents.ppo import train as ppo_train  # noqa: E402


class _ShapeEnv(Env):
    """Minimal brax Env with UR3Pick's obs/action shapes. No physics."""

    @property
    def observation_size(self):
        return 33

    @property
    def action_size(self):
        return 7

    @property
    def backend(self):
        return "generalized"

    def reset(self, rng):
        return State(pipeline_state=None, obs=jnp.zeros(33),
                     reward=jnp.float32(0), done=jnp.float32(0), metrics={})

    def step(self, state, action):
        obs = jnp.tanh(state.obs + jnp.sum(action))
        return state.replace(obs=obs, reward=jnp.sum(obs) * 0.01,
                             done=jnp.float32(0))


def _trees_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)
    )


def _tiny_train(restore_params=None):
    captured = {}

    def ppf(step, make_policy, params):
        captured.setdefault(int(step), params)

    nf = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256,) * 5,
    )
    # brax asserts `num_envs % device_count == 0` (ppo/train.py:374) and
    # `batch_size * num_minibatches % num_envs == 0` (:323). Hardcoding 2 makes
    # this test hardware-dependent: it passes on the HPC (one GPU -> one device)
    # and fails on a Mac running with XLA_FLAGS=--xla_force_host_platform_
    # device_count=N. Derive the sizes instead. At one device these are exactly
    # the original 2 / 2 / 2 / 64.
    n_dev = jax.local_device_count()
    num_envs = 2 * n_dev
    kw = dict(
        environment=_ShapeEnv(), num_envs=num_envs, batch_size=num_envs,
        num_minibatches=1, unroll_length=2, episode_length=4, num_evals=2,
        num_eval_envs=num_envs, num_timesteps=32 * num_envs,
        network_factory=nf, normalize_observations=True,
        policy_params_fn=ppf, seed=0,
    )
    if restore_params is not None:
        kw["restore_params"] = restore_params
    _, final, _ = ppo_train.train(**kw)
    return captured.get(0), final


def test_brax_itself_installs_the_warm_start_at_step_zero():
    src = load_L0()
    warm0, warm_final = _tiny_train(restore_params=src)

    # positive: what brax put in the TrainingState IS the checkpoint
    assert _trees_equal(warm0[1], src[1]), "policy not installed by brax"
    assert _trees_equal(warm0[2], src[2]), "value not installed by brax"
    assert params_sha256(warm0) == params_sha256(src)

    # training really ran -- so the step-0 match is not a zero-budget artifact
    assert not _trees_equal(warm_final[1], src[1])


def test_cold_start_is_the_negative_control():
    src = load_L0()
    cold0, _ = _tiny_train(restore_params=None)
    assert not _trees_equal(cold0[1], src[1])
    assert params_sha256(cold0) != params_sha256(src)
