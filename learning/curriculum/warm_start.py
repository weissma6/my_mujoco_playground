"""Load a published `params.msgpack` into the 3-tuple brax's `restore_params` wants.

WHAT brax EXPECTS
-----------------
`ppo.train(restore_params=...)` (brax 0.13.0, `ppo/train.py:243`) consumes its
argument at `:628-635` as an indexable 3-sequence:

    normalizer_params = restore_params[0]
    policy_params     = restore_params[1]
    value_params      = restore_params[2]   # when restore_value_fn (default True)

`ppo.train` *returns* exactly that 3-tuple (`train.py:754-767`), and
`run_experiment.py:1375` already serialises it straight to disk. So the file
this repo has always written is the file a warm start needs -- no new format.

Two things that are NOT restored, ever, regardless of what is passed:
`optimizer_state` (Adam moments, re-initialised at `train.py:610`) and
`env_steps` (reset at `:615`). A warm-started rung therefore begins with
zero-momentum updates at full learning rate against an already-competent
policy. That is the intended behaviour between curriculum stages -- stale Adam
moments from a different DR distribution would fight the new objective -- but
it does mean a transient dip is expected right after each handoff.

WHY THE ARCHITECTURE GUARD IS NOT OPTIONAL
------------------------------------------
`flax.serialization.from_bytes` does **not** validate shapes. Measured on this
repo's own checkpoint: handing it a deliberately wrong 26-D template still
returns successfully, and the restored normalizer has the FILE's shape (33,),
not the template's. The template contributes structure; the file's arrays
overwrite it wholesale.

Nothing raises at load time. brax does not validate `restore_params` either --
it just assigns. The mismatch surfaces minutes later inside the first
evaluation as

    ValueError: Incompatible shapes for broadcasting: shapes=[(128, 33), (26,)]

i.e. after SLURM has allocated the GPU and MJX has compiled. `load_params_3tuple`
therefore compares the loaded leaf shapes against the expected ones itself and
raises `ArchitectureMismatch` up front. Do not remove that check on the grounds
that "from_bytes would catch it" -- it demonstrably does not.
"""

import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from flax import serialization

__all__ = [
    "ArchitectureMismatch",
    "build_template",
    "load_params_3tuple",
    "rescale_normalizer_count",
    "params_sha256",
]

# Keys metadata.json / inference_config.json use for the network factory. Only
# these are forwarded to make_ppo_networks; anything else would be a silent
# behaviour change (activation, distribution_type and init_noise_std all alter
# the param tree and are never set anywhere in this repo).
_NF_KEYS = (
    "policy_hidden_layer_sizes",
    "value_hidden_layer_sizes",
    "policy_obs_key",
    "value_obs_key",
)


class ArchitectureMismatch(ValueError):
    """The checkpoint's parameter shapes do not match the expected network."""


def _nf_kwargs(network_factory: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in _NF_KEYS:
        if k in network_factory and network_factory[k] is not None:
            v = network_factory[k]
            out[k] = tuple(v) if isinstance(v, list) else v
    return out


def build_template(
    obs_dim: int,
    act_dim: int,
    network_factory: Mapping[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """The {'0','1','2'} template `flax.serialization.from_bytes` needs.

    Mirrors the loader already proven in
    `robots/UR3e/ur3_realrobot_dependencies.py:663-703`; deliberately not a
    second, divergent implementation.
    """
    net = ppo_networks.make_ppo_networks(
        observation_size=int(obs_dim),
        action_size=int(act_dim),
        preprocess_observations_fn=running_statistics.normalize,
        **_nf_kwargs(network_factory),
    )
    rng = jax.random.PRNGKey(seed)
    return {
        "0": running_statistics.init_state(
            jax.ShapeDtypeStruct((int(obs_dim),), jnp.float32)
        ),
        "1": net.policy_network.init(rng),
        "2": net.value_network.init(rng),
    }


def _leaf_shapes(tree: Any) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(jnp.shape(x)) for x in jax.tree_util.tree_leaves(tree))


def _compare(name: str, got: Any, want: Any, problems: list) -> None:
    gs, ws = _leaf_shapes(got), _leaf_shapes(want)
    if gs != ws:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(gs, ws)) if a != b]
        problems.append(
            f"  {name}: {len(gs)} arrays vs {len(ws)} expected"
            + (
                f"; first differing leaf #{diff[0][0]}: file {diff[0][1]} "
                f"vs expected {diff[0][2]}"
                if diff
                else ""
            )
        )


def load_params_3tuple(
    msgpack_path: str,
    obs_dim: int,
    act_dim: int,
    network_factory: Mapping[str, Any],
    strict: bool = True,
) -> Tuple[Any, Any, Any]:
    """Return `(normalizer_params, policy_params, value_params)`.

    Raises ArchitectureMismatch when the file's shapes disagree with the
    network implied by `obs_dim` / `act_dim` / `network_factory` -- because
    neither flax nor brax will.
    """
    if not os.path.exists(msgpack_path):
        raise FileNotFoundError(msgpack_path)

    template = build_template(obs_dim, act_dim, network_factory)
    with open(msgpack_path, "rb") as f:
        loaded = serialization.from_bytes(template, f.read())

    missing = [k for k in ("0", "1", "2") if k not in loaded]
    if missing:
        raise ArchitectureMismatch(
            f"{msgpack_path}: missing key(s) {missing}; expected a 3-tuple "
            f"serialised as {{'0','1','2'}}. Got keys {sorted(loaded)}."
        )

    if strict:
        problems: list = []
        _compare("normalizer", loaded["0"], template["0"], problems)
        _compare("policy", loaded["1"], template["1"], problems)
        _compare("value", loaded["2"], template["2"], problems)
        if problems:
            raise ArchitectureMismatch(
                f"checkpoint does not match the expected architecture\n"
                f"  file: {msgpack_path}\n"
                f"  expected obs_dim={obs_dim} act_dim={act_dim} "
                f"network_factory={dict(_nf_kwargs(network_factory))}\n"
                + "\n".join(problems)
                + "\n  Most likely cause: obs_include_velocity differs between "
                "the rungs (33-D when true, 26-D when false), or the "
                "network_factory changed. Warm-starting across either is not "
                "supported -- brax does not check, and the failure would "
                "otherwise appear only after the GPU is allocated."
            )

    return (loaded["0"], loaded["1"], loaded["2"])


def rescale_normalizer_count(normalizer_params: Any, new_count: float) -> Any:
    """Keep mean/std, reset the observation count so statistics can adapt again.

    OFF BY DEFAULT -- changing it changes the science, so it must be a
    deliberate act.

    `running_statistics.update` is a cumulative Welford statistic weighted by
    count. A published UR3Pick checkpoint carries a count equal to its total env
    steps (measured: 31,948,800 for the archived ladder runs), so a fresh batch
    of ~20k samples moves the running mean/std by ~0.06%. The normaliser
    therefore arrives at the next rung effectively frozen.

    That is mostly what we want: the restored policy keeps seeing observations
    scaled the way it was trained, which is what makes the transfer work at all.
    Resetting the statistics outright would break it. But at L4, `obs_noise`
    genuinely shifts the observation distribution and a frozen normaliser cannot
    track it. Rescaling the count -- keeping mean and std, lowering the inertia
    -- is the middle path.
    """
    if new_count is None:
        return normalizer_params
    if float(new_count) <= 0:
        raise ValueError(f"new_count must be > 0, got {new_count}")
    return normalizer_params.replace(
        count=jnp.asarray(float(new_count), dtype=jnp.asarray(normalizer_params.count).dtype)
    )


def params_sha256(params: Sequence[Any]) -> str:
    """Stable checksum of a params 3-tuple.

    Logged at rung start and on publish so the curriculum's handoff chain is
    verifiable from W&B alone -- otherwise "rung N+1 started from rung N" is an
    unfalsifiable claim after the fact.
    """
    import hashlib

    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        arr = jax.numpy.asarray(leaf)
        h.update(str(arr.shape).encode())
        h.update(jax.numpy.asarray(arr, dtype=jnp.float32).tobytes())
    return h.hexdigest()


def load_metadata(policy_dir: str) -> Dict[str, Any]:
    """Read obs/act/network_factory from a downloaded policy directory.

    Accepts either the downloader's `metadata.json` (`obs_dim`/`action_dim`) or
    a run's `inference_config.json` (`observation_size`/`action_size`).
    """
    for name, obs_k, act_k in (
        ("metadata.json", "obs_dim", "action_dim"),
        ("inference_config.json", "observation_size", "action_size"),
    ):
        p = os.path.join(policy_dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            return {
                "obs_dim": int(m[obs_k]),
                "act_dim": int(m[act_k]),
                "network_factory": m.get("network_factory") or {},
            }
    raise FileNotFoundError(
        f"no metadata.json or inference_config.json in {policy_dir}"
    )
