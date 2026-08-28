"""Source-level guards on run_experiment.py's curriculum contract.

AST-based rather than behavioural: run_experiment() needs wandb.init, a built
MJX env and a GPU, so it can never be executed on the Mac. The repo already
uses this pattern (defence/tests/test_defence.py). What is checked here is that
the *structure* the curriculum depends on is present and that the off-by-one
cannot silently return.

Run:
    python -m pytest learning/curriculum/tests/test_run_experiment_contract.py -q
"""

import ast
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
SRC = os.path.join(REPO, "learning", "notebooks", "run_experiment.py")


@pytest.fixture(scope="module")
def tree():
    with open(SRC, encoding="utf-8") as f:
        return ast.parse(f.read())


def find_func(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in run_experiment.py")


def src_of(node):
    return ast.dump(node)


# --- the off-by-one cannot come back ----------------------------------------

def test_policy_params_fn_no_longer_reads_the_stashed_reward(tree):
    """THE regression guard.

    policy_params_fn must not consult progress_fn's stashed reward: brax fires
    it BEFORE progress_fn for the same step (ppo/train.py:727 vs :748), so that
    read returns the PREVIOUS eval's number and best_params.msgpack ends up one
    eval past the peak.
    """
    fn = find_func(tree, "policy_params_fn")
    dumped = src_of(fn)
    assert "last_eval_reward" not in dumped, (
        "policy_params_fn reads last_eval_reward again -- that is the "
        "off-by-one this WP removed"
    )
    assert "best_ckpt" not in dumped


def test_policy_params_fn_delegates_to_the_recorder(tree):
    fn = find_func(tree, "policy_params_fn")
    calls = [
        n.func.attr
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "on_policy_params" in calls


def test_the_recorder_is_constructed_with_the_checkpoint_dir(tree):
    fn = find_func(tree, "run_experiment")
    ctor = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "BestParamsRecorder"
    ]
    assert len(ctor) == 1
    kw = {k.arg for k in ctor[0].keywords}
    assert {"ckpt_dir", "serialize"} <= kw


def test_best_publish_reads_the_recorder_not_a_stale_dict(tree):
    fn = find_func(tree, "run_experiment")
    dumped = src_of(fn)
    assert "best_params.msgpack" in dumped
    # the old module-level dict must be gone entirely
    assert "best_ckpt" not in dumped


# --- the progress hook the early stop will need -----------------------------

def test_make_progress_wandb_accepts_on_eval(tree):
    fn = find_func(tree, "_make_progress_wandb")
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "on_eval" in args


def test_on_eval_fires_after_the_wandb_log(tree):
    """Ordering matters: the eval that triggers an early stop must be logged
    before the exception unwinds, or it is lost from W&B."""
    inner = find_func(tree, "progress_wandb")
    body = inner.body
    log_i = next(
        i for i, n in enumerate(body)
        if "wandb" in src_of(n) and "log" in src_of(n)
    )
    on_eval_i = next(i for i, n in enumerate(body) if "on_eval" in src_of(n))
    assert on_eval_i > log_i, "on_eval must be called AFTER wandb.log"


def test_train_partial_wires_on_eval_to_the_recorder(tree):
    fn = find_func(tree, "run_experiment")
    dumped = src_of(fn)
    assert "on_eval" in dumped and "on_progress" in dumped


# --- the return value the curriculum driver needs ---------------------------

def test_run_experiment_returns_a_dict_not_none(tree):
    fn = find_func(tree, "run_experiment")
    assert fn.returns is not None
    assert "Dict" in ast.dump(fn.returns)
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, "run_experiment must return its result"


def test_result_carries_the_keys_the_driver_consumes(tree):
    fn = find_func(tree, "run_experiment")
    keys = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    required = {
        "params", "best_params", "best_reward", "best_step",
        "final_metrics", "stopped_early", "wandb_run_id",
        "observation_size", "action_size", "network_factory",
    }
    assert required <= keys, f"missing from the result dict: {sorted(required - keys)}"


def test_the_return_is_outside_the_finally_so_wandb_closes_first(tree):
    """`finally: run.finish()` must still run before the value is handed back."""
    fn = find_func(tree, "run_experiment")
    tries = [n for n in fn.body if isinstance(n, ast.Try)]
    assert tries, "expected the try/finally around training"
    for t in tries:
        for n in ast.walk(t):
            if isinstance(n, ast.Return):
                raise AssertionError("return must not sit inside the try/finally")


# --- existing callers must be unaffected ------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        "batch_runs/scripts/run_one_ur3.py",
        "batch_runs/scripts/run_one_panda.py",
        "batch_runs/scripts/run_one_ur10.py",
    ],
)
def test_existing_callers_still_call_by_keyword_and_ignore_the_return(path):
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        pytest.skip(f"{path} not present")
    t = ast.parse(open(full, encoding="utf-8").read())
    calls = [
        n for n in ast.walk(t)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "run_experiment"
    ]
    assert calls, f"no run_experiment call in {path}"
    for c in calls:
        assert not c.args, "callers pass cfg/out_dir by keyword"
        assert {k.arg for k in c.keywords} == {"cfg", "out_dir"}


def test_module_still_imports_on_this_machine():
    """The positive half of the portability lint."""
    from learning.notebooks.run_experiment import run_experiment  # noqa: F401


# --- WP4: the curriculum keys reach run_experiment, never the PPO validator --
#
# These are BEHAVIOURAL, not AST: _extract_ppo_overrides and
# apply_validated_overrides are pure module-level functions, so the exact
# failure that would otherwise surface only after SLURM has allocated the GPU
# is reproducible here on the Mac.

import importlib  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "learning", "notebooks"))
sys.path.insert(0, os.path.join(REPO, "batch_runs", "sweeps"))


@pytest.fixture(scope="module")
def rex():
    return importlib.import_module("run_experiment")


@pytest.fixture(scope="module")
def ladder():
    """The five DR-ladder rungs, from the one source of truth."""
    mod = importlib.import_module("gen_dr_ladder")
    assert len(mod._CONFIGS) == 5, f"expected 5 rungs, got {len(mod._CONFIGS)}"
    return mod._CONFIGS


@pytest.fixture(scope="module")
def ppo_base():
    from mujoco_playground.config import manipulation_params
    base = manipulation_params.brax_ppo_config("UR3Pick")
    try:
        return dict(base.to_dict())
    except AttributeError:
        return dict(base)


def _rung_cfg(config_id, overrides, *, warm=True, misspell=False):
    """A cfg shaped the way the WP5 driver will hand it over."""
    cfg = {
        "env_name": "UR3Pick",
        "run_id": f"Curr_{config_id}_s0",
        "seed": 0,
        "wandb_project": "UR3_pick_ppo",
        "wandb_group": "curriculum_test",
        "episode_length": 400,
        "obs_include_velocity": True,
        "num_timesteps": 24_000_000,
        "num_evals": 30,
        # the five curriculum keys WP4 reserved
        "curriculum_rung": config_id,
        "early_stop": {"patience": 5, "min_delta": 0.02, "min_steps": 6_000_000},
        "normalizer_count_reset": None,
        "warm_start_from": None if config_id == "L0_none" else "prev",
    }
    if warm:
        cfg["warm_start_params"] = ("normalizer", "policy", "value")
    if misspell:
        cfg["num_timestepz"] = 1
    cfg.update(overrides)
    return cfg


def test_overrides_validate_for_every_rung(rex, ladder, ppo_base):
    """The GPU-burning failure, caught on the Mac.

    apply_validated_overrides(strict=True) runs at run_experiment.py:1239 --
    after SLURM has handed out the GPU. A curriculum key missing from
    `reserved` raises there and wastes the whole allocation.
    """
    for config_id, overrides, _tags in ladder:
        cfg = _rung_cfg(config_id, overrides)
        ov = rex._extract_ppo_overrides(cfg)
        rex.apply_validated_overrides(dict(ppo_base), ov, strict=True)


def test_overrides_reject_a_misspelled_key(rex, ladder, ppo_base):
    """Negative control: the validator must still be armed."""
    config_id, overrides, _ = ladder[0]
    cfg = _rung_cfg(config_id, overrides, misspell=True)
    ov = rex._extract_ppo_overrides(cfg)
    with pytest.raises(ValueError, match="Unknown override keys"):
        rex.apply_validated_overrides(dict(ppo_base), ov, strict=True)


def test_every_curriculum_key_is_reserved(rex, ladder):
    """Each of the five keys individually, so a partial revert is caught.

    Asserting only on the full set would still pass if one key were dropped
    but another masked the failure.
    """
    config_id, overrides, _ = ladder[0]
    keys = ["warm_start_params", "warm_start_from", "early_stop",
            "normalizer_count_reset", "curriculum_rung"]
    for key in keys:
        cfg = _rung_cfg(config_id, overrides, warm=False)
        cfg[key] = cfg.get(key, "x")
        assert key not in rex._extract_ppo_overrides(cfg), (
            f"'{key}' leaked into the PPO overrides -- it will be rejected by "
            f"apply_validated_overrides(strict=True) after the GPU is allocated"
        )


def test_curriculum_keys_change_nothing_about_the_ppo_overrides(rex, ladder):
    """T2 no-op guard, behavioural half.

    A rung cfg WITH the curriculum keys must produce exactly the same PPO
    override dict as the same cfg without them.
    """
    for config_id, overrides, _tags in ladder:
        with_curr = _rung_cfg(config_id, overrides)
        without = {k: v for k, v in with_curr.items() if k not in (
            "warm_start_params", "warm_start_from", "early_stop",
            "normalizer_count_reset", "curriculum_rung")}
        assert rex._extract_ppo_overrides(with_curr) == \
               rex._extract_ppo_overrides(without)


# --- WP4: structural guards on the wiring itself ----------------------------

def _src():
    with open(SRC, encoding="utf-8") as f:
        return f.read()


def test_restore_params_is_never_passed_unconditionally():
    """T2 no-op guard, structural half.

    A no-curriculum run must hand brax the kwarg set it got before this change:
    `restore_params` absent entirely, not present-and-None. So the only place
    it may appear is as a key of the guarded `warm_kwargs` dict.
    """
    src = _src()
    for line in src.splitlines():
        if "restore_params" in line and not line.strip().startswith("#"):
            assert 'warm_kwargs["restore_params"]' in line, (
                f"restore_params reaches ppo.train outside warm_kwargs: {line!r}"
            )


def test_warm_kwargs_is_empty_unless_a_warm_start_is_configured():
    src = _src()
    assert "warm_kwargs = {}" in src
    i = src.index("warm_kwargs = {}")
    j = src.index('warm_kwargs["restore_params"]')
    between = src[i:j]
    assert "if warm_start_params is not None:" in between, (
        "warm_kwargs is populated without the warm-start guard"
    )


def test_train_partial_still_carries_the_pre_change_kwargs(tree):
    """The additive claim: nothing was removed from the ppo.train partial."""
    src = _src()
    i = src.index("train_fn = functools.partial(")
    block = src[i:src.index("stopped_early = False", i)]
    for kw in ("**ppo_params_overwrite", "**dr_kwargs", "network_factory=",
               "progress_fn=", "policy_params_fn=", "seed=seed",
               "**warm_kwargs"):
        assert kw in block, f"{kw} missing from the ppo.train partial"


def test_converged_signal_is_caught_around_the_training_call(tree):
    """The train_fn call sits in a try whose handler catches ConvergedSignal
    and flips stopped_early to True.

    Checked on the AST, not on source text: the assignment is written as a
    tuple unpack (`converged, stopped_early = sig, True`), so a substring test
    for "stopped_early = True" reports a false failure.
    """
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)
             if any("train_fn" == getattr(c.func, "id", None)
                    for c in ast.walk(n) if isinstance(c, ast.Call))]
    assert tries, "the train_fn(...) call is not inside a try block"

    handlers = [h for t in tries for h in t.handlers
                if getattr(h.type, "id", None) == "ConvergedSignal"]
    assert handlers, "no `except ConvergedSignal` around the training call"

    # stopped_early must be bound to True somewhere in that handler.
    flipped = False
    for h in handlers:
        for node in ast.walk(h):
            if not isinstance(node, ast.Assign):
                continue
            targets, values = node.targets[0], node.value
            pairs = (list(zip(targets.elts, values.elts))
                     if isinstance(targets, ast.Tuple) and isinstance(values, ast.Tuple)
                     else [(targets, values)])
            for tgt, val in pairs:
                if (getattr(tgt, "id", None) == "stopped_early"
                        and isinstance(val, ast.Constant) and val.value is True):
                    flipped = True
    assert flipped, "the ConvergedSignal handler never sets stopped_early = True"


def test_early_stop_raises_from_the_eval_hook():
    """The stop must be raised from on_eval, which _make_progress_wandb calls
    AFTER wandb.log -- otherwise the triggering eval is lost."""
    src = _src()
    i = src.index("def _on_eval(")
    block = src[i:src.index("train_fn = functools.partial(", i)]
    assert "tracker.update(" in block
    assert "raise tracker.signal(" in block
    assert "recorder.on_progress(" in block
    assert block.index("recorder.on_progress(") < block.index("tracker.update("), (
        "the recorder must be updated BEFORE the stop can raise, or the eval "
        "that triggered it is not in the recorder when the loop unwinds"
    )


def test_both_checksums_are_logged():
    """WP8 chains published_sha256 -> the successor's params_sha256_at_init."""
    src = _src()
    assert 'wandb.run.summary["curriculum/params_sha256_at_init"]' in src
    assert 'wandb.run.summary["curriculum/published_sha256"]' in src


def test_warm_start_params_never_reaches_wandb_config():
    """A 1.1 MB params tree must not be serialised into run metadata."""
    src = _src()
    i = src.index("def run_experiment(")
    j = src.index("run = wandb.init(", i)
    assert 'k != "warm_start_params"' in src[i:j], (
        "warm_start_params is still in cfg when wandb.init(config=cfg) runs"
    )
