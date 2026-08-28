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
