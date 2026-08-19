"""Source-level guards for defence/record_pick_loop.py.

SAME CONTRACT AS test_defence.py: this module is NEVER imported. It pulls in
Linux-only hardware deps (vrpn.so, the Hand-E XML-RPC client) that do not
exist on the Mac, so every check here is py_compile + ast on the source text.

What these guard is the handful of things that turn a working one-shot script
into a broken loop -- the between-cycle state that record_real_rollout.py
never had to think about because it exits after one episode.
"""

import ast
import os
import py_compile

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFENCE_DIR = os.path.dirname(_THIS_DIR)
SRC_PATH = os.path.join(_DEFENCE_DIR, "record_pick_loop.py")


@pytest.fixture(scope="module")
def src():
  with open(SRC_PATH, encoding="utf-8") as f:
    return f.read()


@pytest.fixture(scope="module")
def tree(src):
  return ast.parse(src)


def _func(tree, name):
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == name:
      return node
  raise AssertionError(f"{name}() not found in record_pick_loop.py")


def _calls(node):
  """Every called attribute/name inside `node`, as dotted strings."""
  out = []
  for n in ast.walk(node):
    if isinstance(n, ast.Call):
      f = n.func
      if isinstance(f, ast.Attribute):
        base = f.value.id if isinstance(f.value, ast.Name) else "?"
        out.append(f"{base}.{f.attr}")
      elif isinstance(f, ast.Name):
        out.append(f.id)
  return out


def test_compiles():
  py_compile.compile(SRC_PATH, doraise=True)


# ---------------------------------------------------------------------------
# The config block: every knob visible at module level, each with a comment.
# ---------------------------------------------------------------------------

EXPECTED_CONSTANTS = [
    "CHECKPOINT", "ACTION_SCALE", "GRIPPER_SCALE", "EPISODE_LENGTH",
    "CONTROL_HZ", "SESSION_SECONDS", "REPOSITION_S", "DROP_SETTLE_S",
    "SETTLE_S", "DROP_TARGET", "HOME_BETWEEN", "ALPHA", "CONTROL_LAW",
    "LOOKAHEAD_TIME", "GAIN", "SERVOJ_A", "SERVOJ_V", "GRIPPER_TAU",
    "INIT_KEYFRAME", "START_FINGER", "FORCE_LIMIT_N", "FORCE_CONSECUTIVE",
    "FORCE_WARMUP", "MOCAP_STALE_S", "MAX_CONSEC_FAILS", "ROBOT_IP",
    "MOCAP_SERVER_IP", "MOCAP_BODY", "GRIPPER_PORT", "GRIPPER_SLAVE_ID",
    "GRIPPER_SPEED_PCT", "GRIPPER_FORCE_PCT",
]


def test_every_config_constant_is_module_level(tree):
  names = {n.targets[0].id for n in tree.body
           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
  missing = [c for c in EXPECTED_CONSTANTS if c not in names]
  assert not missing, f"config constants missing from module level: {missing}"


def test_every_config_constant_has_an_inline_comment(src, tree):
  """The whole point of the block: one line, one short comment, one view."""
  lines = src.splitlines()
  targets = {n.targets[0].id: n.lineno for n in tree.body
             if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in EXPECTED_CONSTANTS}
  bare = [name for name, lineno in targets.items()
          if "#" not in lines[lineno - 1]]
  assert not bare, f"config constants with no inline comment: {bare}"


def test_config_block_is_one_screenful(tree):
  """All knobs in a single contiguous block you can take in at a glance.

  Uses the AST rather than string matching: the docstring mentions several of
  these names at column 0 and a text scan picks those up as false hits.
  """
  linenos = [n.lineno for n in tree.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in EXPECTED_CONSTANTS]
  assert linenos, "no config constants found"
  span = max(linenos) - min(linenos)
  assert span < 60, f"config block spans {span} lines; keep it scannable"


def test_start_finger_matches_the_seed_keyframe(src, tree):
  """START_FINGER must be task_home's per-finger qpos (0.0125).

  run_policy_loop seeds its integrator from INIT_KEYFRAME. If the constant we
  command the hardware to disagrees with that keyframe, every cycle starts
  with the estimate and the physical gripper apart -- the b12f2bc defect.
  """
  consts = {n.targets[0].id: n.value for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
  assert consts["INIT_KEYFRAME"].value == "task_home"
  assert consts["START_FINGER"].value == pytest.approx(0.0125)


def test_no_argparse(tree):
  """Config is the constants block, not flags -- that was the whole request."""
  assert "argparse" not in {
      a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
      for a in n.names
  }


# ---------------------------------------------------------------------------
# The five between-cycle invariants.
# ---------------------------------------------------------------------------

def test_prepare_next_cycle_recommands_the_gripper(tree, src):
  """THE RE-SEED. The load-bearing line of the whole file."""
  fn = _func(tree, "prepare_next_cycle")
  assert "gripper.command" in _calls(fn), (
      "prepare_next_cycle must command the gripper back to START_FINGER; "
      "without it the drop leaves the fingers at 0.0 while run_policy_loop "
      "seeds the estimate at 0.0125 (the b12f2bc defect)")
  seg = ast.get_source_segment(src, fn)
  assert "START_FINGER" in seg, "the re-command must use START_FINGER"


def test_prepare_next_cycle_resets_t0(tree):
  """t0 is set on the first armed receive_feedback and never again."""
  fn = _func(tree, "prepare_next_cycle")
  assigned = {
      f"{t.value.id}.{t.attr}"
      for n in ast.walk(fn) if isinstance(n, ast.Assign)
      for t in n.targets
      if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
  }
  for attr in ("robot.t0_unix", "robot.t0_perf", "robot.t0_monotonic"):
    assert attr in assigned, f"{attr} must be reset between cycles"


def test_prepare_next_cycle_does_not_touch_loop_owned_state(tree, src):
  """run_policy_loop resets these itself as of b12f2bc -- doing it here too
  would fight it, and setting _gripper_ctrl to 0.0 would restore the bug."""
  seg = ast.get_source_segment(src, _func(tree, "prepare_next_cycle"))
  for attr in ("_gripper_ctrl", "_arm_ctrl", "_finger_pos_est",
               "_prev_finger_pos_est"):
    assert f"robot.{attr} =" not in seg, (
        f"{attr} is owned by run_policy_loop; do not assign it here")


def test_force_guard_is_rearmed_every_cycle(tree):
  """Clears _fg_tripped/_fg_peak_n/wrench_log; otherwise cycle 2 inherits
  cycle 1's trip and wrench rows."""
  assert "robot.arm_force_guard" in _calls(_func(tree, "run_cycle"))


def _call_lineno(fn, dotted):
  """Line of the first call to `dotted` inside fn. AST, not string search --
  the surrounding comments mention these names too."""
  for n in ast.walk(fn):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
      base = n.func.value.id if isinstance(n.func.value, ast.Name) else "?"
      if f"{base}.{n.func.attr}" == dotted:
        return n.lineno
  return None


def _disarm_lineno(fn):
  """Line of `<obj>._fg_armed = False` inside fn, if present."""
  for n in ast.walk(fn):
    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) \
        and n.value.value is False:
      for t in n.targets:
        if isinstance(t, ast.Attribute) and t.attr == "_fg_armed":
          return n.lineno
  return None


def test_preflight_read_precedes_arming(tree):
  """receive_feedback returns early while disarmed, so the pre-flight read
  must come first or t0 lands before the loop's first tick."""
  fn = _func(tree, "run_cycle")
  read = _call_lineno(fn, "robot.receive_feedback")
  arm = _call_lineno(fn, "robot.arm_force_guard")
  assert read is not None and arm is not None
  assert read < arm, "pre-flight read must precede arm_force_guard"


def test_guard_is_disarmed_before_the_preflight_read(tree):
  """_fg_armed has no counterpart that clears it -- it only goes False in
  __init__. From cycle 2 on the guard is still live at the pre-flight read,
  which would stamp t0 early and log a stray wrench row, so run_cycle must
  disarm explicitly first."""
  fn = _func(tree, "run_cycle")
  disarm = _disarm_lineno(fn)
  assert disarm is not None, (
      "run_cycle must disarm the force guard before its pre-flight read, or "
      "cycles 2+ stamp t0 from that read instead of the loop's first tick")
  assert disarm < _call_lineno(fn, "robot.receive_feedback")


def test_init_fk_model_is_setup_only(tree):
  """_keyframe_gripper_seed raises without it, so it must run once in setup --
  and NOT per cycle, which would rebuild the MuJoCo model every 12 s."""
  assert "robot.init_fk_model" in _calls(_func(tree, "setup_once"))
  assert "robot.init_fk_model" not in _calls(_func(tree, "run_cycle"))
  assert "robot.init_fk_model" not in _calls(_func(tree, "prepare_next_cycle"))


# ---------------------------------------------------------------------------
# The session loop.
# ---------------------------------------------------------------------------

def test_session_loop_is_bounded_by_session_seconds(tree, src):
  fn = _func(tree, "main")
  whiles = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
  assert whiles, "main() must contain the session while-loop"
  assert any("SESSION_SECONDS" in ast.get_source_segment(src, w.test)
             for w in whiles), "the session loop must test SESSION_SECONDS"


def test_keyboard_interrupt_is_reraised(tree):
  """Ctrl-C must actually stop the session, not be swallowed as a failed
  cycle (run_gap_protocol's rule)."""
  fn = _func(tree, "main")
  names = [h.type.id for n in ast.walk(fn) if isinstance(n, ast.Try)
           for h in n.handlers
           if h.type is not None and isinstance(h.type, ast.Name)]
  assert "KeyboardInterrupt" in names
  reraised = [h for n in ast.walk(fn) if isinstance(n, ast.Try)
              for h in n.handlers
              if isinstance(h.type, ast.Name)
              and h.type.id == "KeyboardInterrupt"
              and any(isinstance(b, ast.Raise) for b in h.body)]
  assert reraised, "KeyboardInterrupt must be re-raised, not swallowed"


def test_a_failed_cycle_does_not_kill_the_session(tree, src):
  """One bad grasp must not end a continuous demo."""
  seg = ast.get_source_segment(src, _func(tree, "main"))
  assert "except Exception" in seg
  assert "FAILED" in seg, "a failed cycle should be logged, not silent"


def test_prepare_next_cycle_runs_after_a_failed_cycle(tree):
  """It must sit OUTSIDE the per-cycle try, or a failure skips the re-seed and
  poisons every cycle that follows.

  Structural check: prepare_next_cycle must be a DIRECT statement of the
  session while-body, a sibling of the try -- not nested inside it. Walking
  transitively would also match the outer try that wraps the whole loop.
  """
  fn = _func(tree, "main")
  whiles = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
  assert whiles, "main() must contain the session while-loop"
  loop = whiles[0]

  # The try that DIRECTLY wraps run_cycle: one of the while-body statements.
  per_cycle_try = [s for s in loop.body
                   if isinstance(s, ast.Try) and "run_cycle" in _calls(s)]
  assert per_cycle_try, "expected a try wrapping run_cycle in the loop body"
  for t in per_cycle_try:
    assert "prepare_next_cycle" not in _calls(t), (
        "prepare_next_cycle must not be inside the per-cycle try -- a failed "
        "cycle would skip the gripper re-seed and poison every cycle after it")

  direct = [c for s in loop.body for c in _calls(s)
            if not isinstance(s, ast.Try)]
  assert "prepare_next_cycle" in direct, (
      "prepare_next_cycle must be a direct statement of the loop body")


# ---------------------------------------------------------------------------
# Reuse and artifacts.
# ---------------------------------------------------------------------------

def test_helpers_are_imported_not_reimplemented(tree):
  """Thin-driver rule: the force guard, scale resolution, timing assertion and
  CSV writer all come from record_real_rollout."""
  imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
              and n.module == "record_real_rollout" for a in n.names}
  for name in ("_ForceGuardedUR3", "_resolve_scales", "_assert_trained_timing",
               "_build_states_csv", "SCHEMA_VERSION"):
    assert name in imported, f"{name} should be reused, not redefined"
  local = {n.name for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
  assert not (local & imported), "imported helpers must not be shadowed"


def test_trained_timing_is_asserted(tree, src):
  """Deploy ctrl_dt/episode_length must equal the trained values."""
  seg = ast.get_source_segment(src, _func(tree, "setup_once"))
  assert "_assert_trained_timing" in seg
  assert "EPISODE_LENGTH" in seg


def test_each_cycle_writes_a_replayable_run(tree, src):
  """Per-cycle dir needs manifest.json + real_states.csv, the two files
  render_sim_rollout.py reads."""
  seg = ast.get_source_segment(src, _func(tree, "run_cycle"))
  assert "manifest.json" in seg
  assert "_build_states_csv" in seg
  assert "SCHEMA_VERSION" in seg, "per-cycle manifest must be defence/1"


def test_session_json_is_written_even_when_interrupted(tree, src):
  """The summary must survive Ctrl-C: it is written after the try/finally,
  not inside the loop."""
  seg = ast.get_source_segment(src, _func(tree, "main"))
  assert "session.json" in seg
  assert "LOOP_SCHEMA_VERSION" in seg


# ---------------------------------------------------------------------------
# Portability (the repo's hardware-interchangeability rule).
# ---------------------------------------------------------------------------

def test_no_hardcoded_absolute_paths(src):
  for needle in ("/Users/", "/home/", "/mnt/", "/scratch/"):
    assert needle not in src, f"hardcoded absolute path {needle!r}"


def test_paths_resolve_from_repo_root(tree):
  names = {n.targets[0].id for n in tree.body
           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
  assert "REPO_ROOT" in names and "_THIS_DIR" in names
