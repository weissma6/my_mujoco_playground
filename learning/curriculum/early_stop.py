"""Per-rung early stopping for the curriculum ladder.

Pure Python on purpose: no jax, no brax, no wandb, no mujoco. This module is the
one piece of the curriculum that must be fully testable on the MacBook, where
MJX may never run.

WHY AN EXCEPTION IS NEEDED TO ACT ON THIS
-----------------------------------------
brax 0.13.0's PPO trainer runs a fixed budget and asserts it was consumed --
`brax/training/agents/ppo/train.py:750-755` raises AssertionError when
`total_steps < num_timesteps` -- and it reads the return value of neither
`progress_fn` nor `policy_params_fn`. There is no "stop here" hook. Raising
`ConvergedSignal` from inside `progress_fn` is the only way out of that loop.

That is safe because of the callback ordering inside one eval iteration:

    train.py:727   policy_params_fn(current_step, make_policy, params)
    train.py:748   progress_fn(current_step, metrics)

Both fire at the same `current_step`, and `metrics` are the result of evaluating
exactly those `params` (train.py:742-745). So when the tracker decides to stop at
eval k, the params already stashed by `policy_params_fn` are the ones that earned
the score being judged.

DIVISION OF LABOUR
------------------
`PatienceTracker.update()` returns a bool and never raises. The caller decides
whether to turn that into control flow. That keeps the stop *policy* testable
without exception plumbing, and keeps the *mechanism* (ConvergedSignal) in the
one place that talks to brax.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "ConvergedSignal",
    "PatienceTracker",
    "WindowedTrendTracker",
    "build_tracker",
]


class ConvergedSignal(Exception):
    """Raised from progress_fn to escape brax's fixed-budget eval loop.

    Carries the BEST observed eval, not the last one -- the last eval is by
    definition a non-improvement (that is why we are stopping), so reporting it
    would understate what the rung achieved.
    """

    def __init__(
        self,
        step: int,
        best_reward: float,
        best_step: int,
        reason: str,
    ) -> None:
        self.step = int(step)
        self.best_reward = float(best_reward)
        self.best_step = int(best_step)
        self.reason = str(reason)
        super().__init__(
            f"converged at step {self.step}: {self.reason} "
            f"(best {self.best_reward:.4g} at step {self.best_step})"
        )


class PatienceTracker:
    """Relative-improvement patience on a scalar eval metric.

    An eval counts as an improvement when

        reward > best + abs(best) * min_delta

    One formula covers both signs. For ``best > 0`` it is exactly
    ``best * (1 + min_delta)``; for ``best < 0`` it correctly asks the reward to
    move *up* by ``min_delta`` of the magnitude (best=-100, min_delta=0.02 ->
    threshold -98). Writing it as ``best * (1 + min_delta)`` would invert in the
    negative regime and demand the reward get *worse*, which is why the
    magnitude form is used instead. UR3Pick reward is positive in practice, but
    it starts near zero and a sign flip early in training must not silently
    disable the stop rule.

    `patience` consecutive non-improvements past the `min_steps` floor -> stop.
    The floor exists because UR3Pick reliably only learns reach/grasp by ~5M
    steps; without it a flat early curve would stop every rung before it starts.

    `count_stale_before_floor=False` (the default) additionally refuses to even
    COUNT non-improvements below the floor. Without that the floor only delays
    the verdict rather than deferring the evidence: a rung whose first seven
    evals are flat -- the normal shape -- would cross the floor with patience
    already spent and stop at exactly `min_steps`. See the test of the same
    name.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.02,
        min_steps: int = 6_000_000,
        metric: str = "eval/episode_reward",
        count_stale_before_floor: bool = False,
    ) -> None:
        if int(patience) < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if float(min_delta) < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if int(min_steps) < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_steps = int(min_steps)
        self.metric = str(metric)
        self.count_stale_before_floor = bool(count_stale_before_floor)

        # The bar: updated only on a *significant* improvement (Keras
        # EarlyStopping semantics). A creeping sub-min_delta rise must not keep
        # resetting patience, or a flat-but-noisy tail never stops.
        self.best_reward: Optional[float] = None
        self.best_step: Optional[int] = None

        # The true running max, for reporting. Diverges from the bar whenever a
        # rise was real but smaller than min_delta.
        self.max_reward: Optional[float] = None
        self.max_step: Optional[int] = None

        self.strikes = 0
        self.stopped = False
        self.reason: Optional[str] = None
        self.history: List[Tuple[int, float]] = []

    def improvement_threshold(self) -> float:
        """The value an eval must exceed to reset patience."""
        if self.best_reward is None:
            return float("-inf")
        return self.best_reward + abs(self.best_reward) * self.min_delta

    def update(self, step: int, metrics: Dict[str, Any]) -> bool:
        """Feed one eval. Return True when the rung should stop.

        Never returns True before `min_steps`. Never raises -- the caller turns
        a True into a ConvergedSignal.
        """
        if self.metric not in metrics:
            return False
        try:
            reward = float(metrics[self.metric])
        except (TypeError, ValueError):
            return False

        step = int(step)
        self.history.append((step, reward))

        if self.max_reward is None or reward > self.max_reward:
            self.max_reward = reward
            self.max_step = step

        if reward > self.improvement_threshold():
            self.best_reward = reward
            self.best_step = step
            self.strikes = 0
        else:
            # Only accrue staleness at or past the floor (default). Otherwise a
            # flat early curve -- which is the NORMAL shape here, UR3Pick only
            # learns reach/grasp by ~5M steps -- arrives at the floor with
            # patience already spent and the rung dies at exactly min_steps.
            # At num_evals=30 over 24M (~828k/eval) the 6M floor is eval ~7, so
            # a patience of 5 would be exhausted before training had begun.
            # Gating guarantees min_steps + patience*eval_interval instead.
            if self.count_stale_before_floor or step >= self.min_steps:
                self.strikes += 1

        if self.strikes >= self.patience and step >= self.min_steps:
            self.stopped = True
            self.reason = (
                f"{self.strikes} consecutive evals without a "
                f">{self.min_delta:.1%} improvement"
            )
            return True
        return False

    def signal(self, step: int) -> ConvergedSignal:
        """Build the ConvergedSignal for a stop decision at `step`."""
        return ConvergedSignal(
            step=step,
            best_reward=self.max_reward if self.max_reward is not None else float("nan"),
            best_step=self.max_step if self.max_step is not None else -1,
            reason=self.reason or "converged",
        )

    def summary(self) -> Dict[str, Any]:
        """JSON-safe record of what the tracker saw. Logged with the rung."""
        return {
            "strategy": "patience",
            "metric": self.metric,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_steps": self.min_steps,
            "count_stale_before_floor": self.count_stale_before_floor,
            "stopped": self.stopped,
            "reason": self.reason,
            "strikes": self.strikes,
            "best_reward": self.max_reward,
            "best_step": self.max_step,
            "n_evals": len(self.history),
            "last_step": self.history[-1][0] if self.history else None,
        }

    def diagnostics(self) -> Dict[str, Any]:
        """JSON-safe snapshot of live tracker state, meant for per-eval logging.

        Unlike `summary()` (the final record written once a rung ends), this is
        cheap to call every eval so the stop decision is visible on the W&B
        curve while training is still running, not just after the fact.
        """
        return {
            "strategy": "patience",
            "strikes": self.strikes,
            "improvement_threshold": self.improvement_threshold(),
            "best_reward": self.best_reward,
            "max_reward": self.max_reward,
            "n_evals": len(self.history),
        }


class WindowedTrendTracker:
    """Windowed-mean-trend early stopping on a scalar eval metric.

    `PatienceTracker`'s running-max ratchet has a real failure mode on this
    project's curves: measured eval-to-eval noise is 6.5% median on L3 and
    4.7% on L4 -- 2-3x the 2% `min_delta` -- so a single noise spike sets a
    `best_reward` bar the curve may never clear again by chance alone. On
    SLURM job 50874 that killed L3 while it was still climbing (eval reward
    5356 -> 6613) and killed L4 on a 0.03% near-miss (6123.0 against a
    6125.1 threshold). Both are false stops: the policy had not converged,
    a single lucky/unlucky eval had.

    This tracker compares two whole *windows* of evals instead of one eval
    against one running peak, so a single spike (in either direction) moves
    a mean by only `1/window` instead of setting or breaking a hard bar:

        gain = (mean(last N evals) - mean(N evals before those)) / max(abs(prior_mean), tiny)

    Same sign-safe magnitude form as `PatienceTracker.improvement_threshold`,
    for the same reason: with `prior_mean < 0` (not expected for UR3Pick
    reward, but not assumed away either) this still asks for the reward to
    move up by `min_delta` of the magnitude rather than inverting.

    A window counts as a non-improvement when `gain <= min_delta`. `patience`
    consecutive non-improving windows -> stop. Windows slide by one eval, not
    by `window` -- every new eval re-draws both the "recent" and "prior"
    windows, so a stop can fire (or be averted) on any single new point, not
    only every `window` evals.

    Needs `2 * window` evals of history before it can compare two windows at
    all; `update()` returns False unconditionally below that, same as
    `PatienceTracker` returning False before its first eval sets a bar.

    `min_steps` gates evidence, not just the verdict, exactly like
    `PatienceTracker`'s default `count_stale_before_floor=False`: a
    non-improving window below the floor is not counted as a strike at all,
    it is simply not evaluated. Unlike `PatienceTracker`, there is no
    `count_stale_before_floor` knob here -- it is not re-exposed by design
    (see the module Gotchas in the curriculum-v2 plan): this tracker is new,
    so there is no legacy "count everything" behaviour to preserve, and a
    second knob doing the same job as `PatienceTracker`'s would just be a
    second thing to keep in sync.

    `signal()` reports the true observed peak (`max_reward` / `max_step`),
    never a window mean -- a window mean is an average of an average and
    would understate what the rung actually reached. This is the same
    peak-not-last-eval contract as `PatienceTracker.signal()`.
    """

    def __init__(
        self,
        window: int = 4,
        patience: int = 3,
        min_delta: float = 0.02,
        min_steps: int = 6_000_000,
        metric: str = "eval/episode_reward",
    ) -> None:
        if int(window) < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if int(patience) < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if float(min_delta) < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if int(min_steps) < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")

        self.window = int(window)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_steps = int(min_steps)
        self.metric = str(metric)

        # The true running max, for reporting -- see class docstring.
        self.max_reward: Optional[float] = None
        self.max_step: Optional[int] = None

        # Live window state, refreshed on every update() that has enough
        # history to compare; exposed verbatim via diagnostics().
        self.recent_mean: Optional[float] = None
        self.prior_mean: Optional[float] = None
        self.last_gain: Optional[float] = None

        self.strikes = 0
        self.stopped = False
        self.reason: Optional[str] = None
        self.history: List[Tuple[int, float]] = []

    def has_enough_history(self) -> bool:
        """Whether two full, non-overlapping windows exist yet."""
        return len(self.history) >= 2 * self.window

    def update(self, step: int, metrics: Dict[str, Any]) -> bool:
        """Feed one eval. Return True when the rung should stop.

        Never returns True before `min_steps` or before `2 * window` evals
        of history exist. Never raises -- the caller turns a True into a
        ConvergedSignal, same contract as `PatienceTracker.update`.
        """
        if self.metric not in metrics:
            return False
        try:
            reward = float(metrics[self.metric])
        except (TypeError, ValueError):
            return False

        step = int(step)
        self.history.append((step, reward))

        if self.max_reward is None or reward > self.max_reward:
            self.max_reward = reward
            self.max_step = step

        if not self.has_enough_history():
            return False

        window = self.window
        recent = [r for _, r in self.history[-window:]]
        prior = [r for _, r in self.history[-2 * window : -window]]
        recent_mean = sum(recent) / window
        prior_mean = sum(prior) / window
        tiny = 1e-8
        gain = (recent_mean - prior_mean) / max(abs(prior_mean), tiny)

        self.recent_mean = recent_mean
        self.prior_mean = prior_mean
        self.last_gain = gain

        # Gate EVIDENCE at the floor, not just the verdict -- see class
        # docstring and PatienceTracker's count_stale_before_floor=False.
        if step < self.min_steps:
            return False

        if gain <= self.min_delta:
            self.strikes += 1
        else:
            self.strikes = 0

        if self.strikes >= self.patience:
            self.stopped = True
            self.reason = (
                f"{self.strikes} consecutive windows (size {window}) without a "
                f">{self.min_delta:.1%} relative gain"
            )
            return True
        return False

    def signal(self, step: int) -> ConvergedSignal:
        """Build the ConvergedSignal for a stop decision at `step`.

        Carries the true observed peak, never a window mean -- see class
        docstring.
        """
        return ConvergedSignal(
            step=step,
            best_reward=self.max_reward if self.max_reward is not None else float("nan"),
            best_step=self.max_step if self.max_step is not None else -1,
            reason=self.reason or "converged",
        )

    def summary(self) -> Dict[str, Any]:
        """JSON-safe record of what the tracker saw. Logged with the rung."""
        return {
            "strategy": "windowed",
            "metric": self.metric,
            "window": self.window,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_steps": self.min_steps,
            "stopped": self.stopped,
            "reason": self.reason,
            "strikes": self.strikes,
            "best_reward": self.max_reward,
            "best_step": self.max_step,
            "n_evals": len(self.history),
            "last_step": self.history[-1][0] if self.history else None,
        }

    def diagnostics(self) -> Dict[str, Any]:
        """JSON-safe snapshot of live tracker state, meant for per-eval logging.

        Cheap to call every eval -- see PatienceTracker.diagnostics.
        """
        return {
            "strategy": "windowed",
            "strikes": self.strikes,
            "recent_mean": self.recent_mean,
            "prior_mean": self.prior_mean,
            "last_gain": self.last_gain,
            "n_evals": len(self.history),
            "has_enough_history": self.has_enough_history(),
        }


def build_tracker(es_cfg: Dict[str, Any]) -> Union[PatienceTracker, WindowedTrendTracker]:
    """Construct the configured early-stop tracker from an `early_stop` dict.

    Dispatches on a popped `'strategy'` key, defaulting to `'patience'` so
    every config written before `WindowedTrendTracker` existed resolves to
    exactly today's behaviour -- no ladder spec needs to change to keep
    working. Copies `es_cfg` first: the caller's dict (typically loaded once
    from JSON and reused across rungs) must not be mutated by this call.

    Raises ValueError on an unrecognized strategy rather than silently
    falling back to patience -- a typo'd 'strategy' key should fail loud at
    tracker construction, not quietly train with the wrong stop rule.
    """
    cfg = dict(es_cfg)
    strategy = cfg.pop("strategy", "patience")
    if strategy == "patience":
        return PatienceTracker(**cfg)
    if strategy == "windowed":
        return WindowedTrendTracker(**cfg)
    raise ValueError(
        f"unknown early_stop strategy {strategy!r}; expected 'patience' or 'windowed'"
    )
